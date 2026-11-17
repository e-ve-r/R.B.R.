import re
import unittest
from dataclasses import replace
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import app as application
from backend.baselines import bm25_scores, make_passages, passage_baseline
from backend.config import Settings
from backend.coverage_retrieval import CoverageBoundaryRetriever
from backend.ollama_client import OllamaClient, OllamaError
from backend.retrieval import Region, RecursiveBoundaryRetriever, answer_was_found, boundary_excerpts, cosine_similarity, direct_prompt, split_three
from benchmark.run import answer_scores


class FakeClient(OllamaClient):
    def __init__(self, settings):
        super().__init__(settings)
        self.requests = []

    def post(self, route, payload):
        self.requests.append((route, payload))
        if route == '/api/embed':
            vectors = []
            for text in payload['input']:
                vector = [1.0, 0.0]
                if text.startswith('search_document: left'):
                    vector = [-1.0, 0.0]
                vectors.append(vector)
            return {'embeddings': vectors}
        user_message = payload['messages'][1]['content']
        context = user_message.split('\n\nQuestion:', 1)[0]
        answer = 'NOT_FOUND'
        if re.search(r'\b2004\b', context):
            answer = '2004'
        return {'message': {'content': answer}, 'prompt_eval_count': len(user_message.split()), 'eval_count': 1}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()

    def document(self, size, position):
        words = []
        for index in range(size):
            words.append('word' + str(index))
        words[position] = '2004'
        return ' '.join(words)

    def test_uneven_partition_preserves_offsets(self):
        left, middle, right = split_three(Region(7, 17))
        self.assertEqual((left.start, left.end, middle.start, middle.end, right.start, right.end), (7, 10, 10, 13, 13, 17))

    def test_boundary_excerpts_preserve_positions(self):
        excerpts = boundary_excerpts('one two three four five six seven eight nine ten'.split(), Region(2, 10), 2)
        self.assertEqual(excerpts[0]['text'], 'three four')
        self.assertEqual(excerpts[1]['text'], 'nine ten')
        self.assertEqual(excerpts[1]['start_word'], 8)

    def test_original_recursion_order(self):
        result = RecursiveBoundaryRetriever(FakeClient(self.settings), self.settings).search(self.document(2700, 950), 'Find the code.')
        positions = []
        for step in result['trace']:
            positions.append((step['depth'], step['side']))
        self.assertEqual(positions, [(0, 'left'), (0, 'right'), (1, 'left')])
        self.assertEqual(result['answer'], '2004')

    def test_right_boundary_early_stop(self):
        result = RecursiveBoundaryRetriever(FakeClient(self.settings), self.settings).search(self.document(2700, 2600), 'Find the code.')
        self.assertEqual(result['metrics']['generation_calls'], 2)
        self.assertEqual(result['trace'][-1]['side'], 'right')

    def test_unseen_outer_interior_can_still_be_missed(self):
        text = self.document(21000, 2000)
        result = RecursiveBoundaryRetriever(FakeClient(self.settings), self.settings).search(text, 'Find the code.')
        direct = direct_prompt(text, 'Find the code.', FakeClient(self.settings), self.settings)
        self.assertFalse(result['found'])
        self.assertTrue(direct['found'])

    def test_small_core_needs_no_embedding(self):
        client = FakeClient(self.settings)
        result = RecursiveBoundaryRetriever(client, self.settings).search('The code is 2004.', 'Find the code.')
        self.assertTrue(result['found'])
        self.assertEqual(client.embedding_requests, 0)
        self.assertEqual(result['trace'][0]['side'], 'core')

    def test_maximum_depth_evaluates_whole_core(self):
        settings = replace(self.settings, maximum_depth=1)
        result = RecursiveBoundaryRetriever(FakeClient(settings), settings).search(self.document(9000, 4500), 'Find the code.')
        self.assertTrue(result['found'])
        self.assertEqual(result['trace'][-1]['region_words'], 3000)

    def test_similarity_does_not_change_traversal(self):
        text = 'left ' * 100 + '2004 ' + 'left ' * 899 + 'middle ' * 1000 + 'right ' * 1000
        result = RecursiveBoundaryRetriever(FakeClient(self.settings), self.settings).search(text, 'Find the code.')
        self.assertTrue(result['found'])
        self.assertEqual(result['trace'][0]['similarity'], 0.0)
        self.assertEqual(len(result['trace']), 1)

    def test_query_prefix_and_duplicate_embedding_cache(self):
        client = FakeClient(self.settings)
        client.embed(['question'], is_query=True)
        client.embed(['passage', 'passage'])
        client.embed(['passage'])
        self.assertEqual(client.requests[0][1]['input'], ['search_query: question'])
        self.assertEqual(client.requests[1][1]['input'], ['search_document: passage'])
        self.assertEqual(client.embedding_requests, 2)
        self.assertEqual(client.embedded_texts, 2)
        self.assertEqual(client.embedding_cache_hits, 1)

    def test_embedding_failure_is_explicit(self):
        client = FakeClient(self.settings)
        with patch.object(client, 'post', return_value={'embeddings': []}):
            with self.assertRaises(OllamaError):
                client.embed(['passage'])

    def test_context_warning_does_not_claim_exact_truncation(self):
        settings = replace(self.settings, context_tokens=10)
        result = direct_prompt('The safe code is 2004.', 'Find the code.', FakeClient(settings), settings)
        self.assertIn('may exceed', result['warning'])
        self.assertIn('does not report exact', result['warning'])

    def test_cosine_dimensions_and_zero_norm(self):
        self.assertAlmostEqual(cosine_similarity([3, 4], [3, 4]), 1.0)
        self.assertEqual(cosine_similarity([0, 0], [1, 2]), 0)
        with self.assertRaises(ValueError):
            cosine_similarity([1], [1, 2])

    def test_not_found_is_a_binary_heuristic(self):
        self.assertFalse(answer_was_found(' NOT_FOUND: absent'))
        self.assertFalse(answer_was_found(''))
        self.assertTrue(answer_was_found('2004'))

    def test_direct_uses_all_text_in_one_request(self):
        client = FakeClient(self.settings)
        text = self.document(3000, 1700)
        result = direct_prompt(text, 'Find the code.', client, self.settings)
        self.assertEqual(client.embedding_requests, 0)
        self.assertEqual(result['metrics']['generation_calls'], 1)
        self.assertIn(text, client.requests[0][1]['messages'][1]['content'])

    def test_baseline_ranking_and_generation(self):
        scores = bm25_scores('cobalt isotope', ['rainfall', 'the cobalt isotope'])
        self.assertGreater(scores[1], scores[0])
        for method in ['bm25', 'dense']:
            result = passage_baseline('The code is 2004.', 'Find the code.', FakeClient(self.settings), self.settings, method)
            self.assertTrue(result['found'])
            self.assertEqual(result['metrics']['generation_calls'], 1)

    def test_answer_scoring_and_overlap(self):
        self.assertEqual(answer_scores('The Blue Tile.', ['blue tile']), {'f1': 1.0, 'exact_match': 1.0})
        self.assertAlmostEqual(answer_scores('blue', ['blue tile'])['f1'], 2 / 3)
        words = ['background'] * 600
        words[250:253] = ['silver', 'orchard', 'gate']
        passages = make_passages(' '.join(words))
        self.assertIn('silver orchard gate', passages[0])
        self.assertIn('silver orchard gate', passages[1])


class ApiTests(unittest.TestCase):
    def make_client(self, settings):
        return FakeClient(settings)

    def test_compare_and_input_validation(self):
        client = TestClient(application.app)
        with patch.object(application, 'OllamaClient', side_effect=self.make_client):
            response = client.post('/api/compare', json={'text': 'The code is 2004.', 'question': 'Find the code.'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['results']['recursive_boundary']['answer'], '2004')
        self.assertEqual(response.json()['results']['direct']['answer'], '2004')
        self.assertEqual(client.post('/api/compare', json={'text': ' ', 'question': 'question'}).status_code, 422)
        self.assertEqual(client.post('/api/compare', json={}).status_code, 422)

    def test_embedding_failure_does_not_hide_the_direct_answer(self):
        class MissingEmbeddingClient(FakeClient):
            def post(self, route, payload):
                if route == '/api/embed':
                    raise OllamaError('Embedding model is unavailable.')
                return super().post(route, payload)

        client = TestClient(application.app)
        with patch.object(application, 'OllamaClient', MissingEmbeddingClient):
            response = client.post('/api/compare', json={
                'text': 'The code is 2004. ' + 'background ' * 1000,
                'question': 'Find the code.',
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['results']['recursive_boundary']['status'], 'error')
        self.assertEqual(response.json()['results']['direct']['answer'], '2004')

    def test_compare_recovers_key_in_discarded_outer_interior(self):
        words = ['history'] * 21000
        words[2000] = '2004'
        with patch.object(application, 'OllamaClient', side_effect=self.make_client):
            response = TestClient(application.app).post('/api/compare', json={
                'text': ' '.join(words), 'question': 'Find the code.',
            })
        result = response.json()['results']['recursive_boundary']
        self.assertEqual(result['answer'], '2004')
        self.assertTrue(result['fallback_used'])
        self.assertEqual(result['variant'], 'boundary_with_coverage')

    def test_concurrent_request_is_rejected(self):
        client = TestClient(application.app)
        application.comparison_lock.acquire()
        try:
            response = client.post('/api/compare', json={'text': 'text', 'question': 'question'})
        finally:
            application.comparison_lock.release()
        self.assertEqual(response.status_code, 429)


class CoverageTests(unittest.TestCase):
    history = (
        'The Mauryan empire expanded across the Indian subcontinent. '
        'Ashoka promoted administration and trade. Later dynasties supported '
        'literature mathematics architecture and regional kingdoms.'
    )

    def test_secret_between_history_paragraphs_at_multiple_positions(self):
        class KeyClient(FakeClient):
            def post(self, route, payload):
                if route == '/api/embed':
                    return super().post(route, payload)
                context = payload['messages'][1]['content'].split('\n\nQuestion:', 1)[0]
                answer = 'NOT_FOUND'
                if 'The secret key is IND-7xQ9-42.' in context:
                    answer = 'IND-7xQ9-42'
                return {'message': {'content': answer}}

        settings = Settings()
        paragraphs = [self.history] * 900
        # Includes both outer interiors, the center, document edges and a
        # sentence straddling a non-overlapping 256-word chunk boundary.
        for position in [0, 78, 137, 450, 762, 899]:
            with self.subTest(position=position):
                text = '\n\n'.join(paragraphs[:position] + ['The secret key is IND-7xQ9-42.'] + paragraphs[position:])
                result = CoverageBoundaryRetriever(KeyClient(settings), settings).search(text, 'What is the secret key?')
                self.assertEqual(result['answer'], 'IND-7xQ9-42')
                self.assertTrue(any('IND-7xQ9-42.' in excerpt['text'] for excerpt in result['answer_excerpts']))
                for step in result['trace']:
                    for excerpt in step['excerpts']:
                        self.assertEqual(excerpt['text'], ' '.join(text.split()[excerpt['start_word']:excerpt['end_word']]))

    def test_no_answer_examines_every_word(self):
        settings = Settings()
        text = self.history * 900
        result = CoverageBoundaryRetriever(FakeClient(settings), settings).search(text, 'Find the code.')
        self.assertFalse(result['found'])
        self.assertTrue(result['fallback_used'])
        self.assertEqual(result['coverage']['examined_words'], len(text.split()))
        self.assertTrue(any(step['side'] == 'fallback' for step in result['trace']))

    def test_overlap_recovers_sentence_crossing_window_boundary(self):
        class PhraseClient(FakeClient):
            def post(self, route, payload):
                if route == '/api/embed':
                    return super().post(route, payload)
                context = payload['messages'][1]['content'].split('\n\nQuestion:', 1)[0]
                answer = 'cobalt lotus' if 'secret key is cobalt lotus' in context else 'NOT_FOUND'
                return {'message': {'content': answer}}

        words = ['history'] * 21000
        words[2046:2051] = ['secret', 'key', 'is', 'cobalt', 'lotus']
        settings = Settings()
        result = CoverageBoundaryRetriever(PhraseClient(settings), settings).search(' '.join(words), 'What is the secret key?')
        self.assertEqual(result['answer'], 'cobalt lotus')
        self.assertEqual(result['trace'][-1]['side'], 'fallback')

    def test_small_context_and_depth_limited_core_use_bounded_views(self):
        settings = replace(Settings(), context_tokens=2048, maximum_depth=0)
        text = ('इतिहास ' * 5000) + '2004'
        retriever = CoverageBoundaryRetriever(FakeClient(settings), settings)
        result = retriever.search(text, 'Find the code.')
        self.assertTrue(result['found'])
        self.assertGreater(result['metrics']['generation_calls'], 1)
        for step in result['trace']:
            supplied = '\n\n[Text between these excerpts was not supplied.]\n\n'.join(excerpt['text'] for excerpt in step['excerpts'])
            self.assertLessEqual(len(supplied.encode('utf-8')), retriever.context_bytes)

    def test_boundary_answer_still_stops_early(self):
        settings = Settings()
        result = CoverageBoundaryRetriever(FakeClient(settings), settings).search('The code is 2004.', 'Find the code.')
        self.assertFalse(result['fallback_used'])
        self.assertEqual(result['metrics']['generation_calls'], 1)

    def test_fully_examined_small_document_does_not_repeat_generation(self):
        settings = Settings()
        result = CoverageBoundaryRetriever(FakeClient(settings), settings).search(self.history, 'Find the code.')
        self.assertFalse(result['found'])
        self.assertFalse(result['fallback_used'])
        self.assertEqual(result['metrics']['generation_calls'], 1)
        self.assertEqual(result['coverage']['examined_words'], result['coverage']['total_words'])

    def test_abstention_variants_do_not_stop_before_hidden_key(self):
        for abstention in ['NOT FOUND.', 'not-found', '`NOT_FOUND`', '"Not found in this passage."']:
            class VariantClient(FakeClient):
                def post(self, route, payload):
                    response = super().post(route, payload)
                    if route == '/api/chat' and response['message']['content'] == 'NOT_FOUND':
                        response['message']['content'] = abstention
                    return response

            with self.subTest(abstention=abstention):
                settings = Settings()
                words = ['history'] * 21000
                words[2000] = '2004'
                result = CoverageBoundaryRetriever(VariantClient(settings), settings).search(' '.join(words), 'Find the code.')
                self.assertTrue(result['fallback_used'])
                self.assertEqual(result['answer'], '2004')


if __name__ == '__main__':
    unittest.main()
