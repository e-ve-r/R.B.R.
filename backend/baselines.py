"""BM25 and dense retrieval for evaluation only; neither changes RBR."""
import math
import re
import time
from collections import Counter

from backend.retrieval import answer_was_found, cosine_similarity


def tokenize(text):
    return re.findall(r'\w+', text.lower())


def make_passages(document, chunk_words=256, overlap_words=48):
    words = document.split()
    passages = []
    for start in range(0, len(words), chunk_words - overlap_words):
        end = min(start + chunk_words, len(words))
        passages.append(' '.join(words[start:end]))
        if end == len(words):
            break
    return passages


def bm25_scores(question, passages):
    frequencies = []
    document_frequencies = Counter()
    lengths = []
    for passage in passages:
        tokens = tokenize(passage)
        frequencies.append(Counter(tokens))
        document_frequencies.update(set(tokens))
        lengths.append(len(tokens))
    average_length = sum(lengths) / max(1, len(lengths))
    query_terms = tokenize(question)
    scores = []
    for index in range(len(passages)):
        score = 0.0
        for term in query_terms:
            term_frequency = frequencies[index][term]
            if term_frequency == 0:
                continue
            matching_documents = document_frequencies[term]
            inverse_frequency = math.log(1 + (len(passages) - matching_documents + 0.5) / (matching_documents + 0.5))
            normalization = 1.5 * (0.25 + 0.75 * lengths[index] / max(1, average_length))
            score += inverse_frequency * term_frequency * 2.5 / (term_frequency + normalization)
        scores.append(score)
    return scores


def passage_baseline(document, question, client, settings, method):
    started = time.perf_counter()
    passages = make_passages(document)
    scores = []
    if method == 'bm25':
        scores = bm25_scores(question, passages)
    elif method == 'dense':
        query_vector = client.embed([question], is_query=True)[0]
        passage_vectors = client.embed(passages)
        for vector in passage_vectors:
            scores.append(cosine_similarity(query_vector, vector))
    else:
        raise ValueError('Unknown passage baseline: ' + method)

    def score_for_index(index):
        return scores[index]

    ranked_indices = sorted(range(len(passages)), key=score_for_index, reverse=True)
    selected_indices = sorted(ranked_indices[:8])
    selected_passages = []
    for index in selected_indices:
        selected_passages.append(passages[index])
    context = '\n\n--- Next passage ---\n\n'.join(selected_passages)
    answer, generation = client.generate(question, context)
    return {
        'method': method,
        'answer': answer,
        'found': answer_was_found(answer),
        'selected_passages': selected_indices,
        'metrics': client.metrics(time.perf_counter() - started),
    }
