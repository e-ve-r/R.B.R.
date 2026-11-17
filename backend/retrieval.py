"""The original recursive boundary algorithm, expressed with explicit ranges."""
import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    start: int
    end: int

    def length(self):
        return self.end - self.start


def answer_was_found(answer):
    normalized = answer.strip().upper()
    if not normalized:
        return False
    if normalized.startswith('NOT_FOUND'):
        return False
    return True


def cosine_similarity(first, second):
    if len(first) != len(second):
        raise ValueError('Embedding vectors have different dimensions.')
    dot_product = 0.0
    first_squared_norm = 0.0
    second_squared_norm = 0.0
    for first_value, second_value in zip(first, second):
        dot_product += first_value * second_value
        first_squared_norm += first_value * first_value
        second_squared_norm += second_value * second_value
    denominator = math.sqrt(first_squared_norm * second_squared_norm)
    if denominator == 0:
        return 0.0
    return dot_product / denominator


def split_three(region):
    third_length = region.length() // 3
    left = Region(region.start, region.start + third_length)
    middle = Region(region.start + third_length, region.start + 2 * third_length)
    right = Region(region.start + 2 * third_length, region.end)
    return left, middle, right


def boundary_excerpts(words, region, boundary_words):
    ranges = []
    if region.length() <= 2 * boundary_words:
        ranges.append(region)
    else:
        ranges.append(Region(region.start, region.start + boundary_words))
        ranges.append(Region(region.end - boundary_words, region.end))
    excerpts = []
    for current_range in ranges:
        excerpts.append({
            'start_word': current_range.start,
            'end_word': current_range.end,
            'text': ' '.join(words[current_range.start:current_range.end]),
        })
    return excerpts


def join_excerpts(excerpts):
    fragments = []
    for excerpt in excerpts:
        fragments.append(excerpt['text'])
    return '\n\n[Text between these excerpts was not supplied.]\n\n'.join(fragments)


class RecursiveBoundaryRetriever:
    def __init__(self, client, settings):
        self.client = client
        self.settings = settings
        self.words = []
        self.question = ''
        self.query_vector = None
        self.trace = []
        self.inspection_notes = []

    def search(self, document, question):
        started = time.perf_counter()
        self.words = document.split()
        self.question = question
        self.trace = []
        self.inspection_notes = []
        self.query_vector = None
        if len(self.words) > self.settings.minimum_region_words:
            self.query_vector = self.client.embed([question], is_query=True)[0]
        answer, excerpts = self.search_region(Region(0, len(self.words)), 0)
        return {
            'method': 'recursive_boundary',
            'answer': answer,
            'found': answer_was_found(answer),
            'trace': self.trace,
            'answer_excerpts': excerpts,
            'metrics': self.client.metrics(time.perf_counter() - started),
            'warning': 'An accepted answer is not independently verified. Unseen text can be discarded.',
        }

    def score_region(self, region):
        texts = []
        chunk_width = self.settings.embedding_chunk_words
        for start in range(region.start, region.end, chunk_width):
            end = min(start + chunk_width, region.end)
            texts.append(' '.join(self.words[start:end]))
        vectors = self.client.embed(texts)
        maximum_similarity = 0.0
        for vector in vectors:
            similarity = cosine_similarity(self.query_vector, vector)
            maximum_similarity = max(maximum_similarity, similarity)
        return maximum_similarity

    def evaluate(self, region, depth, side, similarity, is_core=False):
        excerpts = boundary_excerpts(self.words, region, self.settings.boundary_words)
        if is_core:
            excerpts = [{
                'start_word': region.start,
                'end_word': region.end,
                'text': ' '.join(self.words[region.start:region.end]),
            }]
        notes = ''
        if is_core:
            notes = '\n'.join(self.inspection_notes[-5:])
        answer, generation = self.client.generate(self.question, join_excerpts(excerpts), notes)
        found = answer_was_found(answer)
        self.trace.append({
            'depth': depth,
            'side': side,
            'region_start': region.start,
            'region_end': region.end,
            'region_words': region.length(),
            'similarity': similarity,
            'found': found,
            'answer': answer,
            'excerpts': excerpts,
            'generation': generation,
        })
        if not found:
            note = 'Depth ' + str(depth) + ', ' + side + ': the examined view returned NOT_FOUND.'
            self.inspection_notes.append(note)
        return answer, excerpts

    def search_region(self, region, depth):
        if region.length() <= self.settings.minimum_region_words or depth >= self.settings.maximum_depth:
            return self.evaluate(region, depth, 'core', None, is_core=True)
        left, middle, right = split_three(region)
        # The original algorithm scores both outer thirds before evaluating them.
        # Scores are diagnostic; neither score changes the retained middle third.
        left_similarity = self.score_region(left)
        right_similarity = self.score_region(right)
        answer, excerpts = self.evaluate(left, depth, 'left', left_similarity)
        if answer_was_found(answer):
            return answer, excerpts
        answer, excerpts = self.evaluate(right, depth, 'right', right_similarity)
        if answer_was_found(answer):
            return answer, excerpts
        return self.search_region(middle, depth + 1)


def direct_prompt(document, question, client, settings):
    started = time.perf_counter()
    answer, generation = client.generate(question, document)
    warning = None
    # Byte length is a conservative risk indicator, not an exact tokenizer.
    # Ollama reports processed tokens, not the original pre-truncation length.
    if len(document.encode('utf-8')) > settings.context_tokens:
        warning = (
            'The full text was sent in one request. It may exceed the model context. '
            'Ollama does not report exact input truncation; processed tokens are shown below.'
        )
    return {
        'method': 'direct',
        'answer': answer,
        'found': answer_was_found(answer),
        'trace': [],
        'answer_excerpts': [],
        'metrics': client.metrics(time.perf_counter() - started),
        'warning': warning,
    }
