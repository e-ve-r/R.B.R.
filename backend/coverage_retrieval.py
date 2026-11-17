"""Application RBR variant that searches text omitted by boundary views."""
import re
import time

from backend.baselines import bm25_scores
from backend.ollama_client import ANSWER_INSTRUCTIONS
from backend.retrieval import (
    Region, RecursiveBoundaryRetriever, answer_was_found, boundary_excerpts, join_excerpts,
)


class CoverageBoundaryRetriever(RecursiveBoundaryRetriever):
    def search(self, document, question):
        started = time.perf_counter()
        # UTF-8 bytes conservatively bound tokens; reserve room for the question,
        # system instructions, chat framing, and the requested output.
        self.context_bytes = (
            self.settings.context_tokens - self.settings.output_tokens - 256
            - len(ANSWER_INSTRUCTIONS.encode('utf-8')) - len(question.encode('utf-8'))
        )
        if self.context_bytes < 128:
            raise ValueError('The context window is too small for this question and output budget.')
        if self.settings.embedding_chunk_words < 1:
            raise ValueError('embedding_chunk_words must be positive.')
        result = super().search(document, question)
        result['variant'] = 'boundary_with_coverage'
        result['fallback_used'] = False
        if not result['found']:
            # Rank every overlapping window, including outer interiors. Ranking
            # changes inspection order only: there is no top-k coverage cutoff.
            regions = list(self.windows(Region(0, len(self.words))))
            inspected = [excerpt for step in self.trace for excerpt in step['excerpts']]
            regions = [region for region in regions if not any(
                excerpt['start_word'] <= region.start and excerpt['end_word'] >= region.end
                for excerpt in inspected
            )]
            texts = [' '.join(self.words[region.start:region.end]) for region in regions]
            scores = bm25_scores(question, texts)
            ranked = sorted(range(len(regions)), key=lambda index: scores[index], reverse=True)
            for index in ranked:
                result['fallback_used'] = True
                answer, excerpts = self.inspect(
                    [self.excerpt(regions[index])], 0, 'fallback', None, regions[index],
                )
                result.update(answer=answer, answer_excerpts=excerpts, found=answer_was_found(answer))
                if result['found']:
                    break
        # Count unique word positions actually supplied to generation, not words
        # merely embedded or the enclosing region size.
        ranges = sorted((excerpt['start_word'], excerpt['end_word'])
                        for step in self.trace for excerpt in step['excerpts'])
        covered = 0
        end = 0
        for start, stop in ranges:
            covered += max(0, stop - max(start, end))
            end = max(end, stop)
        result['coverage'] = {'examined_words': covered, 'total_words': len(self.words)}
        result['metrics'] = self.client.metrics(time.perf_counter() - started)
        result['warning'] = (
            'Answers are not independently verified. If boundary views return NOT_FOUND, '
            'overlapping passages are searched until an answer is returned or all text is examined.'
        )
        return result

    def excerpt(self, region):
        return {'start_word': region.start, 'end_word': region.end,
                'text': ' '.join(self.words[region.start:region.end])}

    def windows(self, region):
        start = region.start
        width = self.settings.embedding_chunk_words
        overlap = min(48, width // 4)
        while start < region.end:
            end = start
            byte_count = 0
            while end < min(start + width, region.end):
                next_bytes = len(self.words[end].encode('utf-8')) + (end > start)
                if byte_count + next_bytes > self.context_bytes:
                    break
                byte_count += next_bytes
                end += 1
            if end == start:
                raise ValueError('A single word exceeds the available model context window.')
            yield Region(start, end)
            if end == region.end:
                break
            start = end - min(overlap, (end - start) // 2)

    def inspect(self, excerpts, depth, side, similarity, region):
        answer, generation = self.client.generate(self.question, join_excerpts(excerpts))
        # Local models sometimes vary the requested sentinel's separator or
        # wrap it in quotes/Markdown. Such abstentions must not stop coverage.
        if re.match(r'NOT[\s_-]+FOUND\b', answer.strip(' \t\r\n`"\''), re.IGNORECASE):
            answer = 'NOT_FOUND'
        self.trace.append({
            'depth': depth, 'side': side,
            'region_start': region.start, 'region_end': region.end,
            'region_words': region.length(), 'similarity': similarity,
            'found': answer_was_found(answer), 'answer': answer,
            'excerpts': excerpts, 'generation': generation,
        })
        return answer, excerpts

    def evaluate(self, region, depth, side, similarity, is_core=False):
        excerpts = boundary_excerpts(self.words, region, self.settings.boundary_words)
        if is_core:
            excerpts = [self.excerpt(region)]
        if len(join_excerpts(excerpts).encode('utf-8')) <= self.context_bytes:
            return self.inspect(excerpts, depth, side, similarity, region)
        # A depth-limited core can still be huge. Inspect bounded windows instead
        # of sending an oversized core and allowing silent model truncation.
        answer, supplied = 'NOT_FOUND', []
        for excerpt in excerpts:
            for window in self.windows(Region(excerpt['start_word'], excerpt['end_word'])):
                answer, supplied = self.inspect([self.excerpt(window)], depth, side, similarity, window)
                if answer_was_found(answer):
                    return answer, supplied
        return answer, supplied
