# Fresh LongBench pilot

Ten questions, seed 2026, sampled from 109 eligible 2,000–8,000-word contexts. Observed sample: 2,279–7,293 words. Models: llama3.2 and nomic-embed-text, Ollama 0.24.0, Apple M2, 16 GiB. No historical prototype scores are reused.

| Method | Mean F1 | Mean exact match | Mean answer calls | Median total seconds | Mean prompt tokens |
|---|---:|---:|---:|---:|---:|
| Original RBR | 21.03% | 0.00% | 2.10 | 10.81 | 2380.6 |
| Direct prompting | 16.95% | 0.00% | 1.00 | 31.72 | 6639.8 |
| BM25 | 22.49% | 0.00% | 1.00 | 13.42 | 3332.3 |
| Dense cosine | 26.18% | 0.00% | 1.00 | 15.29 | 3381.2 |

RBR had higher observed F1 and lower median total time than direct prompting in this sample. Dense retrieval had the highest observed F1. These ten examples do not establish a stable ranking. All methods use the same base generation instructions and per-call settings, but RBR can use multiple calls and its final core receives inspection notes. Total budgets differ.

Times include embedding and scoring as well as generation; they are affected by warm-up, model loading, service caches and local machine activity. F1 is reference-word overlap, not evidence accuracy. No method achieved exact match on this sample.

Paired bootstrap intervals for RBR minus each baseline (F1 percentage points):

- Direct prompting: 4.09 points; descriptive 95% interval [-16.03, 24.01].
- BM25: -1.45 points; descriptive 95% interval [-16.17, 11.32].
- Dense cosine: -5.15 points; descriptive 95% interval [-27.31, 15.84].

See `results.json` for predictions, IDs, settings, token counts, model digests and hashes. See `README.md` for the full protocol and limitations.
