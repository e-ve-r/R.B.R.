# Public-data benchmark

Dataset: English MultiFieldQA from [LongBench](https://aclanthology.org/2024.acl-long.172/). The downloader uses a public [JSONL mirror](https://huggingface.co/datasets/Leooyii/longbench/resolve/main/multifieldqa_en.jsonl), SHA-256 `0aac182fd317dcf6d74f8e1e0f3e61029407435346c2e0b3ff9fb45ae49c5c3f`. The upstream dataset is [zai-org/LongBench](https://huggingface.co/datasets/zai-org/LongBench). The file contains 150 examples. Do not redistribute source contexts without considering their upstream terms.

The inclusion rule selects contexts containing 2,000–8,000 whitespace-separated words. The checked-in pilot samples ten eligible questions using seed 2026. This is not a full benchmark or an official leaderboard score. The inclusion rule is based on words, not an exact tokenizer, so actual prompt token counts are retained and context truncation remains a limitation.

## Methods

- **RBR:** the original left/right boundary checks and recursive middle retention, with max-pooled Nomic scores recorded but not used for routing. Query embedding occurs once. Terminal cores include the last five inspection notes.
- **Direct:** full document, one answer call, no embedding or retrieval.
- **BM25:** k1=1.5, b=0.75, top eight overlapping 256-word passages, 48-word overlap.
- **Dense:** cosine similarity using the same Nomic model, top eight of the same passages.

BM25 and dense restore original passage order. All methods use the same answer instructions, model, seed, temperature, context setting and output limit. RBR's final-core inspection notes are an intrinsic difference. It can make multiple calls, so cumulative input and call budgets are not equal. The two single-call retrieval baselines are not modifications to RBR.

## Reproduce

```sh
python benchmark/download.py
python benchmark/run.py benchmark/data/multifieldqa_en.jsonl --examples 10 --seed 2026
```

Warm-up runs before the paired comparison. Method order is randomized per question. Each method receives a new client cache; duplicate passage embeddings may be reused within that method. No benchmark result from an earlier repository version is reused. The script checkpoints after every method and refuses to resume when the recorded code, dataset, models or settings differ.

Total method time includes embedding, lexical/dense scoring, and generation. It is neither a generation-only time nor complete HTTP application latency. Ollama load time, service caches, model switching and the local machine affect these measurements. `prompt_tokens` sums actual Ollama prompt counts over all calls, including repeated tokens and instructions. No tokenizer-equivalent input-length check is performed; Ollama does not expose exact original length before any truncation.

F1 and exact match use best-reference English normalization: lowercase, punctuation and article removal, then whitespace tokenization. Reference overlap is not evidence recall or factuality. No human support grading or passage-level relevance annotation is included. The bootstrap intervals are descriptive, with only ten paired examples; they do not establish superiority or correct for multiple comparisons.

The paper's table comes from `results.json` through `write_paper_results.py`. The full configuration and per-question predictions are retained so the reported numbers can be checked. Run a larger predefined study before making a general performance claim.
