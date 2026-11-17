# Recursive Boundary Retrieval

Compare answers from long documents using Recursive Boundary Retrieval (RBR) and direct prompting, with local models served by [Ollama](https://ollama.com/). This research project implements Veer Vikram Singh's RBR procedure, a coverage-aware application variant, and a reproducible four-method evaluation.

The React interface displays answers side by side, along with inspected passages, recursion traces, elapsed time, and token usage. The FastAPI backend accepts pasted text or text loaded through the interface from a UTF-8 `.txt` or `.md` file.

**The app uses coverage-aware RBR. The recorded benchmark evaluates original RBR.** The pilot results below do not measure the coverage fallback's accuracy or runtime.

[Quick start](#quick-start) · [How retrieval works](#how-retrieval-works) · [Benchmark results](#recorded-benchmark-results) · [Run the benchmark](#run-the-benchmark) · [Tests](#tests-and-validation) · [Research paper](#research-paper)

## Quick start

### Prerequisites

- Python 3.10 or newer; the test suite has been run locally with Python 3.13.
- Node.js 20.19.x or newer in the 20.x series, or Node.js 22.12 or newer, plus npm, as required by the installed Vite toolchain.
- Ollama running locally, with enough memory for the selected models and context window.

Clone the repository and enter the project directory:

```sh
git clone https://github.com/e-ve-r/R.B.R. RBR_project
cd RBR_project
```

Start Ollama using its application or `ollama serve`, then download both models:

```sh
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Start the backend

Run from the project root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell. Backend API documentation is available at `http://127.0.0.1:8000/docs`.

### Start the frontend

In a second terminal, from the project root:

```sh
cd frontend
npm ci
npm run dev
```

Open the URL printed by Vite, normally `http://127.0.0.1:5173`. Paste or load text, enter a question, and select **Compare answers**. Expand the trace to inspect which excerpts reached the answer model.

The methods run sequentially, with their order determined by the question. Long documents and unsuccessful coverage searches can take several minutes. The app accepts up to 5,000,000 text characters and questions up to 2,000 characters. Only one comparison runs at a time per backend process.

The application does not persist documents or questions to disk and has no account or document database. Inference goes to the configured Ollama endpoint, which is local by default.

## How retrieval works

### Original RBR

The implementation in [`backend/retrieval.py`](backend/retrieval.py) follows these steps:

1. Split the current region into left, middle, and right thirds.
2. Embed 256-word chunks in both outer thirds. Record each side's largest nonnegative cosine similarity to the question.
3. Ask the answer model about the first and last 400 words of the left third, then the right third. Supply smaller thirds in full.
4. Accept a nonempty answer that does not begin with `NOT_FOUND`.
5. If both views fail, discard the outer thirds and recurse into the middle.
6. At 500 words or recursion depth eight, evaluate the full remaining core with notes from the last five unsuccessful inspections.

The query is embedded once when recursion is needed; a small terminal core requires no embedding. Duplicate passages can reuse embeddings within a method. Excerpt offsets refer to whitespace-delimited words in the original input, and separated excerpts include an omission marker.

**Similarity scores are diagnostic: they do not choose the retained third or measure answer confidence.** Original RBR can discard evidence from an outer third's unseen interior, even if that passage was embedded. Logarithmic recursion depth also does not imply logarithmic total embedding work.

### Coverage-aware application variant

[`backend/coverage_retrieval.py`](backend/coverage_retrieval.py) tries the boundary traversal first. If it returns no answer, the variant inspects overlapping windows of up to 256 words, normally with 48 words of overlap, across the document. BM25 ranks the remaining windows, with no top-k cutoff; inspection continues until an answer is accepted or all remaining windows are examined. Windows already supplied in full are skipped.

The variant recognizes abstentions such as `NOT FOUND.` and quoted `NOT_FOUND`, reports the number of unique words examined, and bounds supplied views using a conservative UTF-8 byte budget. Oversized boundary views and depth-limited cores are split into smaller windows. Unlike original RBR, this variant does not pass previous inspection notes to generation.

An unsupported early answer can still stop either variant. Coverage can increase the number of model calls substantially, especially when no answer exists. Direct prompting supplies the entire document in one call, but the model may truncate an oversized prompt. Reported processed tokens do not reveal the exact original length before truncation. Neither approach guarantees a correct or supported answer.

## Recorded benchmark results

The checked-in [results file](benchmark/results.json) contains a ten-question pilot on LongBench MultiFieldQA-en. The source dataset has 150 examples; 109 satisfy the 2,000–8,000-word inclusion rule. Sampling used seed **2026**, and the ten selected contexts contain **2,279–7,293 words**.

The recorded environment was an **Apple M2 with 16 GiB memory**, **Ollama 0.24.0**, `llama3.2` for generation, and `nomic-embed-text` for embeddings. Generation used temperature 0, model seed 42, a 32,768-token context setting, and a 256-token output limit. These are saved measurements from that run, not measurements of every subsequent code revision.

| Method | Mean F1 | Exact match | Mean answer calls | Median total time (s) | Mean prompt tokens |
|---|---:|---:|---:|---:|---:|
| Original RBR | 21.03% | 0.00% | 2.10 | 10.81 | 2,380.6 |
| Direct prompting | 16.95% | 0.00% | 1.00 | 31.72 | 6,639.8 |
| BM25 | 22.49% | 0.00% | 1.00 | 13.42 | 3,332.3 |
| Dense cosine | 26.18% | 0.00% | 1.00 | 15.29 | 3,381.2 |

Dense retrieval had the highest observed F1 in this sample. Original RBR had a lower median time and higher observed F1 than direct prompting, but ten examples do not establish a stable ranking. No method achieved exact match.

The paired bootstrap estimates for **RBR minus each baseline**, in F1 percentage points, are:

| Baseline | Mean difference | Descriptive 95% interval |
|---|---:|---:|
| Direct prompting | +4.09 | [-16.03, 24.01] |
| BM25 | -1.45 | [-16.17, 11.32] |
| Dense cosine | -5.15 | [-27.31, 15.84] |

All intervals include zero. They use 2,000 paired bootstrap resamples and do not establish superiority or adjust for multiple comparisons. This pilot is not a full LongBench evaluation or an official leaderboard score. See the [results discussion](benchmark/RESULTS.md) and [benchmark protocol](benchmark/README.md).

### Methods and metrics

| Method | Context supplied to generation | Retrieval setup |
|---|---|---|
| Original RBR | Boundary views followed by a recursive middle/core search | Nomic similarity recorded; scores do not route the search |
| Direct prompting | Entire document in one request | No embeddings or retrieval |
| BM25 | Top eight passages, restored to document order | 256-word passages, 48-word overlap; k1=1.5, b=0.75 |
| Dense cosine | Top eight passages, restored to document order | Same passages, ranked using Nomic cosine similarity |

All four methods share the answer instructions, models, temperature, model seed, context setting, and output limit. RBR can make multiple calls, and its terminal core receives inspection notes, so total call and input budgets differ.

- **F1:** best-reference token overlap after lowercasing, removing ASCII punctuation and English articles, and normalizing whitespace.
- **Exact match:** whether the normalized prediction matches any normalized reference.
- **Total time:** embedding, retrieval/scoring, and generation time for a method; it excludes browser and full application request overhead.
- **Prompt tokens:** actual Ollama prompt counts summed over all answer calls, including repeated content and instructions.

F1 measures reference overlap, not factuality or evidence recall. The evaluation has no human support grading or passage relevance annotations. Context inclusion uses word counts rather than exact tokenizer lengths, so truncation remains possible. Timings depend on hardware, warm-up, model loading, service caches, and other machine activity.

## Run the benchmark

Activate the Python environment, start Ollama, and install both models as described above. Run these commands from the project root:

```sh
python benchmark/download.py
python benchmark/run.py benchmark/data/multifieldqa_en.jsonl \
  --examples 10 --seed 2026 --context-tokens 32768 \
  --output benchmark/results-local.json
```

The downloader checks the public mirror against a pinned SHA-256 before saving it under the ignored `benchmark/data/` directory. The full hash and source links are documented in the [benchmark README](benchmark/README.md).

The runner warms up embedding and generation, randomizes method order deterministically for each question, and creates a separate client cache for each method. It checkpoints after every completed method. Repeating the same command resumes completed work only when the dataset, backend/runner source hash, model digests, and configuration match. If any change, choose a new `--output` path. The example uses a separate file to preserve the checked-in pilot; omitting `--output` defaults to `benchmark/results.json`.

Each output includes selected question IDs, questions, references, predictions, F1/exact match, generation and embedding timings, call/token counts, model digests, dataset/source hashes, aggregate summaries, and paired bootstrap intervals. It does not copy the source contexts into the result file. Model seeds help control sampling but do not guarantee identical output or timing across environments.

For a larger run, change `--examples` and use another output path; the runner selects at most the number of eligible questions. The benchmark still evaluates original RBR, not the app's coverage-aware variant.

## Tests and validation

From the project root, with the Python environment active:

```sh
python -m unittest discover -s tests -v
```

The suite currently contains **27 tests** in [`tests/test_application.py`](tests/test_application.py):

| Test class | Count | What it checks |
|---|---:|---|
| `RetrievalTests` | 16 | Word offsets and excerpts; left/right/middle traversal; early stopping; terminal cores and depth limits; missed outer-interior evidence; similarity behavior; Nomic prefixes and embedding reuse/failures; cosine edge cases; abstention handling; direct-prompt construction; BM25/dense baselines; F1/exact-match scoring and passage overlap |
| `CoverageTests` | 7 | Hidden keys at multiple document positions; complete word coverage when no answer exists; overlapping evidence across window boundaries; bounded views with Unicode text and a small context budget; early stopping; avoiding repeated generation for a fully examined document; abstention variants |
| `ApiTests` | 4 | Comparison responses and invalid input; recovery of evidence missed by original boundary views; an embedding failure leaving the direct answer available; HTTP 429 while a comparison is already running |

These tests use a deterministic fake Ollama client and FastAPI's in-process test client. **They require no running Ollama service or model downloads.** They verify application behavior, not real-model answer quality, GPU behavior, or live HTTP integration with Ollama.

To run one group or one regression case:

```sh
python -m unittest discover -s tests -v -k CoverageTests
python -m unittest discover -s tests -v -k test_no_answer_examines_every_word
```

Validate the frontend separately:

```sh
cd frontend
npm ci
npm run build
```

The production build writes to the ignored `frontend/dist/` directory. This is a compilation check; there is no automated browser or frontend component test suite configured. A manual smoke check is to start both servers, confirm `GET /api/health` reports `available: true`, and compare `The code is 2004.` with the question `What is the code?` in the interface. Inspect the answers and trace; model output is not guaranteed.

## Configuration and troubleshooting

Set backend environment variables before starting Uvicorn:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama service endpoint |
| `GENERATION_MODEL` | `llama3.2` | Answer model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CONTEXT_TOKENS` | `32768` | Requested context window per answer call |
| `OUTPUT_TOKENS` | `256` | Maximum generated tokens per answer call |

Increase the context setting only if the model and available memory support it. The benchmark's `--context-tokens` flag overrides `CONTEXT_TOKENS` for that run. Other algorithm constants live in [`backend/config.py`](backend/config.py).

The Vite development server proxies `/api` to `http://127.0.0.1:8000`. Set `API_PROXY_TARGET` before `npm run dev` if the backend uses another address. For deployment, serve the frontend build and route `/api` to the Python service through the same origin; cross-origin access requires appropriate CORS configuration.

| Symptom | Check |
|---|---|
| Ollama unavailable or models missing | Start Ollama, run both `ollama pull` commands, and inspect `http://127.0.0.1:8000/api/health`. |
| Frontend cannot reach the backend | Confirm Uvicorn is running and the Vite proxy target matches its address. |
| HTTP 422 | Supply nonblank document text and a nonblank question within the character limits. |
| HTTP 429 | Wait for the current comparison to finish; the backend permits one at a time per process. |
| Benchmark refuses to resume | Use a new output path when the source, dataset, model digests, or settings change. |
| Slow comparison | Inspect call counts and coverage; an unsuccessful fallback can inspect the entire document. |

## Project layout

```text
backend/                     FastAPI service, Ollama client, and retrieval methods
frontend/                    React interface and Vite configuration
benchmark/run.py             Four-method paired evaluation and scoring
benchmark/download.py        Dataset download with checksum verification
benchmark/results.json       Recorded ten-question pilot and metadata
benchmark/write_paper_results.py  Generate LaTeX values from the recorded pilot
tests/test_application.py    Deterministic retrieval, coverage, and API tests
paper/                       IEEE manuscript, generated values, figure, and PDF
```

## Research paper

The IEEE-style manuscript is [`paper/main.tex`](paper/main.tex), with **Veer Vikram Singh** as its sole author. Read the [compiled paper](paper/main.pdf) or inspect the original [TikZ diagram](paper/figures/boundary_process.tex).

To regenerate the paper's numerical macros from the completed, checked-in `benchmark/results.json`:

```sh
python benchmark/write_paper_results.py
```

The script validates that all selected questions have four completed methods and writes `paper/benchmark_summary.tex`. It reads the fixed checked-in result path, not `results-local.json`.

Compile with [Tectonic](https://tectonic-typesetting.github.io/):

```sh
cd paper
tectonic main.tex
```

Alternatively, use a LaTeX installation providing IEEEtran, TikZ, algorithm, and algpseudocode, or upload `paper/` to Overleaf and compile `main.tex` with pdfLaTeX. Keep `benchmark_summary.tex` and the `figures/` directory beside the manuscript.

The paper presents a small pilot and ongoing research, including missed evidence, unsupported early answers, multi-passage questions, context limits, and the difference between recursion depth and total computational work.
