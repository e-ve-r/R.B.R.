# Recursive Boundary Retrieval

A small React and Python application for comparing a coverage-aware variant of Veer Vikram Singh's Recursive Boundary Retrieval (RBR) procedure with direct prompting. Paste a long text, ask for a specific detail, and inspect both answers side by side. The interface uses blue and white and keeps the recursion trace available for inspection. The public-data benchmark retains the original procedure described in the paper.

## Start the application

Install and start [Ollama](https://ollama.com/). Download the two local models:

```sh
ollama pull llama3.2
ollama pull nomic-embed-text
```

From the project root, start the backend:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
cd frontend
npm ci
npm run dev
```

Open the Vite URL, normally `http://127.0.0.1:5173`. Paste text or load a UTF-8 `.txt` or `.md` file. Enter a question and choose **Compare answers**. Long documents can take several minutes. The two methods run sequentially to avoid competing for the same local model; their execution order varies deterministically with the question.

The application accepts up to five million characters. It does not save documents or questions to disk. Ollama runs locally by default. Only one comparison runs at a time per backend process. This is a local research app, with no account or document database.

## The original algorithm

1. Split the current region into left, middle, and right thirds.
2. Embed 256-word chunks in both outer thirds with `nomic-embed-text`. Record the largest nonnegative cosine similarity to the question.
3. Ask the answer model about the first and last 400 words of the left third, then the right third. Smaller thirds are supplied in full.
4. Accept a nonempty answer that does not start with `NOT_FOUND`.
5. After two unsuccessful views, discard the outer thirds and recursively process the middle.
6. At 500 words or depth eight, evaluate the full remaining core, with notes from the last five unsuccessful views.

The query is embedded at most once per request; a small terminal core needs no embedding. Identical passages may reuse embeddings inside the request. Excerpt positions refer directly to the pasted text's whitespace-delimited words; there is no reconstruction from overlapping chunks. Separate excerpts have an omission marker.

**Embedding scores do not choose the next third and are not answer confidence.** `backend/retrieval.py` and the benchmark preserve this original algorithm.

Original RBR can miss evidence inside the unseen part of an outer third. For example, a secret key between history paragraphs may never reach the answer model even though its passage was embedded.

The application uses `backend/coverage_retrieval.py`: it first tries the boundary traversal, then, if no answer is returned, searches overlapping 256-word windows across the document. BM25 prioritizes windows containing question terms; there is no top-k cutoff, so low-ranked windows are still inspected if necessary. Windows already supplied in full are skipped. Abstention variants such as `NOT FOUND.` and quoted `NOT_FOUND` are recognized so they do not prematurely stop the search. The trace labels these additional calls as coverage search passages and reports unique words examined. Large boundary views and depth-limited cores are split using a conservative UTF-8 byte budget that reserves space for the question, instructions, and answer. This coverage fallback can require many more calls when no answer exists.

An unsupported early answer can still stop either retrieval variant. Direct prompting sends the entire text in one request, but the model has a finite context window. The interface warns about possible truncation and reports Ollama's processed token counts; those counts do not reveal the exact original input length before truncation. Neither method is guaranteed to answer correctly.

## Configuration

Set environment variables before starting the backend:

| Variable | Default |
|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` |
| `GENERATION_MODEL` | `llama3.2` |
| `EMBEDDING_MODEL` | `nomic-embed-text` |
| `CONTEXT_TOKENS` | `32768` |
| `OUTPUT_TOKENS` | `256` |

Increase `CONTEXT_TOKENS` only if the selected model and local memory support it. This value is a model-token limit, not a word limit. For a Vite server on another port, configure CORS or use the same-origin development proxy. For deployment, route `/api` to the Python service and serve the frontend build through the same origin.

## Tests and public-data benchmark

```sh
python -m unittest discover -s tests -v
cd frontend
npm run build
```

From the project root, obtain the public dataset and run the paired comparison:

```sh
python benchmark/download.py
python benchmark/run.py benchmark/data/multifieldqa_en.jsonl --examples 10 --seed 2026
```

The benchmark evaluates original RBR, direct prompting, BM25, and dense cosine retrieval on the same seeded LongBench MultiFieldQA-en subset. It saves actual predictions, F1, exact match, token usage, embedding costs, total method costs, model digests, source hashes, and paired bootstrap intervals in `benchmark/results.json`. A repeated identical command resumes completed methods; changed code or settings require a different `--output` path. See `benchmark/README.md` for the protocol and limitations.

## Research paper

`paper/main.tex` is the IEEE-style manuscript, with **Veer Vikram Singh** as its only author. The diagram is original TikZ code in `paper/figures/boundary_process.tex`. Measured table values are generated from the benchmark into `paper/benchmark_summary.tex`; the compiled paper is `paper/main.pdf`.

Compile with [Tectonic](https://tectonic-typesetting.github.io/) or a LaTeX installation providing IEEEtran, TikZ, algorithm, and algpseudocode:

```sh
cd paper
tectonic main.tex
```

You can also upload the contents of `paper/` to Overleaf and compile `main.tex` with pdfLaTeX. Keep `benchmark_summary.tex` and the `figures/` directory beside it.

The paper presents a small pilot and ongoing research. It discusses missed evidence, unsupported early answers, multi-passage questions, context limits, and the distinction between logarithmic recursion depth and total embedding cost.
