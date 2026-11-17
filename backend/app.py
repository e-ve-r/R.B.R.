"""Paste text and compare coverage-aware RBR with direct Ollama prompting."""
import hashlib
import threading
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import Settings
from backend.coverage_retrieval import CoverageBoundaryRetriever
from backend.ollama_client import OllamaClient, OllamaError
from backend.retrieval import direct_prompt


settings = Settings.from_environment()
comparison_lock = threading.Lock()
app = FastAPI(title='Recursive Boundary Retrieval', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['GET', 'POST'],
    allow_headers=['Content-Type'],
)


class ComparisonRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000000)
    question: str = Field(min_length=1, max_length=2000)


@app.get('/api/health')
def health():
    client = OllamaClient(settings)
    model_names = []
    try:
        models = client.describe_models()
        for model in models:
            model_names.append(model['name'])
    except OllamaError as error:
        return {'available': False, 'message': str(error), 'settings': asdict(settings)}
    required = [settings.generation_model, settings.embedding_model]
    missing = []
    for model_name in required:
        canonical_name = model_name
        if ':' not in canonical_name:
            canonical_name += ':latest'
        if canonical_name not in model_names:
            missing.append(canonical_name)
    return {
        'available': len(missing) == 0,
        'models': model_names,
        'missing_models': missing,
        'settings': asdict(settings),
    }


@app.post('/api/compare')
def compare(request: ComparisonRequest):
    document = request.text.strip()
    question = request.question.strip()
    if not document or not question:
        raise HTTPException(422, 'Provide both document text and a question.')
    if not comparison_lock.acquire(blocking=False):
        raise HTTPException(429, 'A comparison is already running. Wait for it to finish.')
    try:
        # Vary execution order deterministically; never run both GPU jobs together.
        digest = hashlib.sha256(question.encode('utf-8')).digest()
        method_order = ['recursive_boundary', 'direct']
        if digest[0] % 2 == 1:
            method_order.reverse()
        results = {}
        for method in method_order:
            client = OllamaClient(settings)
            try:
                if method == 'recursive_boundary':
                    retriever = CoverageBoundaryRetriever(client, settings)
                    result = retriever.search(document, question)
                else:
                    result = direct_prompt(document, question, client, settings)
                result['status'] = 'complete'
            except (OllamaError, ValueError) as error:
                result = {'status': 'error', 'method': method, 'error': str(error)}
            results[method] = result
        return {
            'word_count': len(document.split()),
            'execution_order': method_order,
            'generation_model': settings.generation_model,
            'embedding_model': settings.embedding_model,
            'results': results,
        }
    finally:
        comparison_lock.release()
