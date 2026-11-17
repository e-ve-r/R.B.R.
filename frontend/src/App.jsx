import React, { useEffect, useMemo, useState } from 'react';
import './styles.css';

function formatSeconds(value) {
    return value.toFixed(2) + ' s';
}

function Metric(properties) {
    return (
        <div className="metric">
            <dt>{properties.label}</dt>
            <dd>{properties.value}</dd>
        </div>
    );
}

function TraceStep(properties) {
    const step = properties.step;
    let sideLabel = step.side + ' third';
    if (step.side === 'core') {
        sideLabel = 'final core';
    } else if (step.side === 'fallback') {
        sideLabel = 'coverage search passage';
    }
    let answerLabel = 'NOT_FOUND';
    if (step.found) {
        answerLabel = 'answer returned';
    }
    let similarityLabel = 'No embedding score for this view';
    if (step.similarity !== null) {
        similarityLabel = 'Maximum cosine similarity: ' + step.similarity.toFixed(4);
    }
    const excerpts = [];
    for (let index = 0; index < step.excerpts.length; index += 1) {
        const excerpt = step.excerpts[index];
        excerpts.push(
            <div className="excerpt" key={index}>
                <p className="excerpt-position">Words {excerpt.start_word + 1} to {excerpt.end_word}</p>
                <p>{excerpt.text}</p>
            </div>
        );
    }
    return (
        <details className="trace-step">
            <summary>Depth {step.depth} · {sideLabel} · {answerLabel}</summary>
            <p className="muted">Region: {step.region_words.toLocaleString()} words. {similarityLabel}.</p>
            {excerpts}
        </details>
    );
}

function AnswerCard(properties) {
    const result = properties.result;
    let content = <p className="empty-answer">Your answer will appear here.</p>;
    if (properties.isRunning) {
        content = <p className="empty-answer">Running the comparison…</p>;
    }
    if (result && result.status === 'error') {
        content = <p className="error-message" role="alert">{result.error}</p>;
    }
    if (result && result.status === 'complete') {
        let answer = result.answer;
        if (!result.found) {
            answer = 'The model did not find an answer in the text it examined.';
        }
        let warning = null;
        if (result.warning) {
            warning = <p className="result-note">{result.warning}</p>;
        }
        const steps = [];
        for (let index = 0; index < result.trace.length; index += 1) {
            steps.push(<TraceStep step={result.trace[index]} key={index} />);
        }
        let trace = null;
        if (steps.length > 0) {
            trace = (
                <div className="trace">
                    <h3>What the model examined</h3>
                    <p className="muted">Open a step to inspect its excerpts. Similarity is not answer confidence.</p>
                    {steps}
                </div>
            );
        }
        content = (
            <div>
                <p className="answer-text">{answer}</p>
                <dl className="metrics">
                    <Metric label="Total time" value={formatSeconds(result.metrics.elapsed_seconds)} />
                    <Metric label="Answer calls" value={result.metrics.generation_calls} />
                    <Metric label="Processed prompt tokens" value={result.metrics.prompt_tokens.toLocaleString()} />
                    <Metric label="Embedding requests" value={result.metrics.embedding_requests} />
                </dl>
                {warning}
                {result.coverage && (
                    <p className="muted">
                        Examined {result.coverage.examined_words.toLocaleString()} of {result.coverage.total_words.toLocaleString()} words
                        {result.fallback_used ? ' · Coverage search used' : ''}.
                    </p>
                )}
                {trace}
            </div>
        );
    }
    return (
        <section className="answer-card" aria-label={properties.title}>
            <h2>{properties.title}</h2>
            <p className="card-description">{properties.description}</p>
            {content}
        </section>
    );
}

export default function App() {
    const [documentText, setDocumentText] = useState('');
    const [question, setQuestion] = useState('');
    const [comparison, setComparison] = useState(null);
    const [isRunning, setIsRunning] = useState(false);
    const [elapsedSeconds, setElapsedSeconds] = useState(0);
    const [error, setError] = useState('');
    const [connectionMessage, setConnectionMessage] = useState('Checking Ollama…');

    useEffect(function checkConnection() {
        async function loadHealth() {
            try {
                const response = await fetch('/api/health');
                if (!response.ok) {
                    throw new Error('The backend is unavailable.');
                }
                const health = await response.json();
                if (health.available) {
                    setConnectionMessage('Ollama ready · ' + health.settings.generation_model + ' · ' + health.settings.embedding_model);
                } else if (health.missing_models && health.missing_models.length > 0) {
                    setConnectionMessage('Install the missing Ollama models: ' + health.missing_models.join(', '));
                } else {
                    setConnectionMessage(health.message);
                }
            } catch (requestError) {
                setConnectionMessage('Start the Python backend and Ollama to run a comparison.');
            }
        }
        loadHealth();
    }, []);

    useEffect(function trackElapsedTime() {
        if (!isRunning) {
            return undefined;
        }
        const startedAt = Date.now();
        function updateElapsedTime() {
            setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
        }
        const timer = window.setInterval(updateElapsedTime, 1000);
        function stopElapsedTimer() {
            window.clearInterval(timer);
        }
        return stopElapsedTimer;
    }, [isRunning]);

    function handleDocumentChange(event) {
        setDocumentText(event.target.value);
        setComparison(null);
        setError('');
    }

    function handleQuestionChange(event) {
        setQuestion(event.target.value);
        setComparison(null);
        setError('');
    }

    async function handleFileChange(event) {
        const file = event.target.files[0];
        if (!file) {
            return;
        }
        if (file.size > 20000000) {
            setError('Choose a text file smaller than 20 MB.');
            return;
        }
        try {
            const text = await file.text();
            if (text.length > 5000000) {
                throw new Error('The document exceeds the five-million-character limit.');
            }
            setDocumentText(text);
            setComparison(null);
            setError('');
        } catch (fileError) {
            setError(fileError.message);
        }
        event.target.value = '';
    }

    async function handleComparison(event) {
        event.preventDefault();
        if (!documentText.trim() || !question.trim()) {
            setError('Paste some text and enter a question.');
            return;
        }
        setError('');
        setComparison(null);
        setElapsedSeconds(0);
        setIsRunning(true);
        try {
            const response = await fetch('/api/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: documentText, question: question }),
            });
            const body = await response.json();
            if (!response.ok) {
                let message = body.detail;
                if (typeof message !== 'string') {
                    message = 'The request is invalid or too large.';
                }
                throw new Error(message);
            }
            setComparison(body);
        } catch (requestError) {
            setError(requestError.message);
        } finally {
            setIsRunning(false);
        }
    }

    const wordCount = useMemo(function countDocumentWords() {
        const trimmedText = documentText.trim();
        if (!trimmedText) {
            return 0;
        }
        return trimmedText.split(/\s+/).length;
    }, [documentText]);
    let status = <p className="status" role="status">{connectionMessage}</p>;
    if (isRunning) {
        status = (
            <p className="status" role="status" aria-live="polite">
                Comparing both methods · {elapsedSeconds} seconds. Long documents can take several minutes.
            </p>
        );
    }
    let errorMessage = null;
    if (error) {
        errorMessage = <p className="error-message" role="alert">{error}</p>;
    }
    let recursiveResult = null;
    let directResult = null;
    if (comparison) {
        recursiveResult = comparison.results.recursive_boundary;
        directResult = comparison.results.direct;
    }
    let buttonLabel = 'Compare answers';
    if (isRunning) {
        buttonLabel = 'Comparing…';
    }
    return (
        <main className="workspace">
            <header className="page-header">
                <p className="eyebrow">A local text-retrieval experiment</p>
                <h1>Recursive Boundary Retrieval</h1>
                <p>Paste a document, ask for a detail, and compare boundary retrieval with direct prompting.</p>
            </header>
            <form className="input-panel" onSubmit={handleComparison}>
                <div className="field-heading">
                    <label htmlFor="document-text">Document text</label>
                    <label className="file-button">
                        Load a text file
                        <input type="file" accept=".txt,.md,text/plain,text/markdown" onChange={handleFileChange} disabled={isRunning} />
                    </label>
                </div>
                <textarea id="document-text" className="document-input" value={documentText}
                    onChange={handleDocumentChange} disabled={isRunning} maxLength={5000000}
                    placeholder="Paste your long text here…" spellCheck={false} />
                <p className="muted">{wordCount.toLocaleString()} words · Up to 5 million characters · Text is not saved by this app.</p>
                <label htmlFor="question">What would you like to find?</label>
                <input id="question" className="question-input" value={question} onChange={handleQuestionChange}
                    disabled={isRunning} maxLength={2000} placeholder="For example: What is the combination to the safe?" />
                <div className="form-footer">
                    {status}
                    <button type="submit" disabled={isRunning}>{buttonLabel}</button>
                </div>
                {errorMessage}
            </form>
            <div className="comparison-grid">
                <AnswerCard title="Recursive boundary retrieval" description="Checks boundary views, then searches skipped passages if no answer is found."
                    result={recursiveResult} isRunning={isRunning} />
                <AnswerCard title="Direct prompting" description="Sends the full document in one request, without retrieval."
                    result={directResult} isRunning={isRunning} />
            </div>
            <footer className="page-footer">
                Both methods use the same answer model. This is ongoing research; either method can miss information or return an unsupported answer.
            </footer>
        </main>
    );
}
