"""Paired public-data benchmark of RBR, direct prompting, BM25, and dense retrieval."""
import argparse
import hashlib
import json
import platform
import random
import re
import statistics
import string
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.baselines import passage_baseline
from backend.config import Settings
from backend.ollama_client import ANSWER_INSTRUCTIONS, OllamaClient
from backend.retrieval import RecursiveBoundaryRetriever, direct_prompt


METHODS = ['recursive_boundary', 'direct', 'bm25', 'dense']


def normalize_answer(answer):
    lowercase_answer = answer.lower()
    characters = []
    for character in lowercase_answer:
        if character not in string.punctuation:
            characters.append(character)
    without_punctuation = ''.join(characters)
    without_articles = re.sub(r'\b(a|an|the)\b', ' ', without_punctuation)
    return ' '.join(without_articles.split())


def answer_scores(prediction, references):
    best_f1 = 0.0
    best_exact_match = 0.0
    prediction_tokens = normalize_answer(prediction).split()
    for reference in references:
        reference_tokens = normalize_answer(reference).split()
        if normalize_answer(prediction) == normalize_answer(reference):
            best_exact_match = 1.0
        if not prediction_tokens or not reference_tokens:
            current_f1 = float(prediction_tokens == reference_tokens)
        else:
            common = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
            current_f1 = 2 * common / (len(prediction_tokens) + len(reference_tokens))
        best_f1 = max(best_f1, current_f1)
    return {'f1': best_f1, 'exact_match': best_exact_match}


def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.temporary.json')
    temporary.write_text(json.dumps(results, indent=2))
    temporary.replace(path)


def summarize(results):
    summary = {}
    for method in METHODS:
        entries = []
        for example in results:
            if method in example['methods']:
                entries.append(example['methods'][method])
        if not entries:
            continue
        f1_values = []
        exact_values = []
        durations = []
        calls = []
        prompt_tokens = []
        embeddings = []
        for entry in entries:
            f1_values.append(entry['f1'])
            exact_values.append(entry['exact_match'])
            durations.append(entry['metrics']['elapsed_seconds'])
            calls.append(entry['metrics']['generation_calls'])
            prompt_tokens.append(entry['metrics']['prompt_tokens'])
            embeddings.append(entry['metrics']['embedding_seconds'])
        summary[method] = {
            'examples': len(entries),
            'mean_f1': statistics.mean(f1_values),
            'mean_exact_match': statistics.mean(exact_values),
            'median_seconds': statistics.median(durations),
            'mean_generation_calls': statistics.mean(calls),
            'mean_prompt_tokens': statistics.mean(prompt_tokens),
            'mean_embedding_seconds': statistics.mean(embeddings),
        }
    return summary


def paired_intervals(results):
    intervals = {}
    for baseline in ['direct', 'bm25', 'dense']:
        differences = []
        for example in results:
            if 'recursive_boundary' in example['methods'] and baseline in example['methods']:
                differences.append(example['methods']['recursive_boundary']['f1'] - example['methods'][baseline]['f1'])
        if not differences:
            continue
        random_generator = random.Random(2026)
        means = []
        for repetition in range(2000):
            means.append(statistics.mean(random_generator.choices(differences, k=len(differences))))
        means.sort()
        intervals[baseline] = {
            'paired_examples': len(differences),
            'rbr_minus_baseline_f1_points': 100 * statistics.mean(differences),
            'bootstrap_95_percent_points': [100 * means[49], 100 * means[1949]],
        }
    return intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--examples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, default=Path('benchmark/results.json'))
    parser.add_argument('--context-tokens', type=int, default=32768)
    arguments = parser.parse_args()
    if arguments.examples < 1:
        parser.error('--examples must be positive')
    settings = replace(Settings.from_environment(), context_tokens=arguments.context_tokens)
    dataset_bytes = arguments.dataset.read_bytes()
    examples = []
    for line in dataset_bytes.splitlines():
        example = json.loads(line)
        word_count = len(example['context'].split())
        if 2000 <= word_count <= 8000:
            examples.append(example)
    if not examples:
        raise ValueError('No examples meet the 2,000 to 8,000 word inclusion rule.')
    selected = random.Random(arguments.seed).sample(examples, min(arguments.examples, len(examples)))
    metadata_client = OllamaClient(settings)
    installed_models = metadata_client.describe_models()
    model_digests = {}
    for configured_name in [settings.generation_model, settings.embedding_model]:
        canonical_name = configured_name
        if ':' not in canonical_name:
            canonical_name += ':latest'
        for model in installed_models:
            if model['name'] == canonical_name:
                model_digests[configured_name] = model['digest']
        if configured_name not in model_digests:
            raise ValueError('Missing installed model: ' + canonical_name)
    source_hasher = hashlib.sha256()
    for source_path in sorted((Path(__file__).resolve().parents[1] / 'backend').glob('*.py')):
        source_hasher.update(source_path.read_bytes())
    source_hasher.update(Path(__file__).read_bytes())
    selected_ids = []
    for example in selected:
        selected_ids.append(example['_id'])
    configuration = {
        'dataset': 'LongBench MultiFieldQA-en, public mirror',
        'dataset_sha256': hashlib.sha256(dataset_bytes).hexdigest(),
        'eligible_examples': len(examples),
        'inclusion_words': [2000, 8000],
        'seed': arguments.seed,
        'selected_ids': selected_ids,
        'settings': asdict(settings),
        'model_digests': model_digests,
        'source_sha256': source_hasher.hexdigest(),
        'answer_instructions': ANSWER_INSTRUCTIONS,
    }
    output = {
        'configuration': configuration,
        'environment': {'platform': platform.platform(), 'machine': platform.machine()},
        'results': [],
    }
    if arguments.output.exists():
        output = json.loads(arguments.output.read_text())
        if output['configuration'] != configuration:
            raise ValueError('Existing results have different settings or code. Choose a new output path.')
    # Warm both services before the randomized method order.
    metadata_client.embed(['warmup'], is_query=True)
    metadata_client.generate('What color is the tile?', 'The tile is blue.')
    for position, example in enumerate(selected):
        row = None
        for existing_row in output['results']:
            if existing_row['id'] == example['_id']:
                row = existing_row
        if row is None:
            row = {'id': example['_id'], 'question': example['input'], 'references': example['answers'],
                   'words': len(example['context'].split()), 'methods': {}}
            output['results'].append(row)
        method_order = list(METHODS)
        random.Random(str(arguments.seed) + example['_id']).shuffle(method_order)
        row['execution_order'] = method_order
        for method in method_order:
            if method in row['methods']:
                continue
            client = OllamaClient(settings)
            if method == 'recursive_boundary':
                result = RecursiveBoundaryRetriever(client, settings).search(example['context'], example['input'])
            elif method == 'direct':
                result = direct_prompt(example['context'], example['input'], client, settings)
            else:
                result = passage_baseline(example['context'], example['input'], client, settings, method)
            scores = answer_scores(result['answer'], example['answers'])
            # Store predictions and metrics, not a copy of the public source text.
            row['methods'][method] = {'answer': result['answer'], 'found': result['found'],
                                     'f1': scores['f1'], 'exact_match': scores['exact_match'],
                                     'metrics': result['metrics']}
            output['summary'] = summarize(output['results'])
            output['paired_intervals'] = paired_intervals(output['results'])
            save_results(arguments.output, output)
            print(str(position + 1) + '/' + str(len(selected)) + ' ' + method +
                  ' F1=' + str(round(scores['f1'], 3)), flush=True)
    print(json.dumps(output['summary'], indent=2))


if __name__ == '__main__':
    main()
