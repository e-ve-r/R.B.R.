"""Write the paper's measured table directly from the saved benchmark."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / 'benchmark/results.json').read_text())
    rows = results['results']
    selected_count = len(results['configuration']['selected_ids'])
    for row in rows:
        if len(row['methods']) != 4:
            raise ValueError('Every question must have all four completed methods.')
    if len(rows) != selected_count:
        raise ValueError('The benchmark is not complete.')
    lengths = []
    for row in rows:
        lengths.append(row['words'])
    values = {
        'PilotCount': str(len(rows)),
        'EligibleCount': str(results['configuration']['eligible_examples']),
        'MinimumWords': str(min(lengths)),
        'MaximumWords': str(max(lengths)),
    }
    prefixes = {'recursive_boundary': 'Rbr', 'direct': 'Direct', 'bm25': 'Bm', 'dense': 'Dense'}
    for method, prefix in prefixes.items():
        summary = results['summary'][method]
        values[prefix + 'Fone'] = format(100 * summary['mean_f1'], '.2f')
        values[prefix + 'Calls'] = format(summary['mean_generation_calls'], '.2f')
        values[prefix + 'Time'] = format(summary['median_seconds'], '.2f')
    lines = []
    for name, value in values.items():
        lines.append('\\newcommand{\\' + name + '}{' + value + '}')
    (root / 'paper/benchmark_summary.tex').write_text('\n'.join(lines) + '\n')
    print('Paper values generated from completed results.')


if __name__ == '__main__':
    main()
