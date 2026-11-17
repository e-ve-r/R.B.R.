"""Download the public LongBench MultiFieldQA-en mirror and verify its content."""
import argparse
import hashlib
import urllib.request
from pathlib import Path


DATASET_URL = 'https://huggingface.co/datasets/Leooyii/longbench/resolve/main/multifieldqa_en.jsonl'
EXPECTED_SHA256 = '0aac182fd317dcf6d74f8e1e0f3e61029407435346c2e0b3ff9fb45ae49c5c3f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('benchmark/data/multifieldqa_en.jsonl'))
    arguments = parser.parse_args()
    with urllib.request.urlopen(DATASET_URL, timeout=120) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError('Dataset checksum differs from the documented file.')
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(content)
    print('Downloaded and verified:', arguments.output)


if __name__ == '__main__':
    main()
