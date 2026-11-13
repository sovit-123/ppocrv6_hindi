"""Stream annotations and report filename-prefix and Devanagari text statistics.

Prefix candidates are inferred by removing a final numeric identifier such as
``_0001`` or ``-0001`` from each image basename. Unnumbered stems are retained
as complete prefix candidates. Devanagari presence is a script observation, not
a Hindi language classification; review the examples before selecting sources.

The script is read-only, accepts either a JSON array or JSON Lines, and prints
its report to stdout as UTF-8 JSON. Its only third-party dependency is tqdm.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
NUMERIC_SUFFIX = re.compile(r'^(.*?)[_-]\d+$')


def annotations(path, chunk_size=1024 * 1024):
    """Yield records from a top-level JSON array without loading it all."""
    decoder = json.JSONDecoder()
    with path.open('r', encoding='utf-8-sig') as source:
        buffer = source.read(chunk_size)
        position = 0
        eof = not buffer

        def fill():
            nonlocal buffer, position, eof
            buffer = buffer[position:]
            position = 0
            chunk = source.read(chunk_size)
            buffer += chunk
            eof = not chunk
            return bool(chunk)

        def skip_whitespace():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return True
                if not fill():
                    return False

        if not skip_whitespace():
            return
        if buffer[position] != '[':
            source.seek(0)
            for line_number, line in enumerate(source, start=1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f'Invalid JSON Lines record at line {line_number}: {exc}') from exc
            return

        position += 1
        while True:
            if not skip_whitespace():
                raise ValueError('Unexpected end of JSON array')
            if buffer[position] == ']':
                return
            while True:
                try:
                    record, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f'Invalid or incomplete JSON array record: {exc}') from exc
                    fill()
            yield record
            position = end
            if not skip_whitespace():
                raise ValueError('Unterminated JSON array')
            if buffer[position] == ',':
                position += 1
            elif buffer[position] == ']':
                return
            else:
                raise ValueError(f'Expected comma or closing bracket at character {position}')


def image_path(record):
    """Return the sole image path, or None for an unsupported record shape."""
    if not isinstance(record, dict):
        return None
    paths = record.get('images')
    if paths is None:
        paths = [record['image']] if isinstance(record.get('image'), str) else []
    if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
        return None
    return paths[0]


def transcription(record):
    """Return one assistant transcription, matching the preparation schema."""
    turns = record.get('messages')
    if turns is not None:
        values = [turn.get('content') for turn in turns
                  if isinstance(turn, dict) and turn.get('role') == 'assistant']
    else:
        turns = record.get('conversations', [])
        values = [turn.get('value') for turn in turns
                  if isinstance(turn, dict) and turn.get('from') == 'gpt']
    if len(values) != 1 or not isinstance(values[0], str):
        return None
    text = values[0].strip()
    return text or None


def prefix_candidate(path):
    """Infer a source candidate from a basename's trailing numeric ID."""
    name = Path(path).name
    stem = Path(name).stem
    match = NUMERIC_SUFFIX.fullmatch(stem)
    if match:
        return match.group(1)
    if stem.isdigit():
        return '<numeric filenames>'
    return stem or '<empty filename>'


def analyze(path, examples=3):
    """Return streaming counts grouped by inferred filename prefix."""
    summary = Counter()
    groups = {}
    with tqdm(annotations(path), desc='Scanning annotations',
              unit='record', mininterval=1.0) as records:
        for record in records:
            summary['records'] += 1
            path_value = image_path(record)
            if path_value is None:
                summary['not_single_image'] += 1
                continue
            summary['single_image_records'] += 1
            name = Path(path_value).name
            prefix = prefix_candidate(path_value)
            group = groups.setdefault(prefix, {
                'counts': Counter(),
                'example_files': [],
                'example_devanagari_text': [],
            })
            group['counts']['records'] += 1
            if len(group['example_files']) < examples:
                group['example_files'].append(name)
            text = transcription(record)
            if text is None:
                group['counts']['invalid_text'] += 1
                summary['invalid_text'] += 1
            elif any('\u0900' <= char <= '\u097f' for char in text):
                group['counts']['devanagari_text'] += 1
                summary['devanagari_text'] += 1
                if len(group['example_devanagari_text']) < examples:
                    group['example_devanagari_text'].append(text)
            else:
                group['counts']['non_devanagari_text'] += 1
                summary['non_devanagari_text'] += 1
    prefixes = []
    for prefix, group in sorted(groups.items(),
                                key=lambda item: (-item[1]['counts']['records'], item[0])):
        prefixes.append({
            'prefix': prefix,
            **dict(group['counts']),
            'example_files': group['example_files'],
            'example_devanagari_text': group['example_devanagari_text'],
        })
    summary['prefix_candidates'] = len(prefixes)
    return {'summary': dict(summary), 'prefixes': prefixes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path,
                        default=ROOT / 'data' / 'devanagari_ocr.json')
    parser.add_argument('--examples', type=int, default=3,
                        help='Maximum example filenames and Devanagari texts per prefix (default: 3)')
    args = parser.parse_args()
    if args.examples < 0:
        parser.error('--examples must be non-negative')
    if not args.annotations.is_file():
        parser.error(f'Missing annotations file: {args.annotations}')
    report = analyze(args.annotations, args.examples)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
