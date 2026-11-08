"""Copy the held-out test-split crops into data/infer_data for manual testing.

Reads prepared/test.txt (one `<path><TAB><transcription>` row per line, paths
relative to --data-dir) and *copies* each image into --out. Sources are never
moved or modified: evaluation still reads the originals from --data-dir.

The destination is flat (basename only). This project's test.txt basenames
are unique, and the script fails early on duplicates or missing sources
instead of leaving a partial copy behind. Re-running skips files already
present, so it is safe to repeat.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, default=ROOT/'prepared')
    p.add_argument('--data-dir', type=Path, default=ROOT/'data')
    p.add_argument('--out', type=Path, default=ROOT/'data/infer_data')
    a = p.parse_args()
    labels = a.prepared/'test.txt'
    if not labels.is_file():
        p.error(f'Missing {labels}')
    lines = [l for l in labels.read_text(encoding='utf-8').splitlines() if l.strip()]

    entries, seen = [], {}
    for line in lines:
        rel = Path(line.split('\t', 1)[0])
        if rel.name in seen:
            p.error(f'Duplicate basename in test.txt: {seen[rel.name]} and {rel}')
        seen[rel.name] = rel
        entries.append((a.data_dir/rel, a.out/rel.name))
    missing = [source for source, _ in entries if not source.is_file()]
    if missing:
        p.error(f'{len(missing)} test.txt images missing under {a.data_dir}; '
                f'first: {missing[0]}')

    a.out.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for source, target in entries:
        if target.exists():
            skipped += 1
            continue
        shutil.copy2(source, target)
        copied += 1
    print(f'Copied {copied} images ({skipped} already present) to {a.out}')


if __name__ == '__main__':
    main()
