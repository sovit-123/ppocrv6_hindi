"""Download selected shards, extract supported Hindi crops, stream labels, and split.

The selected sources are hindi_local, iiit_hindi, hindidataocr, and kaggle.
Nepali annotations are excluded. The shared annotation file is downloaded once
and scanned once, regardless of the number of selected image archives.

USAGE:
    python prepare.py --batch 1
    python prepare.py --batches 1 2 3 --output prepared_batches_001_003
    python prepare.py --all-batches --output prepared_all
    python prepare.py --batches 1 2 --offline --output prepared_offline

--batch remains compatible with the original single-archive workflow. Use a
fresh output directory for every preparation run. --all-batches downloads all
42 archives and requires substantial disk space.

Run `python run.py setup` before using this CLI, including with --offline.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import tarfile
import unicodedata

from PIL import Image, ImageDraw

PREFIXES = ('hindi_local', 'iiit_hindi', 'hindidataocr', 'kaggle')
REPO = 'himalaya-ai/devanagari_ocr_dataset'
ROOT = Path(__file__).resolve().parent


def annotations(path):
    import ijson
    with path.open('rb') as f:
        first = b''
        while not first:
            c = f.read(1)
            if not c:
                return
            first = c.strip()
        f.seek(0)
        if first == b'[':
            yield from ijson.items(f, 'item')
        else:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def extract(tarpath, target, prefixes, names=None):
    """Extract selected images and reject duplicate basenames across archives."""
    target.mkdir(parents=True, exist_ok=True)
    if names is None:
        names = set()
    with tarfile.open(tarpath, 'r|gz') as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or not name.startswith(prefixes):
                continue
            if name in names:
                raise ValueError(f'Duplicate selected image basename: {name} ({tarpath.name})')
            names.add(name)
            source = archive.extractfile(member)
            with source, (target / name).open('wb') as out:
                shutil.copyfileobj(source, out)
    return names


def text_of(record):
    if 'messages' in record:
        turns = [m.get('content') for m in record['messages'] if m.get('role') == 'assistant']
    else:
        turns = [m.get('value') for m in record.get('conversations', []) if m.get('from') == 'gpt']
    if len(turns) != 1 or not isinstance(turns[0], str):
        return None
    return unicodedata.normalize('NFC', turns[0]).strip()


def ctc_steps(text):
    return len(text) + sum(a == b for a, b in zip(text, text[1:]))


def prepare(data, archives, out, seed=42, max_length=40, min_ratio=1.0):
    """Prepare one combined split from one archive or an iterable of archives."""
    if out.exists():
        raise ValueError(f'{out} exists; choose a new --output directory to preserve splits.')
    if isinstance(archives, (str, Path)):
        archives = [archives]
    archives = [Path(archive) for archive in archives]
    if not archives:
        raise ValueError('Select at least one image archive.')
    if len(set(archives)) != len(archives):
        raise ValueError('Each image archive may be selected only once.')
    names = set()
    for archive in archives:
        extract(archive, data / 'devanagari_ocr', PREFIXES, names)
    stats = Counter(extracted_hindi=len(names))
    samples, seen, digests = [], set(), set()
    for record in annotations(data / 'devanagari_ocr.json'):
        stats['annotation_records'] += 1
        paths = record.get('images')
        if paths is None:
            paths = [record['image']] if isinstance(record.get('image'), str) else []
        if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
            stats['not_single_image'] += 1
            continue
        name = Path(paths[0]).name
        if name not in names:
            continue
        if name in seen:
            raise ValueError(f'Duplicate annotation for {name}; resolve ambiguity before training.')
        seen.add(name)
        text = text_of(record)
        if not text or any(c in text for c in '\t\r\n\x00'):
            stats['invalid_text'] += 1
            continue
        # Devanagari presence is a sanity check; prefixes supply the language proxy.
        if not any('\u0900' <= c <= '\u097f' for c in text):
            stats['no_devanagari'] += 1
            continue
        if len(text) > max_length or ctc_steps(text) > 40:
            stats['too_long_for_ctc'] += 1
            continue
        image_path = data / 'devanagari_ocr' / name
        try:
            with Image.open(image_path) as im:
                im.load()
                w, h = im.size
                if w < 4 or h < 4 or w / h < min_ratio:
                    stats['crop_geometry'] += 1
                    continue
                # Identical decoded RGB images cannot cross splits, even with different names.
                rgb = im.convert('RGB')
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        except (OSError, ValueError):
            stats['bad_image'] += 1
            continue
        if digest in digests:
            stats['duplicate_image'] += 1
            continue
        digests.add(digest)
        samples.append((f'devanagari_ocr/{name}', text))
    samples.sort()
    random.Random(seed).shuffle(samples)
    n = len(samples)
    n_test = round(n * .1)
    n_val = round(n * .1)
    stats['matched_annotation_images'] = len(seen)
    stats['orphan_images'] = len(names - seen)
    if min(n_test, n_val, n - n_test - n_val) < 1:
        print(json.dumps(dict(stats), indent=2))
        raise ValueError(f'Only {n} usable Hindi samples in the selected archives.')
    splits = {'test': samples[:n_test], 'val': samples[n_test:n_test+n_val], 'train': samples[n_test+n_val:]}
    # Fixed script inventory + ASCII; other punctuation/characters learned from train only.
    chars = set(chr(i) for i in range(0x900, 0x980)) | set(chr(i) for i in range(33, 127))
    chars.update(c for _, text in splits['train'] for c in text if c != ' ')
    chars.discard(' ')
    chars = sorted(chars)
    vocab = set(chars) | {' '}
    out.mkdir(parents=True)
    (out / 'hindi_dict.txt').write_text('\n'.join(chars) + '\n', encoding='utf-8')
    with (out / 'splits.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'split'])
        for split, rows in splits.items():
            (out / f'{split}.txt').write_text(''.join(f'{p}\t{t}\n' for p,t in rows), encoding='utf-8')
            writer.writerows((p, split) for p, _ in rows)
    oov = {s: sorted(set(''.join(t for _,t in rows)) - vocab) for s,rows in splits.items()}
    report = {'seed': seed,
              'archives': [{'name': archive.name, 'bytes': archive.stat().st_size}
                           for archive in archives],
              'prefixes': PREFIXES, 'stats': dict(stats), 'counts': {s: len(r) for s,r in splits.items()},
              'per_prefix': {s: dict(Counter(next(p for p in PREFIXES if Path(x).name.startswith(p)) for x,_ in rows)) for s,rows in splits.items()},
              'max_length': max_length, 'ctc_time_steps': 40, 'min_aspect_ratio': min_ratio,
              'dictionary_sha256': hashlib.sha256((out/'hindi_dict.txt').read_bytes()).hexdigest(),
              'out_of_vocabulary': oov,
              'label_sha256': {s: hashlib.sha256((out/f'{s}.txt').read_bytes()).hexdigest() for s in splits}}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    preview = Image.new('RGB', (800, 100 * min(20,n)), 'white')
    draw = ImageDraw.Draw(preview)
    for i,(p,_) in enumerate(samples[:20]):
        with Image.open(data/p) as im:
            im = im.convert('RGB'); im.thumbnail((600, 72)); preview.paste(im, (190,i*100))
        draw.text((4,i*100+10), Path(p).name[:28], fill='black')
    preview.save(out/'crop_preview.jpg')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any(oov.values()):
        raise ValueError('Held-out characters absent from dictionary: see report.json. Do not train until resolved using a predefined inventory.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    batch_group = p.add_mutually_exclusive_group()
    batch_group.add_argument('--batch', '--batches', dest='batches', type=int,
                             nargs='+', choices=range(1,43), metavar='N',
                             help='Batch numbers to download and combine (default: 1)')
    batch_group.add_argument('--all-batches', action='store_true',
                             help='Download and combine all 42 image archives')
    p.add_argument('--data-dir', type=Path, default=ROOT/'data')
    p.add_argument('--output', type=Path, default=ROOT/'prepared')
    p.add_argument('--offline', action='store_true',
                   help='Use existing annotations and selected archives, without downloading')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-length', type=int, default=40)
    p.add_argument('--min-aspect-ratio', type=float, default=1.)
    a = p.parse_args()
    if not 1 <= a.max_length <= 40 or a.min_aspect_ratio <= 0:
        p.error('max-length must be 1..40; min-aspect-ratio must be positive')
    if a.output.exists():
        p.error('Output exists; choose a new --output directory.')
    batches = list(range(1,43)) if a.all_batches else (a.batches or [1])
    if len(set(batches)) != len(batches):
        p.error('Duplicate batch numbers are not allowed.')
    batches.sort()
    shards = [f'data/images_batch_{batch:03d}.tar.gz' for batch in batches]
    a.data_dir = a.data_dir.resolve(); a.output = a.output.resolve()
    from run import require_setup
    try:
        require_setup()
    except ValueError as exc:
        p.error(str(exc))
    if not a.offline:
        from huggingface_hub import HfApi, hf_hub_download
        revision = HfApi().dataset_info(REPO).sha
        hf_hub_download(REPO, 'devanagari_ocr.json', repo_type='dataset',
                        revision=revision, local_dir=a.data_dir)
        for batch, shard in zip(batches, shards):
            hf_hub_download(REPO, shard, repo_type='dataset', revision=revision,
                            local_dir=a.data_dir)
            provenance = {'repo': REPO, 'revision': revision,
                          'files': ['devanagari_ocr.json', shard]}
            source_path = a.data_dir/f'source_batch_{batch:03d}.json'
            source_path.write_text(json.dumps(provenance, indent=2))
    archives = [a.data_dir/shard for shard in shards]
    prepare(a.data_dir, archives, a.output, a.seed, a.max_length, a.min_aspect_ratio)
    sources = []
    for batch in batches:
        provenance = a.data_dir/f'source_batch_{batch:03d}.json'
        if provenance.exists():
            sources.append((provenance, json.loads(provenance.read_text())))
    if len(sources) == 1:
        shutil.copyfile(sources[0][0], a.output/'dataset_source.json')
    elif sources:
        (a.output/'dataset_source.json').write_text(
            json.dumps([source for _, source in sources], indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
