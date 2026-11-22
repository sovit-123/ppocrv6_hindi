"""Verify archive, extraction, manifest, and prepared-data consistency.

Run after `python prepare.py --all-batches`:

    python check_data.py

The default check compares every archive's completion marker and filename
manifest with the physical extraction directory, then validates prepared labels,
paths, splits, dictionary, hashes, and provenance. It does not decompress images
or archives. For a full byte-for-byte comparison between tar members and
extracted files (slow; reads all selected data), use:

    python check_data.py --deep

Expectations are derived from the selected archives and generated metadata; no
tracked global data list is required. The JSON report is written to
`<prepared>/check_data_report.json` by default. Exit status is 0 on success and
1 on validation failure.
"""
import argparse
from collections import Counter
import csv
import hashlib
import heapq
import json
from pathlib import Path
import tarfile
import unicodedata

from tqdm.auto import tqdm

from prepare import PREFIXES, REPO, annotations, archive_layout, completed_archive_files, ctc_steps, text_of

ROOT = Path(__file__).resolve().parent
METADATA_FILES = {'.extract_complete.json', 'filenames.json'}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def text_digest(text):
    return hashlib.sha256(text.encode('utf-8')).digest()


class Validation:
    def __init__(self, max_errors=50):
        self.max_errors = max_errors
        self.checks = 0
        self.failures = 0
        self.errors = []
        self.details = {}

    def check(self, condition, code, message, **context):
        self.checks += 1
        if condition:
            return
        self.failures += 1
        if len(self.errors) < self.max_errors:
            self.errors.append({'code': code, 'message': message, **context})

    def result(self):
        return {'status': 'passed' if not self.failures else 'failed',
                'summary': {'checks': self.checks, 'failed_checks': self.failures,
                            'reported_errors': len(self.errors)},
                'details': self.details, 'errors': self.errors}


def validate_tar_contents(tarpath, final, expected, validation):
    """Compare every selected tar member byte-for-byte with its extracted file."""
    seen = set()
    with tarfile.open(tarpath, 'r|gz') as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or not name.startswith(PREFIXES):
                continue
            validation.check(name not in seen, 'duplicate_tar_basename',
                             f'Duplicate selected basename in {tarpath.name}: {name}')
            seen.add(name)
            path = final / name
            if not path.is_file():
                validation.check(False, 'missing_extracted_file',
                                 f'Tar member has no extracted file: {name}')
                continue
            source = archive.extractfile(member)
            source_hash = hashlib.sha256()
            file_hash = hashlib.sha256()
            with source, path.open('rb') as extracted:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    source_hash.update(chunk)
                for chunk in iter(lambda: extracted.read(1024 * 1024), b''):
                    file_hash.update(chunk)
            validation.check(source_hash.digest() == file_hash.digest(),
                             'extracted_content_mismatch',
                             f'Extracted content differs from tar member: {name}')
    expected = set(expected)
    validation.check(seen == expected, 'tar_manifest_mismatch',
                     f'Tar members differ from manifest in {tarpath.name}',
                     missing_from_tar=sorted(expected - seen)[:10],
                     unexpected_in_tar=sorted(seen - expected)[:10])


def validate_archives(data, archives, validation, deep):
    image_index = {}
    batch_details = []
    for tarpath in tqdm(archives, desc='Validating archives', unit='archive'):
        archive_id, final, part = archive_layout(tarpath, data / 'devanagari_ocr')
        validation.check(tarpath.is_file() and tarpath.stat().st_size > 0,
                         'missing_archive', f'Missing or empty archive: {tarpath}')
        files = completed_files(tarpath, data)
        validation.check(files is not None, 'invalid_extraction_metadata',
                         f'Invalid or incomplete extraction metadata for {archive_id}')
        validation.check(not part.exists(), 'leftover_partial_extraction',
                         f'Partial extraction directory remains: {part}')
        if files is None:
            batch_details.append({'archive': archive_id, 'status': 'invalid_metadata'})
            continue
        actual = set()
        if final.is_dir():
            actual = {entry.name for entry in final.iterdir()
                      if not entry.is_symlink() and entry.is_file()}
        expected_entries = set(files) | METADATA_FILES
        missing = set(files) - actual
        unexpected = actual - expected_entries
        validation.check(not missing, 'missing_extracted_files',
                         f'Files listed in manifest are missing from {archive_id}',
                         examples=sorted(missing)[:10])
        validation.check(not unexpected, 'unexpected_extracted_files',
                         f'Unexpected files in {archive_id}', examples=sorted(unexpected)[:10])
        for name in files:
            previous = image_index.get(name)
            validation.check(previous is None, 'duplicate_extracted_basename',
                             f'Duplicate basename across archives: {name}',
                             previous=previous, current=archive_id)
            image_index[name] = archive_id
        if deep and not missing:
            validate_tar_contents(tarpath, final, files, validation)
        batch_details.append({'archive': archive_id, 'files': len(files),
                              'physical_files': len(actual), 'deep_checked': deep and not missing})
    validation.details['archives'] = batch_details
    return image_index


def completed_files(tarpath, data):
    # Share preparation's exact marker/manifest rules.
    return completed_archive_files(tarpath, data / 'devanagari_ocr', PREFIXES)


def validate_provenance(data, prepared, archives, report, validation):
    source_path = prepared / 'dataset_source.json'
    validation.check(source_path.is_file(), 'missing_provenance',
                     f'Missing {source_path}')
    if not source_path.is_file():
        return
    source = json.loads(source_path.read_text(encoding='utf-8'))
    sources = source if isinstance(source, list) else [source]
    validation.check(len(sources) == len(archives), 'provenance_count',
                     f'Expected {len(archives)} provenance records, found {len(sources)}')
    for entry in sources:
        revision = entry.get('revision')
        validation.check(isinstance(revision,str) and bool(revision.strip()),
                         'provenance_revision_missing', 'Provenance revision must be a nonempty string')
    revisions = {entry.get('revision') for entry in sources}
    validation.check(len(revisions) == 1, 'provenance_revisions',
                     f'Expected one dataset revision, found {sorted(map(str,revisions))}')
    by_archive = {}
    for entry in sources:
        for filename in entry.get('files', []):
            if filename.endswith('.tar.gz'):
                by_archive[Path(filename).name] = entry
    for tarpath in archives:
        entry = by_archive.get(tarpath.name)
        validation.check(entry is not None, 'missing_archive_provenance',
                         f'No provenance for {tarpath.name}')
        if entry is not None:
            validation.check(entry.get('repo') == REPO and
                             set(entry.get('files', [])) == {'devanagari_ocr.json', f'data/{tarpath.name}'},
                             'archive_provenance_mismatch',
                             f'Unexpected provenance for {tarpath.name}')
    validation.check((data / 'devanagari_ocr.json').is_file(),
                     'missing_annotations', f'Missing {data / "devanagari_ocr.json"}')
    archive_metadata = {entry['name']: entry['bytes'] for entry in report.get('archives', [])}
    for tarpath in archives:
        validation.check(tarpath.is_file() and archive_metadata.get(tarpath.name) == tarpath.stat().st_size,
                         'report_archive_mismatch', f'Archive size differs from report: {tarpath.name}')


def validate_annotations(data, report, image_index, label_digests, validation):
    path = data / 'devanagari_ocr.json'
    if not path.is_file():
        return
    records = 0
    not_single_image = 0
    matched = set()
    with tqdm(annotations(path), desc='Validating annotations', unit='record', mininterval=1.0) as stream:
        for record in stream:
            records += 1
            paths = record.get('images')
            if paths is None:
                paths = [record['image']] if isinstance(record.get('image'), str) else []
            if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
                not_single_image += 1
                continue
            name = Path(paths[0]).name
            if name not in image_index:
                continue
            validation.check(name not in matched, 'duplicate_annotation',
                             f'Duplicate selected annotation: {name}')
            matched.add(name)
            if name in label_digests:
                validation.check(label_digests[name] == text_digest(text_of(record)),
                                 'annotation_label_mismatch',
                                 f'Prepared transcription differs from annotation: {name}')
    stats = report.get('stats', {})
    validation.check(records == stats.get('annotation_records'), 'annotation_count',
                     f'Annotation records differ: computed={records}, report={stats.get("annotation_records")}')
    validation.check(not_single_image == stats.get('not_single_image', 0), 'annotation_shape_count',
                     'Unsupported annotation record count differs from report.json')
    validation.check(len(matched) == stats.get('matched_annotation_images'), 'matched_annotation_count',
                     f'Matched annotations differ: computed={len(matched)}, '
                     f'report={stats.get("matched_annotation_images")}')
    orphan_count = sum(1 for name in image_index if name not in matched)
    validation.check(orphan_count == 0, 'actual_orphan_images',
                     f'Extracted images have no selected annotation: {orphan_count}',
                     examples=heapq.nsmallest(10,(name for name in image_index if name not in matched)))


def validate_prepared(data, prepared, archives, image_index, validation):
    required = ['train.txt', 'val.txt', 'test.txt', 'splits.csv', 'hindi_dict.txt',
                'report.json', 'crop_preview.jpg']
    for name in required:
        validation.check((prepared / name).is_file(), 'missing_prepared_file',
                         f'Missing prepared file: {prepared / name}')
    if not all((prepared / name).is_file() for name in required):
        return
    report = json.loads((prepared / 'report.json').read_text(encoding='utf-8'))
    validation.check(report.get('layout') == 'per_archive', 'report_layout',
                     f"Expected per_archive layout, found {report.get('layout')!r}")
    validation.check(list(report.get('prefixes', [])) == list(PREFIXES), 'report_prefixes',
                     'Prepared report prefixes differ from prepare.py')
    max_length = report.get('max_length')
    valid_max_length = type(max_length) is int and 1 <= max_length <= 40
    validation.check(valid_max_length and report.get('ctc_time_steps') == 40,
                     'report_length_policy', 'Unexpected text/CTC length policy')
    if not valid_max_length:
        max_length = 40
    stats = report.get('stats', {})
    validation.check(stats.get('extracted_hindi') == len(image_index), 'report_image_count',
                     f"Report has {stats.get('extracted_hindi')} extracted images; found {len(image_index)}")
    validation.check(stats.get('orphan_images') == 0, 'orphan_images',
                     f"Report lists {stats.get('orphan_images')} orphan images")
    dictionary = (prepared / 'hindi_dict.txt').read_text(encoding='utf-8').splitlines()
    validation.check(file_sha256(prepared / 'hindi_dict.txt') == report.get('dictionary_sha256'),
                     'dictionary_hash', 'Dictionary hash differs from report.json')
    vocab = set(dictionary) | {' '}
    counts, per_prefix, oov = {}, {}, {}
    label_digests = {}
    train_chars = set()
    with (prepared / 'splits.csv').open(newline='', encoding='utf-8') as csv_file:
        rows = csv.reader(csv_file)
        validation.check(next(rows, None) == ['image_path', 'split'], 'split_csv_header',
                         'splits.csv must start with image_path,split')
        for split in ('test', 'val', 'train'):
            split_counts = Counter()
            split_oov = set()
            row_count = 0
            with (prepared / f'{split}.txt').open(encoding='utf-8') as labels:
                for line_number, line in enumerate(labels, start=1):
                    row_count += 1
                    parts = line.rstrip('\n').split('\t')
                    valid_row = line.endswith('\n') and len(parts) == 2 and all(parts)
                    validation.check(valid_row, 'invalid_label_row',
                                     f'{split}.txt:{line_number} is not path<TAB>text')
                    if not valid_row:
                        continue
                    image_path, text = parts
                    csv_row = next(rows, None)
                    validation.check(csv_row == [image_path, split], 'split_csv_mismatch',
                                     f'splits.csv differs at {split}.txt:{line_number}',
                                     csv_row=csv_row)
                    path = Path(image_path)
                    validation.check(path.name not in label_digests, 'duplicate_label_path',
                                     f'Duplicate/cross-split image basename: {image_path}')
                    label_digests[path.name] = text_digest(text)
                    archive_id = image_index.get(path.name)
                    valid_path = (len(path.parts) == 3 and path.parts[0] == 'devanagari_ocr'
                                  and path.name.startswith(PREFIXES) and archive_id is not None
                                  and image_path == f'devanagari_ocr/{archive_id}/{path.name}')
                    validation.check(valid_path, 'invalid_image_path',
                                     f'Path is not in the verified per-archive extraction: {image_path}')
                    valid_text = (text == text.strip() and text == unicodedata.normalize('NFC', text)
                                  and not any(c in text for c in '\t\r\n\x00')
                                  and any('\u0900' <= c <= '\u097f' for c in text)
                                  and len(text) <= max_length and ctc_steps(text) <= 40)
                    validation.check(valid_text, 'invalid_label_text',
                                     f'Invalid text at {split}.txt:{line_number}')
                    split_oov.update(set(text) - vocab)
                    if split == 'train':
                        train_chars.update(text)
                    prefix = next((p for p in PREFIXES if path.name.startswith(p)), None)
                    if prefix is not None:
                        split_counts[prefix] += 1
            counts[split] = row_count
            per_prefix[split] = dict(split_counts)
            oov[split] = sorted(split_oov)
            validation.check(file_sha256(prepared / f'{split}.txt') ==
                             report.get('label_sha256', {}).get(split), 'label_hash',
                             f'{split}.txt hash differs from report.json')
        validation.check(next(rows, None) is None, 'extra_split_csv_rows',
                         'splits.csv has rows beyond the three label files')
    validation.check(counts == report.get('counts'), 'split_counts',
                     f'Split counts differ: computed={counts}, report={report.get("counts")}')
    validation.check(per_prefix == report.get('per_prefix'), 'split_prefix_counts',
                     'Per-prefix split counts differ from report.json')
    validation.check(oov == report.get('out_of_vocabulary'), 'report_oov',
                     'Out-of-vocabulary lists differ from report.json')
    validation.check(not any(oov.values()), 'out_of_vocabulary',
                     f'Out-of-vocabulary characters found: {oov}')
    total = sum(counts.values())
    n_test = counts.get('test', 0)
    n_val = counts.get('val', 0)
    validation.check(n_test == round(total * .1) and n_val == round(total * .1),
                     'split_ratio', f'Unexpected 80/10/10 split rounding: {counts}')
    expected_chars = (set(chr(i) for i in range(0x900, 0x980))
                      | set(chr(i) for i in range(33, 127))
                      | {c for c in train_chars if c != ' '})
    expected_chars.discard(' ')
    validation.check(dictionary == sorted(expected_chars), 'dictionary_contents',
                     'Dictionary is not the expected predefined inventory plus training characters')
    validation.details['prepared'] = {'counts': counts, 'per_prefix': per_prefix,
                                      'unique_label_paths': len(label_digests)}
    validate_provenance(data, prepared, archives, report, validation)
    validate_annotations(data, report, image_index, label_digests, validation)


def check_data(data, archives, prepared, deep=False, max_errors=50):
    validation = Validation(max_errors)
    try:
        image_index = validate_archives(data, archives, validation, deep)
    except Exception as exc:
        image_index = {}
        validation.check(False, 'archive_validation_exception', repr(exc))
    try:
        validate_prepared(data, prepared, archives, image_index, validation)
    except Exception as exc:
        validation.check(False, 'prepared_validation_exception', repr(exc))
    return validation.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--prepared', type=Path, default=ROOT / 'prepared')
    parser.add_argument('--deep', action='store_true',
                        help='Also compare every selected tar member with extracted bytes (slow)')
    parser.add_argument('--report', type=Path,
                        help='JSON report path (default: <prepared>/check_data_report.json)')
    parser.add_argument('--max-errors', type=int, default=50,
                        help='Maximum detailed errors to include in the report')
    args = parser.parse_args()
    if args.max_errors < 0:
        parser.error('--max-errors must be non-negative')
    args.data_dir = args.data_dir.resolve()
    args.prepared = args.prepared.resolve()
    archives = [args.data_dir / 'data' / f'images_batch_{batch:03d}.tar.gz'
                for batch in range(1, 43)]
    result = check_data(args.data_dir, archives, args.prepared, args.deep, args.max_errors)
    report_path = args.report or args.prepared / 'check_data_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'Wrote {report_path}')
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
