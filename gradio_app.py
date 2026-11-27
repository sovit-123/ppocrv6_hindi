"""Gradio UI for testing PP-OCRv6 Hindi recognition checkpoints.

Upload crops/full pages or provide an image directory. The official medium
detector runs by default with the custom recognizer; the cropped checkbox
bypasses it. Uploads and directory contents may be combined. Routes:

- `finetuned` uses the exported inference bundle from --inference-dir and
  supports detector-based full-page OCR and direct cropped recognition. The
  pipeline is cached per recognizer directory and device.
- `checkpoint` runs a portable fine-tuned training checkpoint through
  `run.py infer --checkpoint`, separately per image. It is crop-only.
- `official` runs the stock Chinese/English medium recognizer for cropped
  comparison and is also crop-only.

Dynamic crop routes rewrite `predictions.txt`. Detector-based results include
per-region rows, annotated images, and structured JSON. Nothing is trained.

Run with the same Python environment used for run.py, plus gradio
(`python -m pip install gradio`; opencv-python is in requirements-training.txt).
`launch()` binds to 127.0.0.1:7860 by default; set GRADIO_SERVER_NAME and
GRADIO_SERVER_PORT to expose it elsewhere.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

import cv2
import gradio as gr
from PIL import Image

from run import DET_MODEL_NAME, download_detector, input_files, recognizer_model_name

ROOT = Path(__file__).resolve().parent
RUN_PY = ROOT / 'run.py'
PREDICTIONS = ROOT / 'predictions.txt'
PADDLE_ROOT = ROOT / 'third_party' / 'PaddleOCR'
INFER_DIR = Path(os.environ.get('PPOCRV6_INFER_DIR', ROOT / 'output' / 'hindi_rec_medium' / 'inference'))
CHECKPOINT = None
COMMAND_TIMEOUT = 600  # seconds per dynamic-route image, model load included
_RECOGNIZERS = {}  # device -> TextRecognizer built from the exported bundle
_PAGE_OCR = None
_PAGE_KEY = None


def stage_png(source, work, index):
    """Rewrite an upload as `<work>/upload_NNN.png` and return that path.

    A real PNG with a proper suffix always passes upstream's extension filter
    (ppocr.utils.utility.get_image_file_list) on the official route, lets any
    Gradio-supported format (webp, etc.) through, and rejects undecodable
    files up front for both routes.
    """
    target = work / f'upload_{index:03d}.png'
    with Image.open(source) as image:
        image.convert('RGB').save(target, format='PNG')
    return target


def read_last_prediction():
    """Return (text, score) from the last `predictions.txt` row.

    run.py truncates and rewrites the file per dynamic infer call, so the
    last row is always the image just processed. Rows look like:
    `<image path>\t<predicted text>\t<score>`.
    """
    lines = [line for line in PREDICTIONS.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines:
        raise RuntimeError('predictions.txt is empty')
    parts = lines[-1].split('\t', 2)
    if len(parts) != 3:
        raise RuntimeError(f'unparsable prediction row: {lines[-1]!r}')
    text, raw_score = parts[1], parts[2]
    try:
        score = f'{float(raw_score):.4f}'
    except ValueError:
        score = raw_score
    return text, score


def exported_bundle_problem():
    """Return a user-facing error string, or None if the bundle looks usable."""
    if not (INFER_DIR / 'inference.pdiparams').is_file():
        return (f'Missing {INFER_DIR / "inference.pdiparams"}. Run '
                '`python run.py export` after training to build the exported '
                'inference model.')
    if not ((INFER_DIR / 'inference.pdmodel').is_file() or (INFER_DIR / 'inference.json').is_file()):
        return (f'Missing the model file under {INFER_DIR}. Run '
                '`python run.py export` after training to build the exported '
                'inference model.')
    if not (INFER_DIR / 'inference.yml').is_file():
        return (f'Missing {INFER_DIR / "inference.yml"} (it embeds the Hindi '
                'dictionary). Run `python run.py export` to rebuild the bundle.')
    if not PADDLE_ROOT.is_dir():
        return (f'Missing the pinned PaddleOCR checkout at {PADDLE_ROOT}; '
                'run `python run.py setup` first.')
    return None


def checkpoint_bundle_problem():
    """Return a user-facing error for an incomplete portable checkpoint bundle."""
    if CHECKPOINT is None:
        return 'Pass `--checkpoint /path/to/checkpoint.pdparams` to use the checkpoint route.'
    prefix = str(CHECKPOINT).removesuffix('.pdparams')
    checkpoint = Path(prefix + '.pdparams')
    if not checkpoint.is_file():
        return f'Missing checkpoint {checkpoint}'
    for name in ('checkpoint_config.yml', 'hindi_dict.txt', 'initialization.json'):
        path = checkpoint.parent / name
        if not path.is_file():
            return f'Missing portable checkpoint file {path}'
    if not PADDLE_ROOT.is_dir():
        return (f'Missing the pinned PaddleOCR checkout at {PADDLE_ROOT}; '
                'run `python run.py setup` first.')
    return None


def load_recognizer(device):
    """Return the pinned-checkout recognizer for the exported bundle.

    Cached per device: the static model loads once in seconds instead of the
    dynamic route's per-subprocess weight load. Upstream reads the Hindi
    dictionary from the bundle's `inference.yml` (export embeds it) and writes
    it out as `ppocr_keys.txt` inside the bundle on first load.
    """
    problem = exported_bundle_problem()
    if problem:
        raise gr.Error(problem)
    if device not in _RECOGNIZERS:
        if str(PADDLE_ROOT) not in sys.path:
            sys.path.insert(0, str(PADDLE_ROOT))
        import tools.infer.utility as utility
        from tools.infer.predict_rec import TextRecognizer
        # Reuse upstream's parser defaults (dictionary fallback, image shape,
        # precision, cpu threads, ...) so only the model dir and device vary.
        argv = sys.argv
        sys.argv = ['predict_rec.py',
                    '--rec_model_dir', str(INFER_DIR),
                    '--use_gpu', str(device == 'gpu').lower()]
        try:
            args = utility.parse_args()
        finally:
            sys.argv = argv
        _RECOGNIZERS[device] = TextRecognizer(args)
    return _RECOGNIZERS[device]


def load_page_ocr(device):
    """Load and cache the medium-detector plus custom-recognizer pipeline."""
    global _PAGE_OCR, _PAGE_KEY
    problem = exported_bundle_problem()
    if problem:
        raise gr.Error(problem)
    rec_dir = INFER_DIR.resolve()
    key = (str(rec_dir), device)
    if _PAGE_OCR is None or _PAGE_KEY != key:
        if _PAGE_OCR is not None:
            _PAGE_OCR.close()
            _PAGE_OCR = None
            _PAGE_KEY = None
        rec_name = recognizer_model_name(rec_dir)
        detector_dir = download_detector()
        if str(PADDLE_ROOT) not in sys.path:
            sys.path.insert(0, str(PADDLE_ROOT))
        from paddleocr import PaddleOCR
        _PAGE_OCR = PaddleOCR(
            text_detection_model_name=DET_MODEL_NAME,
            text_detection_model_dir=str(detector_dir),
            text_recognition_model_name=rec_name,
            text_recognition_model_dir=str(rec_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )
        _PAGE_KEY = key
    return _PAGE_OCR


def infer_page(source, staged, device):
    """Run full-page detection and recognition, returning regions and artifacts."""
    rows = []
    images = []
    page_data = []
    log = [f'+ [medium detector + custom recognizer] {staged.name} (device={device})']
    try:
        ocr = load_page_ocr(device)
        destination_root = ROOT / 'output' / 'gradio_ocr' / uuid.uuid4().hex
        for page_number, result in enumerate(ocr.predict_iter(str(staged)), start=1):
            destination = destination_root / f'page_{page_number:03d}'
            destination.mkdir(parents=True, exist_ok=True)
            result.save_to_img(str(destination))
            result.save_to_json(str(destination))
            texts = list(result['rec_texts'])
            scores = list(result['rec_scores'])
            for region_number, (text, score) in enumerate(zip(texts, scores), start=1):
                rows.append([source.name, f'page {page_number}, region {region_number}',
                             text, f'{float(score):.4f}', 'ok'])
            images.extend(str(path) for path in sorted(destination.rglob('*.png')))
            images.extend(str(path) for path in sorted(destination.rglob('*.jpg')))
            page_data.append(result.json)
            log.append(f'Page {page_number}: {len(texts)} regions; saved to {destination}')
    except Exception as exc:
        log.append(traceback.format_exc())
        rows.append([source.name, '', '', '', f'predictor error: {exc}'])
    return rows, log, images, page_data


def infer_finetuned(source, staged, device):
    """Run one staged image through the exported inference model in-process."""
    log = [f'+ [exported inference model] {INFER_DIR} (device={device}) :: {staged.name}']
    img = cv2.imread(str(staged))
    if img is None:
        return [source.name, 'whole image', '', '', f'not a readable image: cv2.imread({staged.name}) failed'], log
    try:
        rec_res, _ = load_recognizer(device)([img])
        text, score = rec_res[0]
    except Exception as exc:
        log.append(traceback.format_exc())
        return [source.name, 'whole image', '', '', f'predictor error: {exc}'], log
    return [source.name, 'whole image', text, f'{score:.4f}', 'ok'], log


def infer_subprocess(source, command):
    """Run one dynamic inference command and parse its prediction row."""
    log = ['+ ' + ' '.join(command)]
    try:
        done = subprocess.run(command, cwd=ROOT, capture_output=True, encoding='utf-8',
                              errors='replace', timeout=COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired:
        log.append(f'[run.py timed out after {COMMAND_TIMEOUT}s]')
        return [source.name, 'whole image', '', '', f'timeout after {COMMAND_TIMEOUT}s'], log
    for stream in ('stdout', 'stderr'):
        text = (getattr(done, stream) or '').strip()
        if text:
            log.append(text)
    if done.returncode != 0:
        return [source.name, 'whole image', '', '', f'run.py exited with code {done.returncode}'], log
    try:
        text, score = read_last_prediction()
    except Exception as exc:
        return [source.name, 'whole image', '', '', f'no prediction parsed: {exc}'], log
    return [source.name, 'whole image', text, score, 'ok'], log


def infer_official(source, staged, device):
    """Run `run.py infer --model official` for one staged image (stock weights)."""
    command = [sys.executable, str(RUN_PY), 'infer', '--model', 'official',
               '--image', str(staged), '--device', device]
    return infer_subprocess(source, command)


def infer_checkpoint(source, staged, device):
    """Run one staged image through a portable fine-tuned training checkpoint."""
    command = [sys.executable, str(RUN_PY), 'infer', '--checkpoint', str(CHECKPOINT),
               '--image', str(staged), '--device', device]
    return infer_subprocess(source, command)


def infer_uploads(files, model, device, cropped, directory):
    """Process uploads and/or an image directory, with detection by default."""
    sources = [Path(f) for f in (files or [])]
    if directory and directory.strip():
        try:
            sources.extend(path for path, _ in input_files(Path(directory.strip()).expanduser()))
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
    unique_sources = []
    seen = set()
    for source in sources:
        resolved = source.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_sources.append(resolved)
    sources = unique_sources
    if not sources:
        raise gr.Error('Upload at least one Hindi crop/page image or provide an image directory.')
    model = model or ('checkpoint' if CHECKPOINT is not None else 'finetuned')
    device = device or 'gpu'
    cropped = bool(cropped) or model != 'finetuned'
    if model == 'finetuned':
        if cropped:
            load_recognizer(device)
        else:
            load_page_ocr(device)
    elif model == 'checkpoint':
        problem = checkpoint_bundle_problem()
        if problem:
            raise gr.Error(problem)
    rows, log, images, page_data = [], [], [], []
    with tempfile.TemporaryDirectory(prefix='ppocrv6_gradio_') as tmp:
        work = Path(tmp)
        for index, source in enumerate(sources, start=1):
            try:
                staged = stage_png(source, work, index)
            except Exception as exc:
                rows.append([source.name, 'whole image', '', '', f'not a readable image: {exc}'])
                log.append(f'# {source.name}: not a readable image ({exc})')
            else:
                if model == 'finetuned' and not cropped:
                    page_rows, lines, page_images, page_json = infer_page(source, staged, device)
                    rows.extend(page_rows)
                    images.extend(page_images)
                    page_data.extend(page_json)
                else:
                    if model == 'finetuned':
                        row, lines = infer_finetuned(source, staged, device)
                    elif model == 'checkpoint':
                        row, lines = infer_checkpoint(source, staged, device)
                    else:
                        row, lines = infer_official(source, staged, device)
                    rows.append(row)
                log.extend(lines)
            yield rows, '\n'.join(log), images, page_data


def crop_mode_for_model(model):
    """Only exported fine-tuned models support detector-based full-page OCR."""
    return gr.Checkbox(value=model != 'finetuned', interactive=model == 'finetuned',
                       label='Input is already cropped (--cropped)')


def build_app():
    model_choices = ['finetuned', 'official']
    model_default = 'finetuned'
    if CHECKPOINT is not None:
        model_choices.insert(0, 'checkpoint')
        model_default = 'checkpoint'
    with gr.Blocks(title='PP-OCRv6 Hindi recognizer') as demo:
        gr.Markdown(
            '# PP-OCRv6 Hindi recognition\n'
            'Upload a crop or full page. The official medium detector runs by '
            'default with your custom recognizer. Select **Input is already '
            'cropped** to bypass detection. There is no language selection.\n\n'
            '- **finetuned**: exported Hindi inference bundle; supports full-page '
            'detector+recognizer and direct cropped recognition.\n'
            '- **checkpoint**: portable fine-tuned training checkpoint for direct '
            'cropped inference without export.\n'
            '- **official**: stock Chinese/English medium recognizer for cropped '
            'comparison; it cannot read Devanagari.\n\n'
            'Nothing is trained here.'
        )
        with gr.Row():
            uploads = gr.File(label='Hindi crop or page images (multiple)',
                              file_count='multiple', file_types=['image'])
            device = gr.Radio(choices=['gpu', 'cpu'], value='gpu', label='Device')
        model = gr.Radio(choices=model_choices, value=model_default,
                         label='Model',
                         info='checkpoint: portable fine-tuned weights without export. '
                              'finetuned: cached exported inference model. official: stock '
                              'pretrained checkpoint via run.py.')
        directory = gr.Textbox(label='Image directory (optional)',
                               placeholder='/absolute/path/to/images',
                               info='Optional directory of images/documents; may be combined with uploads.')
        cropped = crop_mode_for_model(model_default)
        model.change(crop_mode_for_model, inputs=model, outputs=cropped)
        run_btn = gr.Button('Run inference', variant='primary')
        results = gr.Dataframe(headers=['image', 'region', 'predicted text', 'confidence', 'status'],
                               datatype=['str', 'str', 'str', 'str', 'str'],
                               label='Predictions', wrap=True)
        page_images = gr.Gallery(label='Annotated pages')
        page_json = gr.JSON(label='Full-page OCR results')
        log = gr.Textbox(label='console log', lines=14, max_lines=40,
                         autoscroll=True, interactive=False)
        run_btn.click(infer_uploads, inputs=[uploads, model, device, cropped, directory],
                      outputs=[results, log, page_images, page_json], api_name='infer',
                      concurrency_limit=1)
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        help='Portable fine-tuned .pdparams checkpoint (no export required)')
    parser.add_argument('--inference-dir', type=Path, default=INFER_DIR,
                        help='Exported inference bundle directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    INFER_DIR = args.inference_dir.resolve()
    CHECKPOINT = args.checkpoint.resolve() if args.checkpoint else None
    build_app().launch()
