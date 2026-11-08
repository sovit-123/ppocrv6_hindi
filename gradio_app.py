"""Gradio UI for testing the PP-OCRv6 Hindi recognizer.

Upload one or more Hindi word/line crops and press "Run inference". For every
image this app shells out to this project's only entry point:

    python run.py infer --model finetuned|official --image <uploaded image> --device gpu|cpu

`finetuned` (default) loads the trained Hindi checkpoint from the configured
output directory (`output/hindi_rec/best_accuracy`). `official` loads the
stock PP-OCRv6 medium pretrained weights (Chinese/English dictionary; it
cannot read Devanagari) so people can compare against the pretrained model.
This file never imports Paddle, never calls any other run.py mode, and never
touches training files. run.py rewrites `predictions.txt` on every call; this
app simply parses that file for display.

Run it with the same Python environment used for run.py, plus gradio
(`python -m pip install gradio`; not part of requirements.txt):

    python gradio_app.py

`launch()` binds to 127.0.0.1:7860 by default; set GRADIO_SERVER_NAME and
GRADIO_SERVER_PORT to expose it elsewhere.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
RUN_PY = ROOT / 'run.py'
PREDICTIONS = ROOT / 'predictions.txt'
COMMAND_TIMEOUT = 600  # seconds per image, model load included


def stage_png(source, work, index):
    """Rewrite an upload as `<work>/upload_NNN.png` and return that path.

    A real PNG with a proper suffix always passes upstream's extension filter
    (ppocr.utils.utility.get_image_file_list), lets any Gradio-supported
    format (webp, etc.) through, and rejects undecodable files up front.
    """
    target = work / f'upload_{index:03d}.png'
    with Image.open(source) as image:
        image.convert('RGB').save(target, format='PNG')
    return target


def read_last_prediction():
    """Return (text, score) from the last `predictions.txt` row.

    run.py truncates and rewrites the file per infer call, so the last row is
    always the image just processed. Rows look like:
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


def infer_one(source, staged, model, device):
    """Run `run.py infer` for one staged image; return (result row, log lines)."""
    command = [sys.executable, str(RUN_PY), 'infer', '--model', model,
               '--image', str(staged), '--device', device]
    log = ['+ ' + ' '.join(command)]
    try:
        done = subprocess.run(command, cwd=ROOT, capture_output=True, encoding='utf-8',
                              errors='replace', timeout=COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired:
        log.append(f'[run.py timed out after {COMMAND_TIMEOUT}s]')
        return [source.name, '', '', f'timeout after {COMMAND_TIMEOUT}s'], log
    for stream in ('stdout', 'stderr'):
        text = (getattr(done, stream) or '').strip()
        if text:
            log.append(text)
    if done.returncode != 0:
        return [source.name, '', '', f'run.py exited with code {done.returncode}'], log
    try:
        text, score = read_last_prediction()
    except Exception as exc:
        return [source.name, '', '', f'no prediction parsed: {exc}'], log
    return [source.name, text, score, 'ok'], log


def infer_uploads(files, model, device):
    """Predict every uploaded image via run.py infer, streaming results out."""
    sources = [Path(f) for f in (files or [])]
    if not sources:
        raise gr.Error('Upload at least one Hindi crop image first.')
    model, device = model or 'finetuned', device or 'gpu'
    rows, log = [], []
    with tempfile.TemporaryDirectory(prefix='ppocrv6_gradio_') as tmp:
        work = Path(tmp)
        for index, source in enumerate(sources, start=1):
            try:
                staged = stage_png(source, work, index)
            except Exception as exc:
                rows.append([source.name, '', '', f'not a readable image: {exc}'])
                log.append(f'# {source.name}: not a readable image ({exc})')
            else:
                row, lines = infer_one(source, staged, model, device)
                rows.append(row)
                log.extend(lines)
            yield rows, '\n'.join(log)


def build_app():
    with gr.Blocks(title='PP-OCRv6 Hindi recognizer') as demo:
        gr.Markdown(
            '# PP-OCRv6 Hindi recognition\n'
            'Upload Hindi word or short-line crops. Each image is inferred by '
            '`python run.py infer`, which runs the pinned PaddleOCR checkout. '
            '`finetuned` loads the trained Hindi checkpoint from the output '
            'directory (`best_accuracy`); `official` loads the stock '
            'PP-OCRv6 medium pretrained weights, whose Chinese/English '
            'dictionary cannot read Devanagari. Nothing is trained or modified here.'
        )
        with gr.Row():
            uploads = gr.File(label='Hindi crop images (multiple)',
                              file_count='multiple', file_types=['image'])
            device = gr.Radio(choices=['gpu', 'cpu'], value='gpu', label='Device')
        model = gr.Radio(choices=['finetuned', 'official'], value='finetuned',
                          label='Model',
                          info='finetuned: trained Hindi checkpoint (default). '
                               'official: stock PP-OCRv6 medium pretrained '
                               '(Chinese/English; expect empty or wrong text '
                               'on Hindi crops).')
        run_btn = gr.Button('Run inference', variant='primary')
        results = gr.Dataframe(headers=['image', 'predicted text', 'confidence', 'status'],
                               datatype=['str', 'str', 'str', 'str'],
                               label='Predictions', wrap=True)
        log = gr.Textbox(label='run.py console log', lines=14, max_lines=40,
                         autoscroll=True, interactive=False)
        run_btn.click(infer_uploads, inputs=[uploads, model, device],
                      outputs=[results, log], api_name='infer',
                      concurrency_limit=1)
    return demo


if __name__ == '__main__':
    build_app().launch()
