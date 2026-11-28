"""Configure and run upstream PP-OCRv6 recognition tools."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import yaml

ROOT = Path(__file__).resolve().parent
PADDLE_REF = 'dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf'
DEFAULT_UPSTREAM = ROOT/'third_party/PaddleOCR'
WEIGHTS = ROOT/'weights'
MODEL_SIZES = ('tiny','small','medium')
PRETRAINED_FILENAME = 'PP-OCRv6_medium_rec_pretrained.pdparams'
INFERENCE = ROOT/'inference'
DET_MODEL_NAME = 'PP-OCRv6_medium_det'
DET_MODEL_SIZES = ('tiny','small','medium')
RESET_PREFIXES = ('head.ctc_head.fc.', 'head.ctc_head.fc2.',
                  'head.gtc_head.embedding.', 'head.gtc_head.tgt_word_prj.')


def shell(command, cwd=None):
    print('+', ' '.join(map(str,command)), flush=True)
    subprocess.run(list(map(str,command)), cwd=cwd, check=True)


def validate_upstream(upstream):
    if not upstream.is_dir():
        raise ValueError(f'Missing {upstream}; run `python run.py setup` first.')
    try:
        actual = subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()
        dirty = subprocess.check_output(['git','status','--porcelain'],cwd=upstream,text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f'Cannot inspect upstream checkout {upstream}; use a separate checkout.') from exc
    if actual != PADDLE_REF:
        raise ValueError(f'Expected PaddleOCR {PADDLE_REF}; found {actual}. Use a separate checkout.')
    if dirty:
        raise ValueError('Upstream checkout has local changes; use a clean checkout.')


def pretrained_filename(model_size):
    if model_size not in MODEL_SIZES:
        raise ValueError(f'Unknown model size {model_size!r}; choose from {MODEL_SIZES}.')
    return f'PP-OCRv6_{model_size}_rec_pretrained.pdparams'


def pretrained_url(model_size):
    return f'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/{pretrained_filename(model_size)}'


def download_weights(weights, model_size='medium'):
    weights.mkdir(parents=True,exist_ok=True)
    original = weights/pretrained_filename(model_size)
    if not original.exists():
        temp = original.with_suffix('.part')
        print(f'Downloading official {model_size} training weights...',flush=True)
        urllib.request.urlretrieve(pretrained_url(model_size),temp)
        temp.replace(original)
    if not original.is_file() or original.stat().st_size == 0:
        raise ValueError(f'Official {model_size} weights are missing or empty: {original}')
    return original


def inference_bundle_valid(path):
    return (path.is_dir() and (path/'inference.pdiparams').is_file()
            and (path/'inference.yml').is_file()
            and ((path/'inference.json').is_file() or (path/'inference.pdmodel').is_file()))


def detector_model_name(det_size):
    if det_size not in DET_MODEL_SIZES:
        raise ValueError(f'Unknown detector size {det_size!r}; choose from {DET_MODEL_SIZES}.')
    return f'PP-OCRv6_{det_size}_det'


def download_detector(inference=INFERENCE, det_size='medium'):
    """Download and atomically extract the selected official detector bundle."""
    model_name = detector_model_name(det_size)
    final = inference/model_name
    if inference_bundle_valid(final):
        return final
    inference.mkdir(parents=True,exist_ok=True)
    archive = inference/f'{model_name}_infer.tar'
    if not archive.exists():
        temp = archive.with_suffix('.part')
        print(f'Downloading official {det_size} detection model...',flush=True)
        url = ('https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/'
               f'paddle3.0.0/{model_name}_infer.tar')
        urllib.request.urlretrieve(url,temp)
        temp.replace(archive)
    part = inference/f'{model_name}.part'
    if part.is_symlink():
        part.unlink()
    elif part.is_dir():
        shutil.rmtree(part)
    elif part.exists():
        part.unlink()
    if final.is_symlink():
        final.unlink()
    elif final.is_dir():
        shutil.rmtree(final)
    elif final.exists():
        final.unlink()
    part.mkdir()
    with tarfile.open(archive,'r') as tar:
        for member in tar:
            member_path = Path(member.name)
            if (member_path.is_absolute() or '..' in member_path.parts
                    or member.issym() or member.islnk()):
                raise ValueError(f'Unsafe detector archive member: {member.name}')
        tar.extractall(part,filter='data')
    if not inference_bundle_valid(part):
        candidates = list(part.rglob('inference.yml'))
        if len(candidates) != 1:
            raise ValueError(f'Could not locate detector inference.yml in {archive}')
        bundle_root = candidates[0].parent
    else:
        bundle_root = part
    if bundle_root == part:
        part.replace(final)
    else:
        shutil.move(str(bundle_root),str(final))
        shutil.rmtree(part)
    if not inference_bundle_valid(final):
        raise ValueError(f'Incomplete detector inference bundle: {final}')
    return final


def require_setup(upstream=DEFAULT_UPSTREAM, weights=WEIGHTS, model_size=None):
    """Validate setup prerequisites without downloading or modifying anything."""
    validate_upstream(upstream)
    for size in MODEL_SIZES if model_size is None else (model_size,):
        original = weights/pretrained_filename(size)
        if original.is_file() and original.stat().st_size > 0:
            return original
    raise ValueError(f'Setup is incomplete: no usable official weights in {weights}. '
                     'Run `python run.py setup --model-size {tiny,small,medium}` first.')


def setup(upstream, weights=WEIGHTS, model_size='medium', det_size='medium'):
    if not upstream.exists():
        shell(['git','clone','https://github.com/PaddlePaddle/PaddleOCR.git',upstream])
        shell(['git','checkout',PADDLE_REF], upstream)
    validate_upstream(upstream)
    original = download_weights(weights,model_size)
    detector = download_detector(det_size=det_size)
    print(f'Pinned upstream and official {model_size} weights ready: {original}')
    print(f'Official {det_size} detector ready: {detector}')
    print('Install the training dependencies as described in README.')
    return original


def config_for(upstream, data, prepared, output, checkpoint, a, model_size='medium'):
    config_name = f'PP-OCRv6_{model_size}_rec.yml'
    c = yaml.safe_load((upstream/'configs/rec/PP-OCRv6'/config_name).read_text())
    c['Global'].update(character_dict_path=str(prepared/'hindi_dict.txt'),
        max_text_length=40, pretrained_model=str(checkpoint), checkpoints=None,
        save_model_dir=str(output), epoch_num=a.epochs, use_gpu=a.device=='gpu',
        use_space_char=True, distributed=False, seed=42, eval_batch_step=[0,a.eval_every],
        save_epoch_step=1, d2s_train_image_shape=[3,48,320])
    c['Optimizer']['lr']['learning_rate'] = a.lr
    c['Optimizer']['lr']['warmup_epoch'] = min(2, max(0,a.epochs-1))
    # Upstream training derives CTC/NRTR output sizes from the new dictionary.
    for head in c['Architecture']['Head']['head_list']:
        if 'NRTRHead' in head:
            head['NRTRHead']['max_text_length'] = 40
    transforms = [
        {'DecodeImage': {'img_mode':'BGR','channel_first':False}},
        {'MultiLabelEncode': {'gtc_encode':'NRTRLabelEncode'}},
        {'RecResizeImg': {'image_shape':[3,48,320]}},
        {'KeepKeys': {'keep_keys':['image','label_ctc','label_gtc','length','valid_ratio']}}]
    for split, name in [('Train','train'),('Eval','val')]:
        c[split] = {'dataset': {'name':'SimpleDataSet','data_dir':str(data),
                    'label_file_list':[str(prepared/f'{name}.txt')], 'ratio_list':[1.0],
                    'transforms':copy.deepcopy(transforms)},
                    'loader': {'shuffle':split=='Train','drop_last':False,
                               'batch_size_per_card':a.batch_size,'num_workers':a.workers}}
    # Keep Hindi spaces, case, digits and punctuation significant in upstream metrics.
    c['Metric'].update(ignore_space=False, is_filter=False)
    return c


def checkpoint_config_for(config):
    """Return an inference config with no absolute training-data dependencies."""
    c = copy.deepcopy(config)
    c['Global'].update(character_dict_path='hindi_dict.txt',pretrained_model=None,
                       checkpoints=None,save_model_dir='.',save_res_path='predictions.txt')
    for split in ('Train','Eval'):
        c[split]['dataset']['data_dir'] = '.'
        c[split]['dataset']['label_file_list'] = []
    return c


def init_checkpoint(upstream, config, experiment, model_size='medium', weights=WEIGHTS):
    import paddle
    sys.path.insert(0,str(upstream))
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process
    experiment.mkdir(parents=True,exist_ok=True)
    original = weights/pretrained_filename(model_size)
    if not original.is_file() or original.stat().st_size == 0:
        raise ValueError(f'Missing official {model_size} weights {original}; '
                         f'run `python run.py setup --model-size {model_size}` first.')
    c = copy.deepcopy(config)
    decoder = build_post_process(c['PostProcess'],c['Global'])
    n = len(decoder.character)
    c['Architecture']['Head']['out_channels_list'] = {'CTCLabelDecode':n,'NRTRLabelDecode':n+3}
    paddle.seed(42)
    paddle.set_device('cpu')
    model = build_model(c['Architecture'])
    target = model.state_dict()
    source = paddle.load(str(original))
    loaded, reset, mismatch, missing = [], [], [], []
    # Full initialized state avoids relying on missing-key behavior. Always reset
    # vocabulary layers even if a different dictionary happens to have equal size.
    for key in target:
        if key.startswith(RESET_PREFIXES):
            reset.append(key)
        elif key not in source:
            missing.append(key)
        elif list(target[key].shape) != list(source[key].shape):
            mismatch.append(key)
        else:
            target[key] = source[key].astype(target[key].dtype)
            loaded.append(key)
    if not reset or not any(k.startswith('backbone.') for k in loaded) or mismatch or missing:
        raise ValueError(f'Unexpected checkpoint layout: reset={reset}, mismatch={mismatch}, missing={missing}')
    adapted = experiment/'hindi_initialized.pdparams'
    paddle.save(target,str(adapted))
    report = {'model_size':model_size,'source_url':pretrained_url(model_size),
              'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
              'paddle_version':paddle.__version__, 'paddleocr_commit':PADDLE_REF,
              'loaded':loaded,'reset_vocabulary_parameters':reset,'ctc_classes':n,
              'dictionary_sha256':hashlib.sha256(Path(config['Global']['character_dict_path']).read_bytes()).hexdigest()}
    (experiment/'initialization.json').write_text(json.dumps(report,indent=2))
    # Actual training/eval forwards catch source/API/geometry incompatibilities.
    x = paddle.randn([2,3,48,320])
    model.train()
    preds = model(x, [paddle.ones([2,40],dtype='int64'),paddle.ones([2,42],dtype='int64'),paddle.to_tensor([2,2])])
    model.eval()
    with paddle.no_grad():
        result = model(x)
    if result.shape[1] != 40 or result.shape[-1] != n:
        raise ValueError(f'Unexpected eval shape {result.shape}; expected [2,40,{n}]')
    print(f'Initialized {len(loaded)} tensors; reset {reset}; eval shape={result.shape}')
    return adapted


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def subset_for(source, experiment, split, count):
    """Create and verify a deterministic first-N view of a prepared label file."""
    if count < 1:
        raise ValueError('Subset size must be positive.')
    source = source.resolve()
    with source.open(encoding='utf-8') as f:
        total = sum(1 for line in f if line.strip())
    if count > total:
        raise ValueError(f'Requested {count} {split} samples, but {source} has only {total}.')
    source_hash = file_sha256(source)
    subset = experiment/'subsets'/f'{split}_{count}.txt'
    metadata_path = subset.with_suffix('.json')
    expected = {'source_label_file':str(source),'source_sha256':source_hash,
                'source_rows':total,'split':split,'requested_count':count,
                'actual_count':count,'selection_rule':'first_n_of_prepared_shuffle'}
    if subset.exists() or metadata_path.exists():
        if not subset.is_file() or not metadata_path.is_file():
            raise ValueError(f'Incomplete subset artifacts: {subset} and {metadata_path}')
        saved = json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(saved.get(key) != value for key,value in expected.items()):
            raise ValueError(f'Subset metadata does not match its source: {metadata_path}')
        if file_sha256(subset) != saved.get('subset_sha256'):
            raise ValueError(f'Subset hash mismatch: {subset}')
        return subset
    subset.parent.mkdir(parents=True,exist_ok=True)
    temp = subset.with_suffix('.part')
    rows = 0
    with source.open(encoding='utf-8') as src, temp.open('w',encoding='utf-8',newline='') as out:
        for line in src:
            if not line.strip():
                continue
            out.write(line if line.endswith('\n') else line+'\n')
            rows += 1
            if rows == count:
                break
    if rows != count:
        raise ValueError(f'Only {rows} rows could be written for {subset}.')
    temp.replace(subset)
    metadata = dict(expected,subset_sha256=file_sha256(subset))
    metadata_temp = metadata_path.with_suffix('.part')
    metadata_temp.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    metadata_temp.replace(metadata_path)
    return subset


def verify_subset(subset):
    """Verify a configured subset and its original prepared source before resume."""
    subset = subset.resolve()
    metadata_path = subset.with_suffix('.json')
    if not subset.is_file() or not metadata_path.is_file():
        raise ValueError(f'Missing subset or metadata: {subset} and {metadata_path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    source = Path(metadata['source_label_file'])
    if file_sha256(source) != metadata['source_sha256']:
        raise ValueError(f'Prepared source changed after subset creation: {source}')
    if file_sha256(subset) != metadata['subset_sha256']:
        raise ValueError(f'Subset hash mismatch: {subset}')
    return subset


def recognizer_model_name(rec_model_dir):
    if not inference_bundle_valid(rec_model_dir):
        raise ValueError(f'Incomplete recognizer inference bundle: {rec_model_dir}')
    config = yaml.safe_load((rec_model_dir/'inference.yml').read_text(encoding='utf-8'))
    name = config.get('Global',{}).get('model_name')
    if name not in {f'PP-OCRv6_{size}_rec' for size in MODEL_SIZES}:
        raise ValueError(f'Unsupported custom recognition model: {name!r}')
    return name


def benchmark_report(pages, regions, model_load_s, inference_s, output_s, total_s):
    """Build a per-run pages/second report with separate timing stages."""
    return {'pages':pages,'regions':regions,'model_load_s':model_load_s,
            'inference_s':inference_s,'output_s':output_s,'total_s':total_s,
            'pages_per_second_inference':pages/inference_s if inference_s > 0 else None,
            'pages_per_second_total':pages/total_s if total_s > 0 else None,
            'regions_per_second_inference':regions/inference_s if inference_s > 0 else None}


def write_benchmark(output, metrics):
    output.mkdir(parents=True,exist_ok=True)
    (output/'benchmark.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('Benchmark:')
    print(f"  Pages: {metrics['pages']}; regions: {metrics['regions']}")
    print(f"  Model load: {metrics['model_load_s']:.3f}s; inference: {metrics['inference_s']:.3f}s; "
          f"output: {metrics['output_s']:.3f}s; total: {metrics['total_s']:.3f}s")
    if metrics['pages_per_second_inference'] is not None:
        print(f"  Pages/s (inference): {metrics['pages_per_second_inference']:.3f}")
    if metrics['pages_per_second_total'] is not None:
        print(f"  Pages/s (end-to-end): {metrics['pages_per_second_total']:.3f}")


INPUT_EXTENSIONS = {'.jpg','.jpeg','.png','.bmp','.gif','.webp','.tif','.tiff','.pdf'}


def input_files(image):
    """Return (path, output group) for one file or a directory of images/documents."""
    image = image.resolve()
    if image.is_file():
        return [(image,'')]
    if image.is_dir():
        files = sorted((path for path in image.iterdir()
                        if path.is_file() and path.suffix.lower() in INPUT_EXTENSIONS),
                       key=lambda path: path.name.lower())
        if not files:
            raise ValueError(f'No supported images/documents found in {image}')
        return [(path,f'{path.stem}{path.suffix.lower().replace(".","_")}') for path in files]
    raise ValueError(f'Input image/document/directory does not exist: {image}')


def run_page(upstream, rec_model_dir, image, output, device, det_size='medium'):
    """Run full-page OCR with the fixed medium detector and a custom recognizer."""
    total_start = time.perf_counter()
    inputs = input_files(image)
    rec_model_dir = rec_model_dir.resolve()
    rec_name = recognizer_model_name(rec_model_dir)
    validate_upstream(upstream)
    sys.path.insert(0,str(upstream))
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError('Full-page OCR requires paddlex[ocr-core]==3.7.2. '
                           'Install requirements-training.txt first.') from exc
    load_start = time.perf_counter()
    detector_dir = download_detector(det_size=det_size)
    ocr = PaddleOCR(
        text_detection_model_name=detector_model_name(det_size),
        text_detection_model_dir=str(detector_dir),
        text_recognition_model_name=rec_name,
        text_recognition_model_dir=str(rec_model_dir),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
    )
    model_load_s = time.perf_counter() - load_start
    page_count = regions = 0
    inference_s = output_s = 0.0
    try:
        for input_path,group in inputs:
            inference_start = time.perf_counter()
            results = list(ocr.predict_iter(str(input_path)))
            inference_s += time.perf_counter() - inference_start
            page_number = 0
            for result in results:
                page_number += 1
                page_count += 1
                destination = output/group/f'page_{page_number:03d}' if group else output/f'page_{page_number:03d}'
                output_start = time.perf_counter()
                destination.mkdir(parents=True,exist_ok=True)
                result.save_to_img(str(destination))
                result.save_to_json(str(destination))
                output_s += time.perf_counter() - output_start
                texts = list(result['rec_texts'])
                scores = list(result['rec_scores'])
                regions += len(texts)
                print(f'Page {page_count}: {len(texts)} text regions; saved to {destination}')
                for text,score in zip(texts,scores):
                    print(f'  {text}\t{float(score):.4f}')
    finally:
        ocr.close()
    total_s = time.perf_counter() - total_start
    print(f'Processed {page_count} page(s) from {image}')
    write_benchmark(output,benchmark_report(page_count,regions,model_load_s,inference_s,output_s,total_s))


def run_cropped(upstream, rec_model_dir, image, device):
    """Recognize one crop or a directory of crops without running the detector."""
    total_start = time.perf_counter()
    inputs = input_files(image)
    rec_model_dir = rec_model_dir.resolve()
    rec_name = recognizer_model_name(rec_model_dir)
    validate_upstream(upstream)
    sys.path.insert(0,str(upstream))
    try:
        from paddleocr import TextRecognition
    except ImportError as exc:
        raise RuntimeError('Full-page OCR requires paddlex[ocr-core]==3.7.2. '
                           'Install requirements-training.txt first.') from exc
    load_start = time.perf_counter()
    recognizer = TextRecognition(model_name=rec_name,model_dir=str(rec_model_dir),device=device)
    model_load_s = time.perf_counter() - load_start
    page_count = regions = 0
    inference_s = output_s = 0.0
    try:
        with (ROOT/'predictions.txt').open('w',encoding='utf-8') as output:
            for input_path,_ in inputs:
                inference_start = time.perf_counter()
                results = recognizer.predict(str(input_path))
                inference_s += time.perf_counter() - inference_start
                page_count += 1
                regions += len(results)
                output_start = time.perf_counter()
                for result in results:
                    text = result['rec_text']
                    score = float(result['rec_score'])
                    output.write(f'{input_path}\t{text}\t{score}\n')
                    print(f'{input_path}\t{text}\t{score:.4f}')
                output_s += time.perf_counter() - output_start
    finally:
        recognizer.close()
    total_s = time.perf_counter() - total_start
    write_benchmark(ROOT,benchmark_report(page_count,regions,model_load_s,inference_s,output_s,total_s))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['setup','configure','train','eval','test','export','infer','page'])
    p.add_argument('--upstream',type=Path,default=DEFAULT_UPSTREAM)
    p.add_argument('--model-size',choices=MODEL_SIZES,default='medium',
                   help='Model variant to download during setup or configure')
    p.add_argument('--det-size',choices=DET_MODEL_SIZES,default='medium',
                   help='Detector variant for setup and detector-based inference')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--prepared',type=Path,default=ROOT/'prepared')
    p.add_argument('--output',type=Path,
                   help='Experiment directory (default: output/hindi_rec_<model-size>)')
    p.add_argument('--num-samples',type=int,
                   help='Use exactly N deterministic training rows for this run')
    p.add_argument('--num-eval-samples',type=int,
                   help='Use exactly N deterministic validation rows for this run')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--lr',type=float,default=0.0001)
    p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--device',choices=['gpu','cpu'],default='gpu')
    p.add_argument('--checkpoint',type=Path,help='Trained checkpoint prefix or .pdparams file')
    p.add_argument('--resume',type=Path,help='Training checkpoint prefix with .pdopt and .states siblings')
    p.add_argument('--image',type=Path)
    p.add_argument('--rec-model-dir',type=Path,
                   help='Exported custom recognizer directory for full-page OCR')
    p.add_argument('--cropped',action='store_true',
                   help='Input is already a crop; bypass the detector (infer only)')
    p.add_argument('--model',choices=['finetuned','official'],default='finetuned',
                   help='infer weights: fine-tuned Hindi checkpoint or official pretrained')
    a = p.parse_args()
    for key in ('upstream','data_dir','prepared'):
        setattr(a,key,getattr(a,key).resolve())
    use_page_output = a.action=='page' or (a.action=='infer' and a.rec_model_dir is not None)
    default_output = ROOT/'output/page_ocr' if use_page_output else ROOT/'output'/f'hindi_rec_{a.model_size}'
    a.output = (a.output or default_output).resolve()
    if min(a.batch_size,a.epochs,a.eval_every)<=0 or a.workers<0 or a.lr<=0:
        p.error('Invalid training arguments')
    for value in (a.num_samples,a.num_eval_samples):
        if value is not None and value <= 0:
            p.error('Sample counts must be positive')
    if a.model=='official' and a.action!='infer':
        p.error('--model official is only valid with infer')
    if a.model_size!='medium' and a.action not in ('setup','configure'):
        p.error('--model-size is only valid with setup and configure; use --output to select an experiment')
    if a.action!='train' and (a.num_samples is not None or a.num_eval_samples is not None):
        p.error('--num-samples and --num-eval-samples are only valid with train')
    if a.resume and (a.num_samples is not None or a.num_eval_samples is not None):
        p.error('Resume uses the saved effective configuration; do not change sample counts')
    if a.cropped and a.action!='infer':
        p.error('--cropped is only valid with infer')
    if a.rec_model_dir and a.action not in ('infer','page'):
        p.error('--rec-model-dir is only valid with infer and page')
    if a.det_size!='medium' and a.action not in ('setup','infer','page'):
        p.error('--det-size is only valid with setup, infer, and page')
    if a.cropped and a.det_size!='medium':
        p.error('--det-size is not used with --cropped')
    if a.action=='setup':
        setup(a.upstream,model_size=a.model_size,det_size=a.det_size); return
    if a.action=='page':
        if not a.rec_model_dir:
            p.error('--rec-model-dir is required with page')
        if not a.image:
            p.error('--image is required with page')
        run_page(a.upstream,a.rec_model_dir,a.image,a.output,a.device,a.det_size)
        return
    if a.action=='infer' and a.rec_model_dir:
        if not a.image:
            p.error('--image is required with infer')
        if a.cropped:
            run_cropped(a.upstream,a.rec_model_dir,a.image,a.device)
        else:
            run_page(a.upstream,a.rec_model_dir,a.image,a.output,a.device,a.det_size)
        return
    config_path = a.output/'train_config.yml'
    if a.action=='configure':
        setup(a.upstream,model_size=a.model_size)
        report = json.loads((a.prepared/'report.json').read_text())
        if any(report['out_of_vocabulary'].values()):
            raise ValueError('Resolve out-of-vocabulary held-out labels before training.')
        for split in ('train','val','test'):
            if not (a.prepared/f'{split}.txt').is_file():
                raise ValueError(f'Missing {split}.txt')
        initialized = a.output/'hindi_initialized.pdparams'
        c = config_for(a.upstream,a.data_dir,a.prepared,a.output,initialized,a,a.model_size)
        adapted = init_checkpoint(a.upstream,c,a.output,a.model_size)
        c['Global']['pretrained_model'] = str(adapted)
        config_path.write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True),encoding='utf-8')
        portable = checkpoint_config_for(c)
        (a.output/'checkpoint_config.yml').write_text(
            yaml.safe_dump(portable,sort_keys=False,allow_unicode=True),encoding='utf-8')
        source_dict = a.prepared/'hindi_dict.txt'
        bundled_dict = a.output/'hindi_dict.txt'
        if source_dict.resolve() != bundled_dict.resolve():
            shutil.copyfile(source_dict,bundled_dict)
        print(f'Wrote {config_path}'); return
    if a.action=='infer' and a.model=='official':
        # Stock zh/en weights on the pinned upstream config. The Hindi dictionary,
        # generated YAML and dictionary-hash guards do not apply to the official model.
        config_path = a.upstream/'configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml'
        if not config_path.is_file():
            p.error(f'Missing {config_path}; run setup first.')
        weights = WEIGHTS/PRETRAINED_FILENAME
        if not weights.is_file():
            p.error(f'Missing {weights}; run setup first.')
        if a.checkpoint:
            p.error('--checkpoint selects fine-tuned weights; use --model finetuned.')
        if not a.image:
            p.error('--image is required for infer')
        tool = 'infer_rec.py'
        overrides = [f'Global.use_gpu={a.device=="gpu"}', 'Global.distributed=false',
            'Global.checkpoints=null',
            f'Global.pretrained_model={str(weights).removesuffix(".pdparams")}',
            f'Global.infer_img={a.image.resolve()}', f'Global.save_res_path={ROOT/"predictions.txt"}']
    elif a.action=='infer' and a.checkpoint:
        prefix = str(a.checkpoint.resolve()).removesuffix('.pdparams')
        checkpoint = Path(prefix+'.pdparams')
        if not checkpoint.is_file():
            p.error(f'Missing checkpoint {checkpoint}')
        bundle = checkpoint.parent
        config_path = bundle/'checkpoint_config.yml'
        dictionary = bundle/'hindi_dict.txt'
        initialization = bundle/'initialization.json'
        for required in (config_path,dictionary,initialization):
            if not required.is_file():
                p.error(f'Missing portable checkpoint file {required}')
        if not a.image:
            p.error('--image is required for infer')
        init = json.loads(initialization.read_text(encoding='utf-8'))
        if file_sha256(dictionary) != init['dictionary_sha256']:
            raise ValueError('Bundled dictionary does not match initialization metadata.')
        tool = 'infer_rec.py'
        overrides = [f'Global.use_gpu={a.device=="gpu"}',
            f'Global.pretrained_model={prefix}', 'Global.checkpoints=null',
            f'Global.character_dict_path={dictionary}',
            f'Global.infer_img={a.image.resolve()}', f'Global.save_res_path={ROOT/"predictions.txt"}']
    else:
        if not config_path.exists():
            p.error(f'Missing {config_path}; run configure first.')
        if a.action=='train' and a.resume:
            config_path = a.output/'effective_train_config.yml'
            if not config_path.is_file():
                p.error(f'Missing {config_path}; cannot verify the original training configuration.')
        c = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        dictionary = Path(c['Global']['character_dict_path'])
        current_hash = file_sha256(dictionary)
        init = json.loads((a.output/'initialization.json').read_text(encoding='utf-8'))
        if current_hash != init['dictionary_sha256']:
            raise ValueError('Dictionary changed; use a separate experiment and reconfigure.')
        overrides = [f'Global.use_gpu={a.device=="gpu"}']
        if a.action=='train':
            tool='train.py'
            if a.resume:
                for mode in ('Train','Eval'):
                    for label in c[mode]['dataset']['label_file_list']:
                        label_path = Path(label).resolve()
                        if label_path.parent == (a.output/'subsets').resolve():
                            verify_subset(label_path)
                prefix=str(a.resume.resolve()).removesuffix('.pdparams')
                for ext in ('.pdparams','.pdopt','.states'):
                    if not Path(prefix+ext).exists():
                        raise ValueError(f'Missing resume file: {prefix+ext}')
                overrides += [f'Global.checkpoints={prefix}']
            else:
                if a.num_samples is not None:
                    source = Path(c['Train']['dataset']['label_file_list'][0])
                    subset = subset_for(source,a.output,'train',a.num_samples)
                    c['Train']['dataset']['label_file_list'] = [str(subset)]
                if a.num_eval_samples is not None:
                    source = Path(c['Eval']['dataset']['label_file_list'][0])
                    subset = subset_for(source,a.output,'val',a.num_eval_samples)
                    c['Eval']['dataset']['label_file_list'] = [str(subset)]
                config_path = a.output/'effective_train_config.yml'
                config_path.write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True),encoding='utf-8')
        else:
            checkpoint = a.checkpoint or Path(c['Global']['save_model_dir'])/'best_accuracy'
            prefix = str(checkpoint.resolve()).removesuffix('.pdparams')
            if not Path(prefix+'.pdparams').is_file():
                p.error(f'Missing checkpoint {prefix}.pdparams')
            overrides += [f'Global.pretrained_model={prefix}', 'Global.checkpoints=null']
            tool={'eval':'eval.py','test':'eval.py','export':'export_model.py','infer':'infer_rec.py'}[a.action]
            if a.action=='test':
                test_labels = Path(c['Eval']['dataset']['label_file_list'][0]).with_name('test.txt')
                overrides += [f'Eval.dataset.label_file_list=[{json.dumps(str(test_labels))}]']
            if a.action=='export':
                overrides += [f'Global.save_inference_dir={a.output/"inference"}']
            if a.action=='infer':
                if not a.image:
                    p.error('--image is required for infer')
                overrides += [f'Global.infer_img={a.image.resolve()}',f'Global.save_res_path={ROOT/"predictions.txt"}']
    env = os.environ.copy()
    env['PYTHONPATH'] = str(a.upstream)+os.pathsep+env.get('PYTHONPATH','')
    env['FLAGS_allocator_strategy']='auto_growth'
    subprocess.run([sys.executable,str(a.upstream/'tools'/tool),'-c',str(config_path),'-o',*overrides],cwd=a.upstream,env=env,check=True)

if __name__=='__main__':
    main()
