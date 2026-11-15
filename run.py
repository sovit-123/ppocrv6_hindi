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
import urllib.request
import yaml

ROOT = Path(__file__).resolve().parent
PADDLE_REF = 'dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf'
DEFAULT_UPSTREAM = ROOT/'third_party/PaddleOCR'
WEIGHTS = ROOT/'weights'
MODEL_SIZES = ('tiny','small','medium')
PRETRAINED_FILENAME = 'PP-OCRv6_medium_rec_pretrained.pdparams'
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


def require_setup(upstream=DEFAULT_UPSTREAM, weights=WEIGHTS, model_size=None):
    """Validate setup prerequisites without downloading or modifying anything."""
    validate_upstream(upstream)
    for size in MODEL_SIZES if model_size is None else (model_size,):
        original = weights/pretrained_filename(size)
        if original.is_file() and original.stat().st_size > 0:
            return original
    raise ValueError(f'Setup is incomplete: no usable official weights in {weights}. '
                     'Run `python run.py setup --model-size {tiny,small,medium}` first.')


def setup(upstream, weights=WEIGHTS, model_size='medium'):
    if not upstream.exists():
        shell(['git','clone','https://github.com/PaddlePaddle/PaddleOCR.git',upstream])
        shell(['git','checkout',PADDLE_REF], upstream)
    validate_upstream(upstream)
    original = download_weights(weights,model_size)
    print(f'Pinned upstream and official {model_size} weights ready: {original}')
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['setup','configure','train','eval','test','export','infer'])
    p.add_argument('--upstream',type=Path,default=DEFAULT_UPSTREAM)
    p.add_argument('--model-size',choices=MODEL_SIZES,default='medium',
                   help='Model variant to download during setup or configure')
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
    p.add_argument('--model',choices=['finetuned','official'],default='finetuned',
                   help='infer weights: fine-tuned Hindi checkpoint or official pretrained')
    a = p.parse_args()
    for key in ('upstream','data_dir','prepared'):
        setattr(a,key,getattr(a,key).resolve())
    a.output = (a.output or ROOT/'output'/f'hindi_rec_{a.model_size}').resolve()
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
    if a.action=='setup':
        setup(a.upstream,model_size=a.model_size); return
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
