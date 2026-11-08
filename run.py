"""Configure and run upstream PP-OCRv6 recognition tools."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import yaml

ROOT = Path(__file__).resolve().parent
PADDLE_REF = 'dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf'
URL = 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams'
RESET_PREFIXES = ('head.ctc_head.fc.', 'head.gtc_head.embedding.', 'head.gtc_head.tgt_word_prj.')


def shell(command, cwd=None):
    print('+', ' '.join(map(str,command)), flush=True)
    subprocess.run(list(map(str,command)), cwd=cwd, check=True)


def setup(upstream):
    if not upstream.exists():
        shell(['git','clone','https://github.com/PaddlePaddle/PaddleOCR.git',upstream])
        shell(['git','checkout',PADDLE_REF], upstream)
    actual = subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()
    if actual != PADDLE_REF:
        raise ValueError(f'Expected PaddleOCR {PADDLE_REF}; found {actual}. Use a separate checkout.')
    if subprocess.check_output(['git','status','--porcelain'], cwd=upstream,text=True).strip():
        raise ValueError('Upstream checkout has local changes; use a clean checkout.')
    print('Pinned upstream ready. Install its training dependencies as described in README.')


def config_for(upstream, data, prepared, output, checkpoint, a):
    c = yaml.safe_load((upstream/'configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml').read_text())
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


def init_checkpoint(upstream, config, weights):
    import paddle
    sys.path.insert(0,str(upstream))
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process
    weights.mkdir(parents=True,exist_ok=True)
    original = weights/'PP-OCRv6_medium_rec_pretrained.pdparams'
    if not original.exists():
        temp = original.with_suffix('.part')
        print('Downloading official training weights...',flush=True)
        urllib.request.urlretrieve(URL,temp)
        temp.replace(original)
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
    adapted = weights/'hindi_initialized.pdparams'
    paddle.save(target,str(adapted))
    report = {'source_url':URL,'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
              'paddle_version':paddle.__version__, 'paddleocr_commit':PADDLE_REF,
              'loaded':loaded,'reset_vocabulary_parameters':reset,'ctc_classes':n,
              'dictionary_sha256':hashlib.sha256(Path(config['Global']['character_dict_path']).read_bytes()).hexdigest()}
    (weights/'initialization.json').write_text(json.dumps(report,indent=2))
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['setup','configure','train','eval','test','export','infer'])
    p.add_argument('--upstream',type=Path,default=ROOT/'third_party/PaddleOCR')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--prepared',type=Path,default=ROOT/'prepared')
    p.add_argument('--output',type=Path,default=ROOT/'output/hindi_rec')
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
    for key in ('upstream','data_dir','prepared','output'):
        setattr(a,key,getattr(a,key).resolve())
    if min(a.batch_size,a.epochs,a.eval_every)<=0 or a.workers<0 or a.lr<=0:
        p.error('Invalid training arguments')
    if a.model=='official' and a.action!='infer':
        p.error('--model official is only valid with infer')
    if a.action=='setup':
        setup(a.upstream); return
    config_path = a.prepared/'hindi_rec.yml'
    if a.action=='configure':
        setup(a.upstream)
        report = json.loads((a.prepared/'report.json').read_text())
        if any(report['out_of_vocabulary'].values()):
            raise ValueError('Resolve out-of-vocabulary held-out labels before training.')
        for split in ('train','val','test'):
            if not (a.prepared/f'{split}.txt').is_file():
                raise ValueError(f'Missing {split}.txt')
        c = config_for(a.upstream,a.data_dir,a.prepared,a.output,ROOT/'weights/hindi_initialized.pdparams',a)
        adapted = init_checkpoint(a.upstream,c,ROOT/'weights')
        c['Global']['pretrained_model'] = str(adapted)
        config_path.write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True))
        print(f'Wrote {config_path}'); return
    if a.action=='infer' and a.model=='official':
        # Stock zh/en weights on the pinned upstream config. The Hindi dictionary,
        # generated YAML and dictionary-hash guards do not apply to the official model.
        config_path = a.upstream/'configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml'
        if not config_path.is_file():
            p.error(f'Missing {config_path}; run setup first.')
        weights = ROOT/'weights/PP-OCRv6_medium_rec_pretrained.pdparams'
        if not weights.is_file():
            p.error(f'Missing {weights}; run configure once to download official weights.')
        if a.checkpoint:
            p.error('--checkpoint selects fine-tuned weights; use --model finetuned.')
        if not a.image:
            p.error('--image is required for infer')
        tool = 'infer_rec.py'
        overrides = [f'Global.use_gpu={a.device=="gpu"}', 'Global.distributed=false',
            'Global.checkpoints=null',
            f'Global.pretrained_model={str(weights).removesuffix(".pdparams")}',
            f'Global.infer_img={a.image.resolve()}', f'Global.save_res_path={ROOT/"predictions.txt"}']
    else:
        if not config_path.exists():
            p.error('Run configure first.')
        c = yaml.safe_load(config_path.read_text())
        current_hash = hashlib.sha256((a.prepared/'hindi_dict.txt').read_bytes()).hexdigest()
        init = json.loads((ROOT/'weights/initialization.json').read_text())
        if current_hash != init['dictionary_sha256']:
            raise ValueError('Dictionary changed; use a separate project/run and reconfigure.')
        overrides = [f'Global.use_gpu={a.device=="gpu"}']
        if a.action=='train':
            tool='train.py'
            if a.resume:
                prefix=str(a.resume.resolve()).removesuffix('.pdparams')
                for ext in ('.pdparams','.pdopt','.states'):
                    if not Path(prefix+ext).exists():
                        raise ValueError(f'Missing resume file: {prefix+ext}')
                overrides += [f'Global.checkpoints={prefix}']
        else:
            checkpoint = a.checkpoint or Path(c['Global']['save_model_dir'])/'best_accuracy'
            prefix = str(checkpoint.resolve()).removesuffix('.pdparams')
            if not Path(prefix+'.pdparams').is_file():
                p.error(f'Missing checkpoint {prefix}.pdparams')
            overrides += [f'Global.pretrained_model={prefix}', 'Global.checkpoints=null']
            tool={'eval':'eval.py','test':'eval.py','export':'export_model.py','infer':'infer_rec.py'}[a.action]
            if a.action=='test':
                overrides += [f'Eval.dataset.label_file_list=[{json.dumps(str(a.prepared/"test.txt"))}]']
            if a.action=='export':
                overrides += [f'Global.save_inference_dir={ROOT/"inference/hindi_rec"}']
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
