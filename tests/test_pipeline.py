import importlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
from PIL import Image
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare as prepare_module
import run as run_module
from prepare import prepare, annotations, ctc_steps
from run import (checkpoint_config_for, config_for, download_weights, file_sha256, require_setup, subset_for,
                 ROOT, MODEL_SIZES, RESET_PREFIXES, pretrained_filename)

class PipelineTest(unittest.TestCase):
    def test_single_shard_schema_filter_split_and_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data'; data.mkdir(); tar=data/'batch.tar.gz'
            records=[]
            with tarfile.open(tar,'w:gz') as f:
                for i in range(30):
                    name=f'hindi_local_{i}.png'
                    im=Image.new('RGB',(80,20),(i,100,200))
                    b=io.BytesIO(); im.save(b,format='PNG'); raw=b.getvalue()
                    member=tarfile.TarInfo('nested/'+name); member.size=len(raw)
                    f.addfile(member,io.BytesIO(raw))
                    records.append({'images':['devanagari_ocr/'+name], 'messages':[{'role':'assistant','content':'हिन्दी पाठ'}]})
                # Identical image under a second basename must not cross splits.
                name='iiit_hindi_duplicate.png'
                member=tarfile.TarInfo(name); member.size=len(raw); f.addfile(member,io.BytesIO(raw))
                records.append({'images':[name],'messages':[{'role':'assistant','content':'हिन्दी पाठ'}]})
            records[0]['messages'][0]['content']='क'*41
            records[1]['messages'][0]['content']='क\nख'
            (data/'devanagari_ocr.json').write_text(json.dumps(records,ensure_ascii=False))
            prepare(data,tar,root/'out')
            report=json.loads((root/'out/report.json').read_text())
            self.assertEqual(sum(report['counts'].values()),28)
            self.assertEqual(report['stats']['duplicate_image'],1)
            self.assertEqual(report['stats']['too_long_for_ctc'],1)
            rows=[set((root/'out'/f'{s}.txt').read_text().splitlines()) for s in ('train','val','test')]
            self.assertFalse(rows[0]&rows[1] or rows[1]&rows[2] or rows[0]&rows[2])
            prepare(data,tar,root/'out2')
            self.assertEqual((root/'out/train.txt').read_bytes(),(root/'out2/train.txt').read_bytes())
            self.assertEqual(ctc_steps('क्क'),3)
            self.assertEqual(ctc_steps('कक'),3)
            with self.assertRaises(ValueError): prepare(data,tar,root/'out')

    def test_stream_array_and_jsonl(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.json'
            for text in ('[{"value":"{hi}"},{"value":"नमस्ते"}]', '{"value":"{hi}"}\n{"value":"नमस्ते"}\n'):
                p.write_text(text)
                self.assertEqual(len(list(annotations(p))),2)

    def test_prepare_cli_requires_setup(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            argv=['prepare.py','--offline','--data-dir',str(root/'data'),'--output',str(root/'out')]
            with mock.patch.object(sys,'argv',argv), mock.patch('run.require_setup',side_effect=ValueError('setup required')) as check:
                with self.assertRaises(SystemExit) as stopped:
                    prepare_module.main()
                self.assertEqual(stopped.exception.code,2)
                check.assert_called_once_with()
            self.assertFalse((root/'out').exists())

    def test_setup_cli_accepts_model_size(self):
        with tempfile.TemporaryDirectory() as t:
            upstream=Path(t)/'upstream'
            argv=['run.py','setup','--upstream',str(upstream),'--model-size','tiny']
            with mock.patch.object(sys,'argv',argv), mock.patch('run.setup') as setup:
                run_module.main()
                setup.assert_called_once_with(upstream.resolve(),model_size='tiny')

    def test_weight_download_is_cached_for_every_model_size(self):
        with tempfile.TemporaryDirectory() as t:
            for size in MODEL_SIZES:
                with self.subTest(size=size):
                    weights=Path(t)/size
                    with mock.patch('run.urllib.request.urlretrieve',
                                    side_effect=lambda _url,path: Path(path).write_bytes(b'weights')) as fetch:
                        original=download_weights(weights,size)
                        self.assertEqual(original.name,pretrained_filename(size))
                        self.assertEqual(original.read_bytes(),b'weights')
                        self.assertFalse(original.with_suffix('.part').exists())
                        self.assertEqual(download_weights(weights,size),original)
                        self.assertEqual(fetch.call_count,1)
                        self.assertTrue(fetch.call_args.args[0].endswith(original.name))

    def test_require_setup_checks_upstream_and_weights(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); upstream=root/'upstream'; weights=root/'weights'
            with mock.patch('run.validate_upstream') as validate:
                with self.assertRaisesRegex(ValueError,'run.py setup'):
                    require_setup(upstream,weights)
                weights.mkdir()
                original=weights/pretrained_filename('tiny')
                original.write_bytes(b'')
                with self.assertRaisesRegex(ValueError,'run.py setup'):
                    require_setup(upstream,weights)
                original.write_bytes(b'weights')
                self.assertEqual(require_setup(upstream,weights),original)
                self.assertEqual(validate.call_count,3)

    def test_config_changes_both_decoders_and_uses_all_training_rows(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); (p/'configs/rec/PP-OCRv6').mkdir(parents=True)
            (p/'configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml').write_bytes((ROOT/'vendor_config/PP-OCRv6_medium_rec.yml').read_bytes())
            a=SimpleNamespace(epochs=2,device='cpu',eval_every=1,lr=.0001,batch_size=2,workers=0)
            c=config_for(p,p,p,p,p/'weights',a)
            self.assertEqual(c['Train']['dataset']['ratio_list'],[1.])
            self.assertEqual(c['Train']['dataset']['name'],'SimpleDataSet')
            self.assertEqual(c['Architecture']['Head']['head_list'][1]['NRTRHead']['max_text_length'],40)
            self.assertFalse(c['Metric']['ignore_space'])
            yaml.safe_load(yaml.safe_dump(c))

    def test_tiny_ctc_vocabulary_projection_is_reset(self):
        self.assertTrue('head.ctc_head.fc2.weight'.startswith(RESET_PREFIXES))
        self.assertFalse('head.ctc_head.fc1.weight'.startswith(RESET_PREFIXES))

    def test_config_selects_every_model_size(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); config_dir=root/'configs/rec/PP-OCRv6'; config_dir.mkdir(parents=True)
            base=yaml.safe_load((ROOT/'vendor_config/PP-OCRv6_medium_rec.yml').read_text())
            for size in MODEL_SIZES:
                config=dict(base)
                config['Global']=dict(base['Global'],model_name=f'marker_{size}')
                config['Architecture']=dict(base['Architecture'])
                config['Architecture']['Backbone']=dict(base['Architecture']['Backbone'],model_size=size)
                (config_dir/f'PP-OCRv6_{size}_rec.yml').write_text(yaml.safe_dump(config,sort_keys=False))
            a=SimpleNamespace(epochs=2,device='cpu',eval_every=1,lr=.0001,batch_size=2,workers=0)
            for size in MODEL_SIZES:
                with self.subTest(size=size):
                    c=config_for(root,root,root,root,root/'initialized',a,size)
                    self.assertEqual(c['Global']['model_name'],f'marker_{size}')
                    self.assertEqual(c['Architecture']['Backbone']['model_size'],size)

    def test_configure_uses_selected_size_and_experiment(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); upstream=root/'upstream'; prepared=root/'prepared'; output=root/'experiment'
            config_dir=upstream/'configs/rec/PP-OCRv6'; config_dir.mkdir(parents=True)
            config=yaml.safe_load((ROOT/'vendor_config/PP-OCRv6_medium_rec.yml').read_text())
            config['Global']['model_name']='PP-OCRv6_tiny_rec'
            config['Architecture']['Backbone']['model_size']='tiny'
            (config_dir/'PP-OCRv6_tiny_rec.yml').write_text(yaml.safe_dump(config,sort_keys=False))
            prepared.mkdir()
            (prepared/'hindi_dict.txt').write_text('क\n',encoding='utf-8')
            (prepared/'report.json').write_text(json.dumps({'out_of_vocabulary':{}}))
            for split in ('train','val','test'):
                (prepared/f'{split}.txt').write_text('')
            argv=['run.py','configure','--upstream',str(upstream),'--prepared',str(prepared),
                  '--data-dir',str(root/'data'),'--output',str(output),'--model-size','tiny']
            def fake_init(_upstream,_config,experiment,model_size):
                self.assertEqual(model_size,'tiny')
                experiment.mkdir(parents=True)
                return experiment/'hindi_initialized.pdparams'
            with mock.patch.object(sys,'argv',argv), mock.patch('run.setup') as setup, \
                    mock.patch('run.init_checkpoint',side_effect=fake_init) as initialize:
                run_module.main()
                setup.assert_called_once_with(upstream.resolve(),model_size='tiny')
                initialize.assert_called_once()
            saved=yaml.safe_load((output/'train_config.yml').read_text())
            self.assertEqual(saved['Global']['model_name'],'PP-OCRv6_tiny_rec')
            self.assertEqual(saved['Global']['save_model_dir'],str(output.resolve()))
            self.assertEqual(saved['Global']['pretrained_model'],str(output.resolve()/'hindi_initialized.pdparams'))
            portable=yaml.safe_load((output/'checkpoint_config.yml').read_text())
            self.assertEqual(portable['Global']['character_dict_path'],'hindi_dict.txt')
            self.assertEqual(portable['Global']['save_model_dir'],'.')
            self.assertEqual(portable['Eval']['dataset']['label_file_list'],[])
            self.assertEqual((output/'hindi_dict.txt').read_text(encoding='utf-8'),'क\n')
            self.assertNotIn(str(prepared), (output/'checkpoint_config.yml').read_text())

    def test_subset_is_exact_deterministic_and_hashed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); source=root/'train.txt'; experiment=root/'experiment'
            rows=[f'devanagari_ocr/hindi_local_{i}.png\tहिन्दी {i}\n' for i in range(6)]
            source.write_text(''.join(rows),encoding='utf-8')
            original=source.read_bytes()
            subset=subset_for(source,experiment,'train',3)
            self.assertEqual(subset.read_text(encoding='utf-8'),''.join(rows[:3]))
            metadata=json.loads(subset.with_suffix('.json').read_text())
            self.assertEqual(metadata['source_sha256'],file_sha256(source))
            self.assertEqual(metadata['subset_sha256'],file_sha256(subset))
            self.assertEqual(metadata['requested_count'],3)
            self.assertEqual(metadata['actual_count'],3)
            self.assertEqual(metadata['selection_rule'],'first_n_of_prepared_shuffle')
            self.assertEqual(subset_for(source,experiment,'train',3),subset)
            with self.assertRaisesRegex(ValueError,'only 6'):
                subset_for(source,experiment,'train',7)
            self.assertEqual(source.read_bytes(),original)

    def test_train_uses_subsets_and_resume_reuses_effective_config(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); upstream=root/'upstream'; prepared=root/'prepared'; output=root/'experiment'
            config_dir=upstream/'configs/rec/PP-OCRv6'; config_dir.mkdir(parents=True)
            (config_dir/'PP-OCRv6_medium_rec.yml').write_bytes((ROOT/'vendor_config/PP-OCRv6_medium_rec.yml').read_bytes())
            prepared.mkdir(); output.mkdir()
            (prepared/'hindi_dict.txt').write_text('क\n',encoding='utf-8')
            for split,count in (('train',6),('val',4),('test',4)):
                rows=''.join(f'devanagari_ocr/hindi_local_{i}.png\tहिन्दी {i}\n' for i in range(count))
                (prepared/f'{split}.txt').write_text(rows,encoding='utf-8')
            a=SimpleNamespace(epochs=1,device='cpu',eval_every=1,lr=.0001,batch_size=2,workers=0)
            c=config_for(upstream,root/'data',prepared,output,output/'hindi_initialized',a)
            (output/'train_config.yml').write_text(yaml.safe_dump(c,sort_keys=False))
            (output/'initialization.json').write_text(json.dumps({'dictionary_sha256':file_sha256(prepared/'hindi_dict.txt')}))
            argv=['run.py','train','--upstream',str(upstream),'--output',str(output),
                  '--num-samples','3','--num-eval-samples','2']
            with mock.patch.object(sys,'argv',argv), mock.patch('run.subprocess.run') as execute:
                run_module.main()
                command=execute.call_args.args[0]
            effective_path=output/'effective_train_config.yml'
            self.assertIn(str(effective_path),command)
            effective=yaml.safe_load(effective_path.read_text())
            self.assertEqual(effective['Train']['dataset']['label_file_list'],
                             [str(output/'subsets/train_3.txt')])
            self.assertEqual(effective['Eval']['dataset']['label_file_list'],
                             [str(output/'subsets/val_2.txt')])
            self.assertEqual(len((output/'subsets/train_3.txt').read_text().splitlines()),3)
            self.assertEqual(len((output/'subsets/val_2.txt').read_text().splitlines()),2)
            for ext in ('.pdparams','.pdopt','.states'):
                (output/f'latest{ext}').write_bytes(b'checkpoint')
            argv=['run.py','train','--upstream',str(upstream),'--output',str(output),
                  '--resume',str(output/'latest')]
            with mock.patch.object(sys,'argv',argv), mock.patch('run.subprocess.run') as execute:
                run_module.main()
                command=execute.call_args.args[0]
            self.assertIn(str(effective_path),command)
            self.assertIn(f'Global.checkpoints={output}/latest',command)
            (output/'subsets/train_3.txt').write_text('corrupt\n',encoding='utf-8')
            with mock.patch.object(sys,'argv',argv), mock.patch('run.subprocess.run') as execute:
                with self.assertRaisesRegex(ValueError,'Subset hash mismatch'):
                    run_module.main()
                execute.assert_not_called()

    def test_portable_checkpoint_inference_uses_bundle_only(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); bundle=root/'shared'; bundle.mkdir(); upstream=root/'upstream'
            checkpoint=bundle/'best_accuracy.pdparams'; checkpoint.write_bytes(b'checkpoint')
            dictionary=bundle/'hindi_dict.txt'; dictionary.write_text('क\n',encoding='utf-8')
            (bundle/'initialization.json').write_text(json.dumps({'dictionary_sha256':file_sha256(dictionary)}))
            base=yaml.safe_load((ROOT/'vendor_config/PP-OCRv6_medium_rec.yml').read_text())
            (bundle/'checkpoint_config.yml').write_text(yaml.safe_dump(checkpoint_config_for(base),sort_keys=False))
            image=root/'crop.png'; image.write_bytes(b'image')
            argv=['run.py','infer','--upstream',str(upstream),'--checkpoint',str(checkpoint),
                  '--image',str(image),'--device','cpu']
            with mock.patch.object(sys,'argv',argv), mock.patch('run.subprocess.run') as execute:
                run_module.main()
                command=execute.call_args.args[0]
            self.assertIn(str(bundle/'checkpoint_config.yml'),command)
            self.assertIn(f'Global.pretrained_model={checkpoint.with_suffix("")}',command)
            self.assertIn(f'Global.character_dict_path={dictionary}',command)
            self.assertIn(f'Global.infer_img={image}',command)

    def test_gradio_checkpoint_route_uses_portable_bundle(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.dict(sys.modules,{'gradio':mock.MagicMock()}):
            app=importlib.import_module('gradio_app')
            root=Path(t); bundle=root/'shared'; bundle.mkdir(); upstream=root/'upstream'; upstream.mkdir()
            checkpoint=bundle/'best_accuracy.pdparams'; checkpoint.write_bytes(b'checkpoint')
            with mock.patch.object(sys,'argv',['gradio_app.py','--checkpoint',str(checkpoint)]):
                args=app.parse_args()
            app.CHECKPOINT=args.checkpoint.resolve(); app.PADDLE_ROOT=upstream
            self.assertIn('checkpoint_config.yml',app.checkpoint_bundle_problem())
            for name in ('checkpoint_config.yml','hindi_dict.txt','initialization.json'):
                (bundle/name).write_text('{}' if name.endswith('.json') else '',encoding='utf-8')
            self.assertIsNone(app.checkpoint_bundle_problem())
            source=root/'source.png'; staged=root/'staged.png'
            with mock.patch.object(app,'infer_subprocess',return_value=(['source.png','text','1.0','ok'],['log'])) as infer:
                app.infer_checkpoint(source,staged,'cpu')
                command=infer.call_args.args[1]
            self.assertEqual(command[:3],[sys.executable,str(app.RUN_PY),'infer'])
            self.assertIn('--checkpoint',command)
            self.assertIn(str(checkpoint),command)
            self.assertIn(str(staged),command)

if __name__=='__main__': unittest.main()
