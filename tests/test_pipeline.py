import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from PIL import Image
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare import prepare, annotations, ctc_steps
from run import config_for, ROOT

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

if __name__=='__main__': unittest.main()
