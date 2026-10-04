"""Write deterministic path manifests without extracting the image archive."""
from pathlib import Path
import argparse, hashlib, json, zipfile
import numpy as np

def prepare(archive_path, output):
    output.mkdir(parents=True, exist_ok=True)
    prefix = 'tiny-imagenet-200/'
    with zipfile.ZipFile(archive_path) as archive:
        classes = archive.read(prefix + 'wnids.txt').decode().split()
        assert len(classes) == len(set(classes)) == 200
        mapping = {c:i for i,c in enumerate(classes)}
        rng = np.random.default_rng(42)
        records = {'train':[], 'val':[], 'test':[]}
        names = archive.namelist()
        for c in classes:
            files = sorted(p for p in names if p.startswith(prefix+'train/'+c+'/images/') and p.endswith('.JPEG'))
            assert len(files) == 500, (c, len(files))
            valid = set(rng.permutation(500)[:50].tolist())
            for i,p in enumerate(files):
                records['val' if i in valid else 'train'].append((p.removeprefix(prefix),mapping[c]))
        annotations = archive.read(prefix+'val/val_annotations.txt').decode().splitlines()
        assert len(annotations) == 10000
        for line in annotations:
            name,c,*_ = line.split('\t')
            records['test'].append(('val/images/'+name,mapping[c]))
    sets = {k:{p for p,_ in rows} for k,rows in records.items()}
    assert not sets['train'] & sets['val']
    assert not (sets['train'] | sets['val']) & sets['test']
    manifest = {'seed':42,'classes':classes,'splits':{}}
    for split,rows in records.items():
        digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
        counts = np.bincount([label for _,label in rows],minlength=200).tolist()
        expected = {'train':450,'val':50,'test':50}[split]
        assert counts == [expected]*200
        (output/(split+'.jsonl')).write_text(''.join(json.dumps({'path':p,'label':y})+'\n' for p,y in rows))
        manifest['splits'][split] = {'count':len(rows),'sha256':digest,'per_class':counts}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print({k:v['count'] for k,v in manifest['splits'].items()})

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('splits'))
    args = ap.parse_args()
    prepare(args.archive,args.output)
