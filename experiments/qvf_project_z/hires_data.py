"""Frozen, genuinely higher-resolution Stanford Dogs reconstruction data.

The training environment is unchanged. An existing base-environment PyArrow is
used only to extract JPEG bytes from a pinned public Parquet mirror.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version
import requests

from experiments.qvf_project_a.train import sha256, write_json

REPOSITORY = 'maurice-fp/stanford-dogs'
REVISION = 'ee102b9a35f00829e65bdad7cda8adcfa185b400'
FILES = {
    'train.parquet': ('07a5729ae8cf8c21c1f18d8f93b05fc0c7d7e010300082ad028b1ea8b42ed3c1', 455081658),
    'test.parquet': ('a3f55298e59d6e87012e4183390373cd39266494f960ac24f9165344343298ee', 321347256),
}
REMOTE_FILES = {split+'.parquet':f'data/{split}-00000-of-00001.parquet' for split in ('train','test')}
BASE_PYTHON = '/home/agroz/anaconda3/bin/python'


def fetch(root):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    def one(item):
        name, (digest, size) = item
        dest = root / name
        if dest.exists():
            assert dest.stat().st_size == size and sha256(dest) == digest
            return
        url = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{REMOTE_FILES[name]}'
        temporary = dest.with_suffix('.download')
        with requests.get(url, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            with temporary.open('wb') as handle:
                for block in response.iter_content(1024*1024):
                    handle.write(block)
        assert temporary.stat().st_size == size and sha256(temporary) == digest, name
        temporary.rename(dest)
        print(f'Downloaded and SHA256-verified {name}', flush=True)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        list(pool.map(one, FILES.items()))
    for name, remote in [('README.md','README.md'),('train_metadata.csv','train/metadata.csv'),('test_metadata.csv','test/metadata.csv')]:
        dest = root / name
        if not dest.exists():
            response = requests.get(f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{remote}', timeout=30)
            response.raise_for_status(); dest.write_bytes(response.content)
    return root


EXTRACT_SCRIPT = r'''
import csv,json,sys
from pathlib import Path
import pyarrow,pyarrow.parquet as pq
root=Path(sys.argv[1]); rows=[]
for split,expected in [('train',12000),('test',8580)]:
    folder=root/'raw'/split;folder.mkdir(parents=True,exist_ok=True)
    metadata=list(csv.DictReader((root/(split+'_metadata.csv')).open()))
    names={Path(r['file_name']).name:r for r in metadata}
    assert len(names)==expected
    count=0
    for batch in pq.ParquetFile(root/(split+'.parquet')).iter_batches(batch_size=128):
        for row in batch.to_pylist():
            original_path=row['image']['path']
            name=Path(original_path).name if original_path else Path(metadata[count]['file_name']).name
            meta=names[name]
            assert int(meta['label'])==int(row['label'])
            if name in ('','..','.') or not name.lower().endswith(('.jpg','.jpeg','.png')):
                raise ValueError(name)
            path=folder/name
            data=row['image']['bytes']
            if not isinstance(data,bytes):raise ValueError('Missing embedded image bytes')
            if path.exists():
                if path.read_bytes()!=data:raise ValueError('Conflicting image names')
            else:path.write_bytes(data)
            rows.append({'official_split':split,'image_id':meta['file_name'],
                'label':int(row['label']),'class_name':meta['file_name'].split('/')[0],
                'parquet_row':count,'parquet_image_path':original_path,
                'path':str(path.relative_to(root))})
            count+=1
    assert count==expected,(split,count)
    print('Extracted',split,count,flush=True)
(root/'raw_manifest.json').write_text(json.dumps({'python':sys.version,
    'pyarrow_version':pyarrow.__version__,'records':rows},indent=2)+'\n')
'''


def prepare(root, size=256, seed=20261110):
    root = fetch(root)
    raw_manifest = root / 'raw_manifest.json'
    if not raw_manifest.exists():
        subprocess.run([BASE_PYTHON, '-c', EXTRACT_SCRIPT, str(root)], check=True)
    output = root / f'prepared_{size}'
    if output.exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        for name, digest in manifest['array_sha256'].items():
            assert sha256(output / name) == digest
        return output
    output.mkdir()
    raw = json.loads(raw_manifest.read_text())['records']
    eligible = {'train': [], 'test': []}; excluded = []
    for row in raw:
        try:
            with Image.open(root / row['path']) as image:
                width, height = image.size
            if min(width, height) < size:
                excluded.append({'id': row['image_id'], 'reason': 'native_dimensions_below_target', 'size': [width,height]})
                continue
            eligible[row['official_split']].append({**row, 'native_size': [width,height]})
        except Exception as error:
            excluded.append({'id': row['image_id'], 'reason': str(error)})
    rng = np.random.default_rng(seed)
    for rows in eligible.values():
        rng.shuffle(rows)
    train_pool = iter(eligible['train']); test_pool = iter(eligible['test'])
    seen_ids, seen_rgb = set(), set(); records = {}; arrays = {}; duplicate_exclusions=[]
    for split, count, pool in [('train',4744,train_pool),('development',256,train_pool),('test',64,test_pool)]:
        rows, images = [], []
        while len(rows) < count:
            row = next(pool)
            if row['image_id'] in seen_ids:
                duplicate_exclusions.append({'id':row['image_id'],'reason':'duplicate_id'});continue
            with Image.open(root / row['path']) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                width,height=image.size; side=min(width,height)
                assert side>=size
                left,top=(width-side)//2,(height-side)//2
                image=image.crop((left,top,left+side,top+side)).resize((size,size),Image.Resampling.LANCZOS)
                pixels=np.array(image,dtype=np.uint8)
            digest=hashlib.sha256(pixels.tobytes()).hexdigest()
            if digest in seen_rgb:
                duplicate_exclusions.append({'id':row['image_id'],'reason':'duplicate_processed_rgb'});continue
            seen_ids.add(row['image_id']);seen_rgb.add(digest)
            images.append(pixels)
            rows.append({**row,'jpeg_sha256':sha256(root/row['path']),'processed_rgb_sha256':digest,
                         'crop_box':[left,top,left+side,top+side]})
        name=split+'.npy';np.save(output/name,np.stack(images));arrays[name]=sha256(output/name)
        records[split]=rows
        print(f'Prepared {split}: {count} genuine {size}x{size} targets',flush=True)
    manifest={'dataset':'Stanford Dogs','official_source':'http://vision.stanford.edu/aditya86/ImageNetDogs/',
        'catalogue':'https://www.tensorflow.org/datasets/catalog/stanford_dogs',
        'mirror':REPOSITORY,'revision':REVISION,'download_sha256':{k:v[0] for k,v in FILES.items()},
        'raw_manifest_sha256':sha256(raw_manifest),'image_size':size,'split_seed':seed,
        'preprocessing':'EXIF orientation, RGB conversion, integer centred square crop of the shorter side, Pillow LANCZOS downsample. Native dimensions must both be >= target; no upsampling.',
        'pillow_version':pillow_version,'numpy_version':np.__version__,
        'selection':'Seeded random subsets. Training/development come only from the mirror-labelled official train split; final test comes only from its official test split. Exact image-ID and processed-RGB duplicates excluded across selected sets. Near duplicates are not independently ruled out.',
        'source_limit':'Original Stanford server was unreachable; mirror train/test labels and counts are retained and recorded, not independently matched against the original MATLAB split lists.',
        'counts':{k:len(v) for k,v in records.items()},'eligible_counts':{k:len(v) for k,v in eligible.items()},
        'excluded_native_or_decode':excluded,'duplicate_exclusions':duplicate_exclusions,
        'records':records,'array_sha256':arrays}
    write_json(output/'manifest.json',manifest)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--size',type=int,default=256)
    args=parser.parse_args();print(prepare(args.root,args.size))
