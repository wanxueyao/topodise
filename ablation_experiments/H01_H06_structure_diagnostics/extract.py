"""Extract structure evidence with explicit train/validation/test group manifests."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF,InterpolationMode
from ..G01_G05_encoder_comparison.extract import load_encoder,encode_pairs
from .descriptors import compute_mask_descriptors
from topodise.topo_targets import compute_topo_target
from topodise.training.runtime.protocol import file_hash,write


def validate_manifest(manifest,protocol,pretraining_inventory=None):
    rows=manifest['samples'];ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate structure sample IDs')
    splits={s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
    if any(not v for v in splits.values()) or sum(map(len,splits.values()))!=len(rows): raise ValueError('Explicit train/validation/test splits required')
    owners={};hashes={}
    for row in rows:
        group=row.get('group_id')
        if not group: raise ValueError('Explicit patient/eye/acquisition source group required')
        if owners.setdefault(group,row['split'])!=row['split']: raise ValueError('Source group crosses structure splits')
        for key in ('image','mask'):
            sha=file_hash(row[key]);hashes[row[key]]=sha
            if key=='image' and owners.setdefault('hash:'+sha,row['split'])!=row['split']: raise ValueError('Duplicate image crosses splits')
    if protocol=='STR-B':
        if not pretraining_inventory: raise ValueError('STR-B requires complete encoder pretraining inventory; STR-A cannot be renamed')
        inventory=json.loads(Path(pretraining_inventory).read_text())
        if not inventory.get('complete') or not inventory.get('checkpoint_sha256'): raise ValueError('Inventory must attest completeness and bind encoder checkpoint hash')
        seen=set(inventory['image_sha256']);groups=set(inventory['group_ids'])
        if any(hashes[r['image']] in seen or r['group_id'] in groups for r in splits['test']): raise ValueError('STR-B evaluation overlaps encoder pretraining')
    return rows,hashes


@torch.no_grad()
def extract(manifest,backend,checkpoint,output,protocol='STR-A',inventory=None,device='cpu'):
    rows,hashes=validate_manifest(manifest,protocol,inventory)
    if protocol=='STR-B' and checkpoint and json.loads(Path(inventory).read_text())['checkpoint_sha256']!=file_hash(checkpoint): raise ValueError('Pretraining inventory does not describe this encoder')
    encoder,transform=load_encoder(backend,checkpoint,device)
    features=[];targets=[];descriptors=[];images=[];masks=[]
    for row in rows:
        image=Image.open(row['image']).convert('L');mask=Image.open(row['mask']).convert('L')
        if image.size!=mask.size: raise ValueError('Image/mask geometry mismatch')
        m=TF.center_crop(TF.resize(mask,224,InterpolationMode.NEAREST),[224,224]);m=(TF.to_tensor(m)>0.5).float()
        x=transform(image).unsqueeze(0).to(device)
        features.append(encode_pairs(encoder,backend,x,x).cpu().squeeze(0).numpy())
        targets.append(compute_topo_target(m.unsqueeze(0),mode='gt',mask=m.unsqueeze(0))[0].numpy())
        descriptors.append(compute_mask_descriptors(m));masks.append(m.squeeze(0).numpy())
        images.append(np.asarray(TF.center_crop(TF.resize(image,224),[224,224])))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,features=features,targets=targets,descriptors=descriptors,images=images,masks=masks,
        sample_ids=[r['id'] for r in rows],groups=[r['group_id'] for r in rows],splits=[r['split'] for r in rows],image_hashes=[hashes[r['image']] for r in rows])
    write(output.with_suffix('.json'),dict(protocol_id=protocol,encoder=backend,checkpoint_sha256=file_hash(checkpoint) if checkpoint else None,
        input_sha256=hashes,manifest=manifest,descriptor_algorithm='Zhang-Suen max256; 8-neighbor branch pixels >=3 / skeleton length; endpoint pixels ==1; empty ->0',image_size=224))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--encoder',default='topodise');p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--protocol',choices=['STR-A','STR-B'],default='STR-A');p.add_argument('--pretraining-inventory',type=Path);p.add_argument('--device',default='cpu')
    p.add_argument('--dataset-root', type=Path, help='Root for relative image and mask paths in the manifest')
    a=p.parse_args(argv)
    manifest = json.loads(a.manifest.read_text())
    if a.dataset_root is not None:
        for row in manifest['samples']:
            for key in ('image', 'mask'):
                row[key] = str(a.dataset_root / row[key])
    extract(manifest,a.encoder,a.checkpoint,a.output,a.protocol,a.pretraining_inventory,a.device)

if __name__=='__main__': main()
