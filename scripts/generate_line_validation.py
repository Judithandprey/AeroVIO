"""Fresh room-disjoint long flights; unchanged capture and first-success policy."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    cfg=json.loads((ROOT/'configs/long-validation-v13.json').read_text())
    previous=set()
    for version in ('v10','v11','v12'):
        old=json.loads((ROOT/f'data/long-validation-{version}/data_manifest.json').read_text())
        previous.update(e['seed'] for e in old['attempts'])
    assert not previous.intersection(range(cfg['seed'],cfg['seed']+12))
    base=ROOT/'scripts/generate_long_validation.py';source=base.read_text()
    needle="config_path=ROOT/'configs/long-validation-v10.json'";assert source.count(needle)==1
    source=source.replace(needle,"config_path=ROOT/'configs/long-validation-v13.json'")
    exec(compile(source,str(base),'exec'),{'__name__':'__main__','__file__':__file__})
    p=ROOT/cfg['logical_output']/'data_manifest.json';record=json.loads(p.read_text())
    record['reused_generator_sha256']=hashlib.sha256(base.read_bytes()).hexdigest()
    record['prior_long_seeds_disjoint']=True
    record['development_scene_disjoint']=True
    record['generation_used_estimator_or_line_scores']=False
    p.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
