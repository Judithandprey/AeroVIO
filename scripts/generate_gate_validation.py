"""Generate the predeclared fresh V12 set using the unchanged capture protocol."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    cfg=json.loads((ROOT/'configs/long-validation-v12.json').read_text())
    prior=set()
    for n in ('v10','v11'):
        record=json.loads((ROOT/f'data/long-validation-{n}/data_manifest.json').read_text())
        prior.update(e['seed'] for e in record['attempts'])
    assert not prior.intersection(range(cfg['seed'],cfg['seed']+18))
    base=ROOT/'scripts/generate_long_validation.py';source=base.read_text()
    old="config_path=ROOT/'configs/long-validation-v10.json'";assert source.count(old)==1
    source=source.replace(old,"config_path=ROOT/'configs/long-validation-v12.json'")
    exec(compile(source,str(base),'exec'),{'__name__':'__main__','__file__':__file__})
    p=ROOT/cfg['logical_output']/'data_manifest.json';record=json.loads(p.read_text())
    record['reused_generator_sha256']=hashlib.sha256(base.read_bytes()).hexdigest()
    record['prior_long_seeds_disjoint']=True;p.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
