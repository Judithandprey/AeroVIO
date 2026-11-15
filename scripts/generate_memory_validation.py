"""Reuse the unchanged V10 capture protocol with predeclared fresh V11 seeds."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    cfg=json.loads((ROOT/'configs/long-validation-v11.json').read_text())
    old=json.loads((ROOT/'data/long-validation-v10/data_manifest.json').read_text())
    seeds={e['seed'] for e in old['attempts']}
    assert not seeds.intersection(range(cfg['seed'],cfg['seed']+18))
    base=ROOT/'scripts/generate_long_validation.py';source=base.read_text()
    original="config_path=ROOT/'configs/long-validation-v10.json'"
    assert source.count(original)==1
    source=source.replace(original,"config_path=ROOT/'configs/long-validation-v11.json'")
    exec(compile(source,str(base),'exec'),{'__name__':'__main__','__file__':__file__})
    manifest=ROOT/cfg['logical_output']/'data_manifest.json';record=json.loads(manifest.read_text())
    record['reused_generator_sha256']=hashlib.sha256(base.read_bytes()).hexdigest()
    record['v10_seeds_disjoint']=True
    manifest.write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
