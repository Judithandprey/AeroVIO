"""Timing-only three-model follow-up without a duplicate artifact warming caches."""
import itertools,json
from pathlib import Path
import numpy as np
from viobench.tools.datasets.storage import read_episode
from viobench.tools.runner.engine import Engine
from viobench.tools.runner.sandbox import Sandbox,Limits
from run_joint_balance import ROOT,OUT,ARCHIVE,PROTOCOL,sha,write

def main():
    output=OUT/'timing-three-models';output.mkdir()
    cfg=json.loads(PROTOCOL.read_text());manifest=json.loads((ROOT/cfg['long_manifest']).read_text())
    artifacts=json.loads((ARCHIVE/'artifacts.json').read_text());names=list(artifacts)
    assert names==['teacher','category-v9','joint-v10']
    permutations=list(itertools.permutations(names));rng=np.random.default_rng(20261112)
    write(output/'plan.json',{'scope':'timing only, after accuracy evaluation; unchanged three distinct artifacts',
        'reason':'four-way run included a duplicate V10 instance and showed cache-sensitive timing',
        'order':'all six permutations per six-frame block, shuffled with a fixed independent seed',
        'seed':20261112,'episodes':6,'frames_per_episode':1200,
        'artifacts_sha256':sha(ARCHIVE/'artifacts.json'),'selection_sha256':sha(ARCHIVE/'selection.json')})
    engine=Engine();limits=Limits.from_dict(json.loads((ROOT/cfg['limits']).read_text()))
    pooled={n:[] for n in names};rows=[]
    for ei,entry in enumerate(manifest['validation']):
        ep=read_episode(ROOT/entry['path'],frames='lazy');config,frames=ep.adapt((320,240))
        sb={};expected={};expected_fuel={};walls={n:[] for n in names};order_counts={n:[0,0,0] for n in names}
        for name in names:
            wasm=Path(artifacts[name]['path']);assert sha(wasm)==artifacts[name]['sha256']
            sb[name]=Sandbox(wasm,engine,limits=limits,log_dir=output/str(ei)/name);sb[name].init(config)
            with np.load(OUT/'long-validation'/str(ei)/name/'trace.npz') as z:
                expected[name]=z['outputs'];expected_fuel[name]=z['fuel']
        for i,(t,frame) in enumerate(zip(ep.frame_times,frames,strict=True)):
            if i%6==0:block=rng.permutation(6)
            order=permutations[int(block[i%6])];imu=ep.imu_window(i).copy()
            for position,name in enumerate(order):
                r=sb[name].step(float(t),frame,imu)
                assert np.array_equal(np.r_[r.pose,r.twist],expected[name][i]),(ei,i,name)
                assert r.fuel==expected_fuel[name][i]
                walls[name].append(r.wall_seconds);order_counts[name][position]+=1
        for name in names:
            assert order_counts[name]==[400,400,400]
            pooled[name].extend(walls[name]);np.save(output/str(ei)/name/'wall.npy',np.array(walls[name]))
        row={'scene':entry['scene'],'seed':entry['seed'],'position_counts':order_counts,
            'pose_twist_fuel_exact_vs_accuracy_run':True,
            'median_step_ms':{n:float(np.median(walls[n])*1000) for n in names}}
        rows.append(row);print('TIMED',ei,row['median_step_ms'],flush=True)
    ratios={n:[r['median_step_ms'][n]/r['median_step_ms']['teacher'] for r in rows] for n in names}
    write(ARCHIVE/'timing-comparison.json',{'plan':json.loads((output/'plan.json').read_text()),
        'source_sha256':sha(__file__),'episodes':rows,'frames_per_model':7200,
        'median_step_ms':{n:float(np.median(pooled[n])*1000) for n in names},
        'median_episode_ratio_vs_teacher':{n:float(np.median(v)) for n,v in ratios.items()},
        'scope':'fixed random order with exact first/second/third-position balance; host-specific, no accuracy/selection change'})

if __name__=='__main__':main()
