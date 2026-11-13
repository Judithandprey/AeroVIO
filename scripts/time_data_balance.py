"""Timing-only paired follow-up after serial V9 timing showed host variability.

Fixed artifacts and inputs; alternate which model steps first on every frame.
Verify full outputs/fuel against completed public reports; never select weights.
"""
import json
from pathlib import Path

import numpy as np
from viobench.tools.datasets.sources import episodes
from viobench.tools.runner.engine import Engine
from viobench.tools.runner.sandbox import Sandbox, Limits

from run_data_balance import ROOT, OUT, ARCHIVE, PROTOCOL, sha, write


def main():
    destination=OUT/'paired-timing'; destination.mkdir()
    cfg=json.loads(PROTOCOL.read_text()); artifacts=json.loads((ARCHIVE/'artifacts.json').read_text())
    write(destination/'plan.json',{'purpose':'Timing only; serial same-binary median timings varied substantially.',
        'models':['teacher','balanced'],'frames':'all 10 unchanged public episodes; no subsampling',
        'order':'teacher first on even (episode index + frame index), balanced first on odd',
        'reported':'all-frame pooled median, median per-episode ratio, and full per-episode medians',
        'artifact_sha256':{n:artifacts[n]['sha256'] for n in ('teacher','balanced')},
        'selection_sha256':sha(ARCHIVE/'selection.json'),'no_training_or_selection_change':True})
    references=json.loads((ROOT/cfg['public_episodes']).read_text())
    engine=Engine(); limits=Limits.from_dict(json.loads((ROOT/cfg['limits']).read_text()))
    rows=[];all_wall={n:[] for n in ('teacher','balanced')}
    for ei,episode in enumerate(episodes(references,frames='lazy')):
        config,frames=episode.adapt((320,240));sandboxes={};results={n:[] for n in all_wall}
        fuels={n:[] for n in all_wall};wall={n:[] for n in all_wall}
        for name in all_wall:
            wasm=Path(artifacts[name]['path']);assert sha(wasm)==artifacts[name]['sha256']
            sandboxes[name]=Sandbox(wasm,engine,limits=limits,log_dir=destination/episode.hash/name)
            sandboxes[name].init(config)
        for i,(t,f) in enumerate(zip(episode.frame_times,frames,strict=True)):
            imu=episode.imu_window(i).copy()
            order=('teacher','balanced') if (ei+i)%2==0 else ('balanced','teacher')
            for name in order:
                r=sandboxes[name].step(float(t),f,imu)
                results[name].append(np.r_[r.pose,r.twist]);fuels[name].append(r.fuel);wall[name].append(r.wall_seconds)
        medians={}
        for name in all_wall:
            reports=list((OUT/('public-'+name)).glob('report-*/report.json'));assert len(reports)==1
            log=reports[0].parent/'logs'/episode.hash
            original=np.c_[np.loadtxt(log/'estimate.tum')[:,1:],np.loadtxt(log/'twist.txt')[:,1:]]
            assert np.allclose(results[name],original,atol=1e-7,rtol=1e-9),(ei,name)
            np.testing.assert_array_equal(fuels[name],np.load(log/'fuel.npy'))
            np.savez_compressed(destination/episode.hash/name/'trace.npz',outputs=results[name],fuel=fuels[name],wall=wall[name])
            all_wall[name].extend(wall[name]);medians[name]=float(np.median(wall[name])*1000)
        rows.append({'hash':episode.hash,'median_step_ms':medians,
                     'balanced_teacher_ratio':medians['balanced']/medians['teacher'],
                     'fuel_exact':True,'saved_text_pose_twist_tolerance':1e-7})
        print('PAIRED',ei,medians,flush=True)
    write(ARCHIVE/'paired-timing.json',{
        'plan':json.loads((destination/'plan.json').read_text()),'episodes':rows,'pairs':sum(len(t) for t in [all_wall['teacher']]),
        'median_step_ms':{n:float(np.median(t)*1000) for n,t in all_wall.items()},
        'median_episode_ratio':float(np.median([r['balanced_teacher_ratio'] for r in rows])),
        'episode_ratio_quartiles':np.quantile([r['balanced_teacher_ratio'] for r in rows],[.25,.75]).tolist(),
        'source_sha256':sha(__file__),
        'interpretation':'Paired warm-process timing reduces temporal host drift. It is not a server or hardware-independent latency guarantee.'})


if __name__=='__main__':main()
