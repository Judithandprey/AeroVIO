"""Generate fresh 60-second validation flights using the unchanged simulator setup.

Raw data goes through a new logical symlink to D:, preserving C: disk space.
Original generator functions and successful-flight capture semantics are reused.
"""
from datetime import datetime,timezone
import importlib.metadata,json,os
from pathlib import Path
import platform
from generate_training_data import ROOT,world_header,capture,digest,write_json

def main():
    os.environ.setdefault('HYPERLUX_BACKEND','generic');os.environ.setdefault('HYPERLUX_BUILD_JOBS','4')
    import hyperdrone,hyperlux
    config_path=ROOT/'configs/long-validation-v10.json';cfg=json.loads(config_path.read_text())
    output=ROOT/cfg['logical_output'];physical=Path(cfg['physical_output'])
    physical.mkdir(parents=True,exist_ok=True)
    if not output.exists() and not output.is_symlink():output.symlink_to(physical,target_is_directory=True)
    assert output.is_symlink() and output.resolve()==physical.resolve()
    manifest_path=output/'data_manifest.json'
    if manifest_path.exists():raise FileExistsError('preserve existing generation evidence')
    old=json.loads((ROOT/'data/hyperdrone-motion-v2/data_manifest.json').read_text())
    old_seeds={e['seed'] for split in ('train','validation') for e in old[split]}
    assert not {s['scene'] for s in cfg['validation']}&{'ProcTHOR-Train-1','ProcTHOR-Train-2','ProcTHOR-Train-3','ProcTHOR-Train-4'}
    header=world_header(output)
    assert digest(header)==old['world_header_sha256'],'unchanged world setup required'
    manifest={'schema_version':1,'status':'generating','train':[],'validation':[],'attempts':[],
        'config_sha256':digest(config_path),'generator_sha256':digest(Path(__file__)),
        'original_capture_source_sha256':digest(ROOT/'scripts/generate_training_data.py'),
        'world_header_sha256':digest(header),'created_utc':datetime.now(timezone.utc).isoformat(),
        'physical_output':str(physical),'duration_seconds':cfg['duration_seconds'],
        'packages':{p:importlib.metadata.version(p) for p in ('hyperdrone','hyperlux','viobench')},
        'host':platform.platform(),'estimator_scores_used_for_generation':False}
    write_json(manifest_path,manifest)
    catalog=hyperlux.datasets.procthor.Train200();seed=cfg['seed']
    for spec in cfg['validation']:
        scene=spec['scene'];reference=catalog[scene].reference;completed=0
        for _ in range(cfg['max_attempts_per_requested_episode']*spec['episodes']):
            assert seed not in old_seeds
            env=None
            try:
                env=hyperdrone.HyperDrone([reference],config=hyperdrone.Config(spec_header=str(header),device='cpu'),
                    seed=seed,parameters={'cameras':{'camera0':{'fov':cfg['fov_degrees']}}})
                print('GENERATE',scene,seed,flush=True)
                item,stats=capture(env,cfg['duration_seconds'],seed,scene,f'validation-{scene}-seed{seed}',output)
            finally:
                if env is not None:env.close()
            manifest['attempts'].append(stats);seed+=1
            if item is not None:
                assert item['frames']==1200
                manifest['validation'].append(item);completed+=1
            write_json(manifest_path,manifest);print(json.dumps(stats),flush=True)
            if completed==spec['episodes']:break
        if completed!=spec['episodes']:
            manifest['status']='incomplete';write_json(manifest_path,manifest)
            raise RuntimeError('predeclared non-crash quota not met: '+scene)
    manifest['status']='complete';manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
    write_json(manifest_path,manifest)
    print('COMPLETE',len(manifest['validation']),'new long flights',flush=True)

if __name__=='__main__':main()
