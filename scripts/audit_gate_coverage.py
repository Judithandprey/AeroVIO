"""Count motion regimes from truth only, without reading estimator scores."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from viobench.tools.datasets.storage import read_episode
from training.gate_metrics import cruise_metrics
from evaluate_velocity_memory import regime_metrics

def main():
    cfg=json.loads((ROOT/'configs/memory-gate-v12.json').read_text())
    for split in ('regression','confirmation'):
        manifest_path=ROOT/cfg['evaluation'][split+'_manifest']
        manifest=json.loads(manifest_path.read_text());rows=[]
        for entry in manifest['validation']:
            ep=read_episode(ROOT/entry['path'],frames='lazy');truth=ep.ground_truth_state()
            regime=regime_metrics(truth,truth,ep.frame_times);cruise=cruise_metrics(truth,truth,ep.frame_times,cfg['motion'])
            rows.append({'scene':entry['scene'],'seed':entry['seed'],'cruise_windows':cruise['windows'],
                'turning_windows':regime['regimes']['turning']['windows'],'stationary_windows':regime['regimes']['stationary']['windows'],
                'stop_events':len(regime['stop_events'])})
        result={'scope':split,'status':manifest['status'],'estimator_scores_used':False,'episodes':rows,
            'totals':{k:sum(r[k] for r in rows) for k in ('cruise_windows','turning_windows','stationary_windows','stop_events')},
            'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
        if manifest['status']=='complete':
            p=ROOT/'experiments/memory-gate-v12'/(split+'-coverage.json');p.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
