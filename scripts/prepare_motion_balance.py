"""Freeze training-only motion labels and balanced sampling before either fit."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from viobench.tools.datasets.storage import read_episode
from viobench.tools.geometry import poses_to_matrices, rotation_angles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.motion_sampling import classify_motion, balanced_order


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol_path = ROOT/'configs/data-balance-v9.json'
    cfg = json.loads(protocol_path.read_text())
    output = ROOT/'data/motion-balance-v9'
    output.mkdir()  # refuse overwrites
    manifest_path = ROOT/cfg['manifest']
    manifest = json.loads(manifest_path.read_text())
    cache = ROOT/cfg['cache']
    record = {'protocol_sha256': sha(protocol_path), 'manifest_sha256': sha(manifest_path),
              'groups': cfg['groups'], 'splits': {}, 'public_data_used': False}
    for split in ('train', 'validation'):
        labels, speeds, turns, accels, per_episode = [], [], [], [], []
        with np.load(cache/(split+'.npz')) as z:
            target, episode_ids = z['target'], z['episode_id']
        for i, entry in enumerate(manifest[split]):
            episode = read_episode(ROOT/entry['path'], frames='lazy')
            truth, times = episode.ground_truth_state(), episode.frame_times
            speed = np.linalg.norm(np.diff(truth[:,:3],axis=0),axis=1)/np.diff(times)
            r = poses_to_matrices(truth[:,:7])[:,:3,:3]
            turn = np.degrees(rotation_angles(np.swapaxes(r[:-1],1,2)@r[1:]))/np.diff(times)
            group, accel = classify_motion(times, speed, turn, cfg['groups'])
            indices = np.flatnonzero(episode_ids == i)
            assert len(indices) == len(group)
            assert np.allclose(np.linalg.norm(target[indices,:3],axis=1), speed*np.diff(times),atol=1e-6)
            labels.extend(group); speeds.extend(speed); turns.extend(turn); accels.extend(accel)
            per_episode.append({'scene': entry['scene'], 'seed': entry['seed'],
                                'counts': np.bincount(group,minlength=3).tolist(),
                                'episode_json_sha256': sha(ROOT/entry['path']/'episode.json')})
        labels = np.array(labels, dtype=np.int64)
        assert len(labels) == len(target)
        np.savez_compressed(output/(split+'-labels.npz'),labels=labels, speed=speeds,
                            turn_deg_s=turns, tangential_acceleration=accels, episode_id=episode_ids)
        record['splits'][split] = {'pairs':len(labels), 'counts':np.bincount(labels,minlength=3).tolist(),
            'source_cache_sha256':sha(cache/(split+'.npz')), 'episodes':per_episode}
        if split == 'train':
            settings = cfg['training']
            order = np.stack([balanced_order(labels,e,settings['seed']) for e in range(settings['epochs'])])
            assert order.shape == (settings['epochs'],settings['samples_per_epoch'])
            counts = np.stack([np.bincount(labels[o],minlength=3) for o in order])
            assert np.all(counts == settings['samples_per_epoch']//3)
            np.savez_compressed(output/'balanced-plan.npz',order=order)
            record['balanced_plan_sha256'] = sha(output/'balanced-plan.npz')
            record['sampled_group_counts_each_epoch'] = counts[0].tolist()
            record['unique_training_pairs_seen'] = int(len(np.unique(order)))
            record['sampled_per_episode_total'] = np.bincount(episode_ids[order.ravel()],minlength=len(per_episode)).tolist()
    (output/'sampling-audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__': main()
