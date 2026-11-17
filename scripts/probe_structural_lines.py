"""Actual-WASM structural-line feasibility on the four existing development flights.

No truth or pose-estimator scores are read. Wall timing is deliberately excluded
because this diagnostic may overlap independent data generation. Fuel measures
the isolated line module only; it is not the full estimator's peak frame fuel.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import wasmtime
from viobench.tools.datasets.storage import read_episode
from viobench.tools.runner.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/hyperdrone-motion-v2/data_manifest.json'
FUEL_LIMIT = 200_000_000
HEADER = ('valid_image', 'raw_lines', 'retained_lines', 'directions',
          'rejected_short', 'rejected_duplicate', 'tested_pairs', 'rejected_geometry')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Probe:
    def __init__(self, engine, module):
        self.engine = engine
        self.store = wasmtime.Store(engine.wasmtime)
        self.store.set_fuel(FUEL_LIMIT)
        self.store.set_epoch_deadline(engine.ticks(60))
        wasi = wasmtime.WasiConfig(); wasi.inherit_stdout(); wasi.inherit_stderr()
        self.store.set_wasi(wasi)
        linker = wasmtime.Linker(engine.wasmtime); linker.define_wasi()
        self.instance = linker.instantiate(self.store, module)
        self.exports = self.instance.exports(self.store)
        self.exports['_initialize'](self.store)
        self.gray_ptr = self.exports['alloc'](self.store, 160*120)
        self.out_ptr = self.exports['alloc'](self.store, 128*8)
        self.memory = self.exports['memory']

    def run(self, gray, intrinsics):
        self.memory.write(self.store, gray.tobytes(), self.gray_ptr)
        self.store.set_fuel(FUEL_LIMIT)
        self.store.set_epoch_deadline(self.engine.ticks(60))
        status = self.exports['probe'](self.store, self.gray_ptr, *intrinsics, self.out_ptr)
        fuel = FUEL_LIMIT - self.store.get_fuel()
        assert status == 0, status
        values = np.frombuffer(self.memory.read(self.store, self.out_ptr, self.out_ptr+128*8), dtype='<f8').copy()
        assert np.isfinite(values).all()
        return values, fuel

def green160(rgb):
    assert rgb.shape == (240, 320, 3) and rgb.dtype == np.uint8
    # Exact native make_gray: 2x2 GREEN sum with integer round-half-up.
    sums = rgb[:, :, 1].astype(np.uint16).reshape(120, 2, 160, 2).sum(axis=(1, 3), dtype=np.uint16)
    return np.ascontiguousarray(((sums+2)//4).astype(np.uint8))

def record(values, fuel, frame, gray):
    row = {k: int(values[i]) for i, k in enumerate(HEADER)}
    row.update(frame=frame, fuel=fuel, gray_sha256=hashlib.sha256(gray.tobytes()).hexdigest())
    row['lines'] = values[8:104].reshape(24, 4)[:row['retained_lines']].tolist()
    row['vanishing_directions'] = [dict(zip(('axis_x','axis_y','axis_z','support','residual_rad',
        'conditioning','center_span_px','support_mask'), d.tolist()))
        for d in values[104:].reshape(3, 8)[:row['directions']]]
    return row

def overlay(gray, row, label):
    colors = [(80, 230, 90), (255, 150, 40), (220, 70, 220)]
    panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, coordinates in enumerate(row['lines']):
        color = (140, 140, 140)
        for j, direction in enumerate(row['vanishing_directions']):
            if int(direction['support_mask']) & (1 << i): color = colors[j]
        a,b,c,d = (int(round(x)) for x in coordinates)
        cv2.line(panel, (a,b), (c,d), color, 1, cv2.LINE_8)
    panel = cv2.resize(panel, (480, 360), interpolation=cv2.INTER_NEAREST)
    title = np.full((48, 480, 3), 245, dtype=np.uint8)
    cv2.putText(title, label, (8,18), cv2.FONT_HERSHEY_SIMPLEX, .44, (0,0,0), 1)
    cv2.putText(title, f"lines {row['retained_lines']} dirs {row['directions']} fuel {row['fuel']:,}",
                (8,39), cv2.FONT_HERSHEY_SIMPLEX, .44, (0,0,0), 1)
    return np.concatenate((title, panel))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wasm', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['status'] == 'complete' and len(manifest['validation']) == 4
    engine = Engine(); module = engine.module(args.wasm.read_bytes())
    cases = Probe(engine, module)
    cases.store.set_fuel(FUEL_LIMIT)
    cases.store.set_epoch_deadline(engine.ticks(60))
    failures = cases.exports['run_tests'](cases.store)
    assert failures == 0, 'native detector synthetic tests failed'
    tests_fuel = FUEL_LIMIT-cases.store.get_fuel()
    episodes=[]; panels=[]; allrows=[]
    for entry in manifest['validation']:
        episode=read_episode(ROOT/entry['path'], frames='lazy')
        config, frames=episode.adapt((320,240))
        intrinsics=(config['fx']/2,config['fy']/2,(config['cx']+.5)/2-.5,(config['cy']+.5)/2-.5)
        primary=Probe(engine,module); repeated=Probe(engine,module)
        rows=[]; images={}
        for i,rgb in enumerate(frames):
            if i % 8: continue
            gray=green160(rgb)
            values,fuel=primary.run(gray,intrinsics)
            again,repeat_fuel=repeated.run(gray,intrinsics)
            assert np.array_equal(values,again) and fuel == repeat_fuel
            row=record(values,fuel,i,gray); rows.append(row); images[i]=gray
        assert len(rows)==25 and [r['frame'] for r in rows] == list(range(0,200,8))
        # Four fixed times plus first missing-structure and first two-direction frame.
        selected={0,64,128,192}
        for condition in (lambda r:r['directions']==0, lambda r:r['directions']>=2):
            found=next((r['frame'] for r in rows if condition(r)),None)
            if found is not None: selected.add(found)
        for row in rows:
            if row['frame'] in selected:
                panels.append(overlay(images[row['frame']],row,f"{entry['scene']} seed{entry['seed']} frame{row['frame']}"))
        summary={'scene':entry['scene'],'seed':entry['seed'],'path':entry['path'], 'probes':len(rows),
            'mean_retained_lines':float(np.mean([r['retained_lines'] for r in rows])),
            'zero_directions':sum(r['directions']==0 for r in rows),
            'one_direction':sum(r['directions']==1 for r in rows),
            'at_least_two_directions':sum(r['directions']>=2 for r in rows),
            'at_least_two_fraction':sum(r['directions']>=2 for r in rows)/len(rows),
            'max_probe_fuel':max(r['fuel'] for r in rows),
            'mean_probe_fuel':float(np.mean([r['fuel'] for r in rows])),
            'selected_overlay_frames':sorted(selected),'camera_intrinsics_160':intrinsics,'rows':rows}
        episodes.append(summary);allrows.extend(rows)
        print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
    width=4; blank=np.full_like(panels[0],245)
    while len(panels)%width: panels.append(blank.copy())
    contact=np.concatenate([np.concatenate(panels[i:i+width],axis=1) for i in range(0,len(panels),width)],axis=0)
    assert cv2.imwrite(str(args.out/'contact-sheet.png'),contact)
    sources=['workspace/structural_lines_native/structural_lines.h','tests/structural_detector_cases.h',
        'tests/structural_line_probe.cpp','tests/line_probe/CMakeLists.txt','scripts/probe_structural_lines.py']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'four original development flights only',
        'truth_or_pose_scores_read':False,'frame_interval':8,'preprocessing':'integer rounded 2x2 GREEN from320x240 adapted RGB',
        'wall_timing_excluded':True,'fuel_scope':'isolated WASM probe; full estimator peak fuel must be measured separately',
        'wasm':str(args.wasm),'wasm_sha256':sha(args.wasm),'manifest_sha256':sha(MANIFEST),
        'source_sha256':{p:sha(ROOT/p) for p in sources},'synthetic_tests_failed':failures,'synthetic_tests_fuel':tests_fuel,
        'all_outputs_and_fuel_repeat_exact':True,'probes':len(allrows),
        'max_probe_fuel':max(r['fuel'] for r in allrows),'mean_probe_fuel':float(np.mean([r['fuel'] for r in allrows])),
        'at_least_two_fraction':sum(r['directions']>=2 for r in allrows)/len(allrows),
        'episodes':episodes,'packages':{p:importlib.metadata.version(p) for p in ('numpy','wasmtime','viobench','opencv-python-headless')}}
    (args.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PROBE COMPLETE',len(allrows),'frames; max fuel',result['max_probe_fuel'],flush=True)

if __name__=='__main__': main()
