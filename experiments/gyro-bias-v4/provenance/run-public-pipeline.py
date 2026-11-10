"""Fixed serial v4 packaging/public comparison after independent validation."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

root = Path('/root/projects/AeroVIO')
protocol = root / 'configs/gyro-bias-v4.json'
selection_path = root / 'experiments/gyro-bias-v4/selection.json'
selection = json.loads(selection_path.read_text())
if not selection['validation_complete'] or not selection['repeat_verified']:
    raise SystemExit('Complete repeated validation is required before public evaluation')
if selection['protocol_sha256'] != hashlib.sha256(protocol.read_bytes()).hexdigest():
    raise SystemExit('Protocol changed after validation')
for name, expected in selection['source_sha256'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit('Source changed after validation: ' + name)
selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
names = {'current': 'gyro-v4-current', 'gyro-only': 'gyro-v4-raw',
         'calibrated-residual': 'gyro-v4-cal-residual', 'calibrated-gyro': 'gyro-v4-cal-gyro'}
if selection['selected_variant'] not in names:
    raise SystemExit('Unknown selected variant')
status = root / 'logs/gyro-v4-pipeline.json'
if status.exists():
    raise SystemExit('Preserve existing pipeline evidence')
stages = []
for variant, name in names.items():
    command = ['.venv/bin/python', 'scripts/package_motion.py', '--model',
               'experiments/learned-motion-v1/warp', '--name', name]
    if variant.startswith('calibrated-'):
        command.append('--calibrate-gyro')
    if variant in ('gyro-only', 'calibrated-gyro'):
        command.append('--disable-rotation-residual')
    stages.append(('package-' + variant, command))
    stages.append(('check-' + variant, ['.venv/bin/python', '-u', '-m', 'viobench',
        'check', f'artifacts/{name}.wasm', '--frames', '40', '--logs', f'logs/check-{name}']))
for variant, name in [('baseline', 'klt_imu'), *names.items(),
                      ('selected-repeat', names[selection['selected_variant']]),
                      ('calibrated-gyro-repeat', names['calibrated-gyro'])]:
    command = ['.venv/bin/python', '-u', 'scripts/evaluate_public.py', '--wasm',
               f'artifacts/{name}.wasm', '--run-dir', f'results/gyro-v4-{variant}']
    if variant == 'baseline':
        command.append('--verify-baseline')
    stages.append(('evaluate-' + variant, command))
records = []
for stage, command in stages:
    if hashlib.sha256(selection_path.read_bytes()).hexdigest() != selection_sha:
        raise SystemExit('Selection changed during public evaluation')
    print('START ' + stage, flush=True)
    log = root / ('logs/gyro-v4-' + stage + '.log')
    start = time.perf_counter()
    with log.open('x') as handle:
        result = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
    records.append({'stage': stage, 'command': command, 'exit_code': result.returncode,
                    'elapsed_s': time.perf_counter() - start, 'log': str(log),
                    'selection_sha256': hashlib.sha256(selection_path.read_bytes()).hexdigest()})
    status.write_text(json.dumps(records, indent=2) + '\n')
    print(f'END {stage}: {result.returncode}', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
