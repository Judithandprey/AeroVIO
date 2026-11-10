"""One-off serial evaluation driver; preserve all existing run directories."""
from pathlib import Path
import json
import subprocess
import time

root = Path('/root/projects/AeroVIO')
records = []
status = root / 'logs/rotation-v3-pipeline.json'
if status.exists():
    raise SystemExit('Existing pipeline evidence must be preserved')
stages = []
for variant in ('control', 'focus'):
    name = f'rotation-{variant}-v3'
    stages.append((f'package-{variant}', ['.venv/bin/python', 'scripts/package_motion.py',
        '--model', f'models/{name}-retained', '--name', name]))
    stages.append((f'check-{variant}', ['.venv/bin/python', '-u', '-m', 'viobench',
        'check', f'artifacts/{name}.wasm', '--frames', '20', '--logs', f'logs/check-{name}']))
    final_name = name + '-final'
    stages.append((f'package-{variant}-final', ['.venv/bin/python', 'scripts/package_motion.py',
        '--model', f'models/{name}-retained/diagnostic-last', '--name', final_name]))
    stages.append((f'check-{variant}-final', ['.venv/bin/python', '-u', '-m', 'viobench',
        'check', f'artifacts/{final_name}.wasm', '--frames', '20', '--logs', f'logs/check-{final_name}']))
for variant, wasm in [('baseline', 'klt_imu'), ('current', 'motion-current-v2'),
                      ('control', 'rotation-control-v3'), ('focus', 'rotation-focus-v3'),
                      ('control-final', 'rotation-control-v3-final'), ('focus-final', 'rotation-focus-v3-final')]:
    command = ['.venv/bin/python', '-u', 'scripts/evaluate_public.py', '--wasm',
               f'artifacts/{wasm}.wasm', '--run-dir', f'results/rotation-v3-{variant}']
    if variant == 'baseline':
        command.append('--verify-baseline')
    stages.append((f'evaluate-{variant}', command))
for stage, command in stages:
    print('START ' + stage, flush=True)
    log = root / f'logs/rotation-v3-{stage}.log'
    started = time.perf_counter()
    with log.open('x') as handle:
        result = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
    records.append({'stage': stage, 'command': command, 'exit_code': result.returncode,
                    'elapsed_s': time.perf_counter() - started, 'log': str(log)})
    status.write_text(json.dumps(records, indent=2) + '\n')
    print(f'END {stage}: {result.returncode}', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
