"""Run the fixed V8 experiment into a new named evidence directory.

Example: .venv/bin/python scripts/reproduce_native_fusion.py --name native-fusion-v8-repro1
Requires the same WSL dependencies, local validation flights and preserved
teacher reference artifact as the original run. Refuses to overwrite results.
"""
import argparse
from datetime import datetime,timezone
import json
import re
import shutil

import run_native_fusion as experiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    args=p.parse_args()
    if not re.fullmatch(r'native-fusion-v8-[a-zA-Z0-9_-]+',args.name):
        p.error('name must start with native-fusion-v8- and contain only letters, numbers, hyphens or underscores')
    experiment.OUT=experiment.ROOT/'results'/args.name
    experiment.ARCHIVE=experiment.ROOT/'experiments'/args.name
    if experiment.OUT.exists() or experiment.ARCHIVE.exists(): p.error('preserve existing evidence; choose a new name')
    protocol=json.loads(experiment.PROTOCOL.read_text())
    environment=experiment.verify_environment(protocol)
    experiment.OUT.mkdir(); experiment.ARCHIVE.mkdir()
    environment['reproduction_wrapper_sha256']=experiment.sha(__file__)
    experiment.write(experiment.ARCHIVE/'environment.json',environment)
    artifacts=experiment.build(protocol)
    experiment.validate(protocol,artifacts)
    shutil.copy2(experiment.OUT/'artifacts.json',experiment.ARCHIVE/'artifacts.json')
    selection=json.loads((experiment.ARCHIVE/'selection.json').read_text())
    experiment.write(experiment.ARCHIVE/'public-repeat-plan.json',{
        'selected_variant':selection['selected_variant'],
        'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'reason':'Full public repeat of validation-selected candidate, frozen before public runs.'})
    experiment.public(protocol,artifacts)
    # Import only after assigning the fresh paths; reporter imports these values.
    import report_native_fusion
    report_native_fusion.main()


if __name__=='__main__': main()
