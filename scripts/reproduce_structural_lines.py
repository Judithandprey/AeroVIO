"""Replay the frozen line experiment into new directories without replacing evidence.

Requires the already-generated development/confirmation data and original V8
artifact. This wrapper does not regenerate data or alter the frozen estimator.
"""
import argparse,contextlib,hashlib,json,re,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    args=parser.parse_args()
    if not re.fullmatch(r'structural-lines-v13-repro-[a-zA-Z0-9_-]+',args.name):
        parser.error('name must start with structural-lines-v13-repro- and contain safe letters/numbers')
    output=ROOT/'results'/args.name;archive=ROOT/'experiments'/args.name
    if output.exists() or archive.exists():parser.error('preserve existing evidence; choose an unused name')
    cfg=json.loads((ROOT/'configs/structural-lines-v13.json').read_text())
    for field in ('development_manifest','confirmation_manifest'):
        record=json.loads((ROOT/cfg[field]).read_text())
        if record['status']!='complete':parser.error('complete preserved data are required')
    output.mkdir();archive.mkdir()
    record={'name':args.name,'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'started_utc':datetime.now(timezone.utc).isoformat(),'data_regenerated':False,
            'original_evidence_overwritten':False,'note':'Run with no concurrent generation/build/training for comparable timing.'}
    (archive/'reproduction-wrapper.json').write_text(json.dumps(record,indent=2)+'\n')
    def invoke(label,module,arguments):
        previous=sys.argv;sys.argv=[str(module.__file__),*arguments]
        print('START',label,flush=True)
        try:
            with (output/(label+'.log')).open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                module.main()
        finally:sys.argv=previous
        print('DONE',label,flush=True)
    import build_structural_lines as build
    build.OUT=output;build.ARCHIVE=archive
    invoke('build',build,['--attempt','repro'])
    import evaluate_structural_lines as evaluate
    evaluate.OUT=output;evaluate.ARCHIVE=archive
    invoke('development',evaluate,['--phase','development'])
    invoke('confirmation',evaluate,['--phase','confirmation'])
    # The reporter imports output paths from the already configured evaluator.
    import report_structural_lines as report
    invoke('report',report,[])
    print('COMPLETE',archive/'README.md',flush=True)
if __name__=='__main__':main()
