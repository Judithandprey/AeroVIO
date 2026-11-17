"""Build isolated V8+line variants; unique attempt folders preserve all failures."""
import argparse,hashlib,json,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/structural-lines-v13';ARCHIVE=ROOT/'experiments/structural-lines-v13'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);args=p.parse_args()
    if not args.attempt.isalnum():raise ValueError('attempt must be alphanumeric')
    cfg=json.loads((ROOT/'configs/structural-lines-v13.json').read_text())
    assert sha(ROOT/cfg['baseline_path'])==cfg['baseline_sha256']
    assert sha(ROOT/cfg['checkpoint'])==cfg['checkpoint_sha256']
    if (ARCHIVE/'frozen-before-confirmation.json').exists():raise RuntimeError('preserve frozen experiment')
    folder=OUT/'builds'/args.attempt;folder.mkdir(parents=True,exist_ok=False)
    ARCHIVE.mkdir(exist_ok=True)
    native=ROOT/'workspace/structural_lines_native'
    sources={str(f.relative_to(ROOT)):sha(f) for f in native.iterdir() if f.is_file()}
    for name in ('inference.h','export_weights.py','gyro_bias.h','rotation.h','visual_direction.h','fusion_geometry.h','split_corners.h'):
        assert sha(native/name)==sha(ROOT/'workspace/visual_native'/name),name
    def run(label,command):
        print('START',label,flush=True)
        with (folder/(label+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        with (folder/'pipeline.jsonl').open('a') as f:f.write(json.dumps({'label':label,'command':command,'exit_code':result.returncode})+'\n')
        if result.returncode:raise RuntimeError(label+' failed; evidence retained')
        print('DONE',label,flush=True)
    baseline=ROOT/cfg['baseline_path']
    artifacts={'baseline':{'path':str(baseline),'sha256':sha(baseline),'bytes':baseline.stat().st_size}}
    for name,mode in [('control',0),('detect',1),('correct',2)]:
        build=folder/('build-'+name)
        run('configure-'+name,['cmake','-S',str(native),'-B',str(build),
            '-DCMAKE_TOOLCHAIN_FILE='+str(ROOT/'workspace/build/toolchain/wasi-sdk/share/cmake/wasi-sdk-p1.cmake'),
            '-DPython3_EXECUTABLE='+sys.executable,'-DVIOBENCH_OPENCV_DIR='+str(ROOT/'workspace/build/runtime/opencv-build'),
            '-DVISUAL_CONFIDENCE_WEIGHTING=OFF','-DVISUAL_ROTATION_CORRECTION=ON',
            '-DLINE_MODE='+str(mode),'-DBUILD_FUSION_TESTS='+('ON' if mode==0 else 'OFF')])
        run('build-'+name,['cmake','--build',str(build),'-j','4'])
        target=folder/('AeroVIO-native-v13-'+name+'.wasm');shutil.copy2(build/'solution.wasm',target)
        run('check-'+name,[sys.executable,'-m','viobench','check',str(target),'--frames','80','--logs',str(folder/('check-'+name))])
        artifacts[name]={'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size}
        if mode==0:
            run('geometry-tests',[sys.executable,str(ROOT/'scripts/test_structural_lines.py'),str(build/'fusion_tests.wasm')])
    assert sources=={str(f.relative_to(ROOT)):sha(f) for f in native.iterdir() if f.is_file()}
    write(folder/'artifacts.json',artifacts);write(ARCHIVE/'artifacts.json',artifacts)
    write(folder/'build-verification.json',{'original_v8_components_unchanged':True,'source_sha256':sources,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'attempt':args.attempt})
    shutil.copy2(folder/'build-verification.json',ARCHIVE/'build-verification.json')
    shutil.copy2(folder/'geometry-tests.log',ARCHIVE/('geometry-tests-'+args.attempt+'.txt'))
    print('BUILT',args.attempt,flush=True)
if __name__=='__main__':main()
