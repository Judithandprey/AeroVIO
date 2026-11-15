"""Package selected gates; preserve V10/V11 and verify frozen predictors/repeat."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/memory-gate-v12';A=ROOT/'experiments/memory-gate-v12'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def run(label,command):
    print('START',label,flush=True)
    with (OUT/(label+'.log')).open('x') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',label,flush=True)

def main():
    cfg=read(ROOT/'configs/memory-gate-v12.json');A.mkdir()
    runs={arm:read(ROOT/'models'/('gate-v12-'+arm)/'run.json') for arm in cfg['arms']}
    for key in ('model_sha256','initial_gate_sha256','best_epoch','best_validation','frozen_predictor_tensor_sha256'):
        assert runs['adaptive'][key]==runs['adaptive-repeat'][key],key
    assert read(ROOT/'models/gate-v12-adaptive/history.json')==read(ROOT/'models/gate-v12-adaptive-repeat/history.json')
    for arm,r in runs.items():
        for p,h in r['source_sha256'].items():assert sha(ROOT/p)==h,p
        assert r['predictors_unchanged'] and r['steps']==1520
        dest=A/arm;dest.mkdir()
        for name in ('meta.json','net.onnx','run.json','history.json'):shutil.copy2(ROOT/'models'/('gate-v12-'+arm)/name,dest/name)
    for name in ('memory-gate-v12.json','long-validation-v12.json'):shutil.copy2(ROOT/'configs'/name,A/name)
    shutil.copy2(OUT/'host-tests.log',A/'host-tests.txt')
    for arm in cfg['arms']:shutil.copy2(OUT/('train-'+arm+'.log'),A/('train-'+arm+'.log'))
    original=read(ROOT/'experiments/velocity-memory-v11/artifacts.json')
    artifacts={'v10':original['v10'],'v11':original['feedback-final']}
    for p in artifacts.values():assert sha(p['path'])==p['sha256']
    for label,arm,force,diagnostic in [('constant','constant',-1,False),('adaptive','adaptive',-1,False),
                                     ('endpoint-v10','adaptive',0,False),('endpoint-v11','adaptive',1,False),
                                     ('adaptive-parity','adaptive',-1,True)]:
        src=OUT/('source-'+label);shutil.copytree(ROOT/'experiments/velocity-memory-v11/source-feedback-final',src)
        checkpoint=ROOT/'models'/('gate-v12-'+arm)
        for file in ('inference.h','export_weights.py'):shutil.copy2(ROOT/'workspace/memory_gate_native'/file,src/file)
        write(src/'model-paths.json',{'base':str(ROOT/cfg['v10']),'memory':str(ROOT/cfg['v11']),'gate':str(checkpoint)})
        p=src/'CMakeLists.txt';text=p.read_text()
        old='set(checkpoint "/root/projects/AeroVIO/models/velocity-v11-feedback-final")';assert text.count(old)==1
        text=text.replace(old,'set(checkpoint "'+str(checkpoint)+'")')
        text=text.replace('target_compile_definitions(solution PRIVATE','target_compile_definitions(solution PRIVATE\n    GATE_FORCE='+str(force)+' GATE_DIAGNOSTICS='+str(int(diagnostic)))
        p.write_text(text)
        if diagnostic:
            p=src/'solution.cpp';text=p.read_text();needle='            state.memory_valid = memory_dt > 0. && memory_dt <= .2;'
            assert text.count(needle)==1
            text=text.replace(needle,needle+'''
            std::fprintf(stderr,"GATE");
            for(float x:model_input)std::fprintf(stderr," %.9g",double(x));
            for(float x:last_gate_features)std::fprintf(stderr," %.9g",double(x));
            std::fprintf(stderr," %.9g",double(last_gate_value));
            for(float x:motion)std::fprintf(stderr," %.9g",double(x));
            for(float x:state.memory_velocity)std::fprintf(stderr," %.9g",double(x));
            for(double x:preintegrated.gyro_rotation.val)std::fprintf(stderr," %.17g",x);
            std::fprintf(stderr,"\\n");''');p.write_text(text)
        for name in ('gyro_bias.h','rotation.h','visual_direction.h','split_corners.h'):
            assert sha(src/name)==sha(ROOT/'experiments/velocity-memory-v11/source-feedback-final'/name)
        write(src/'provenance.json',{'parent_commit':cfg['parent_commit'],'force_gate':force,'diagnostic':diagnostic,
            'frozen_models':{k:sha(ROOT/cfg[k]/'net.onnx') for k in ('v10','v11')},'gate_model_sha256':sha(checkpoint/'net.onnx'),
            'source_sha256':{p.name:sha(p) for p in src.iterdir() if p.name!='provenance.json'}})
        build=OUT/('build-'+label)
        run('configure-'+label,['cmake','-S',str(src),'-B',str(build),
            '-DCMAKE_TOOLCHAIN_FILE='+str(ROOT/'workspace/build/toolchain/wasi-sdk/share/cmake/wasi-sdk-p1.cmake'),
            '-DPython3_EXECUTABLE='+sys.executable,'-DVIOBENCH_NATIVE_DIR='+str(ROOT/'workspace/runtime/native'),
            '-DVIOBENCH_OPENCV_DIR='+str(ROOT/'workspace/build/runtime/opencv-build'),'-DVISUAL_DOWNSAMPLE=2','-DVISUAL_SPARSE_DIRECTION=ON'])
        run('build-'+label,['cmake','--build',str(build),'-j','4'])
        wasm=OUT/('AeroVIO-native-v12-'+label+'.wasm');shutil.copy2(build/'solution.wasm',wasm)
        run('check-'+label,[sys.executable,'-m','viobench','check',str(wasm),'--frames','80','--logs',str(OUT/('check-'+label))])
        artifacts[label]={'path':str(wasm),'sha256':sha(wasm),'bytes':wasm.stat().st_size}
        shutil.copytree(src,A/('source-'+label))
    write(A/'artifacts.json',artifacts)
    write(A/'training-verification.json',{'adaptive_repeat_exact':True,'all_frozen_predictors_unchanged':True,
        'same_target_sampling_and_steps':True,'chosen_epochs':{a:r['best_epoch'] for a,r in runs.items()},
        'trainable_parameters':{a:r['trainable_parameters'] for a,r in runs.items()}})
    print('BUILD COMPLETE',flush=True)

if __name__=='__main__':main()
