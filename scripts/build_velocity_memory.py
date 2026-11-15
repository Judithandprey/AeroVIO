"""Build both 18-input arms from teacher sources; retain source and binary provenance."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/velocity-memory-v11';ARCHIVE=ROOT/'experiments/velocity-memory-v11'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def replace(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)
def run(label,cmd):
    print('START',label,flush=True)
    with (OUT/(label+'.log')).open('x') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',label,flush=True)

def source_tree(cfg,arm,diagnostic=False):
    name=arm+('-parity' if diagnostic else '');src=OUT/('source-'+name);src.mkdir()
    checkpoint=ROOT/'models'/('velocity-v11-'+arm)
    paths=subprocess.check_output(['git','ls-tree','--name-only',cfg['teacher_commit'],'workspace/visual_native/'],cwd=ROOT,text=True).splitlines()
    original={}
    for name0 in paths:
        data=subprocess.check_output(['git','show',cfg['teacher_commit']+':'+name0],cwd=ROOT)
        p=src/Path(name0).name;p.write_bytes(data);original[p.name]=sha(p)
    p=src/'export_weights.py';s=p.read_text()
    s=replace(s,'CHECKPOINT = ROOT / "experiments/motion-ablation-v2/imu-only-compact"','CHECKPOINT = Path('+repr(str(checkpoint))+')')
    s=replace(s,'[(layers[0], 14, 64), (layers[2], 64, 6)]','[(layers[0], 18, 64), (layers[2], 64, 6)]');p.write_text(s)
    p=src/'CMakeLists.txt';s=p.read_text();s=replace(s,'set(checkpoint "${CMAKE_CURRENT_SOURCE_DIR}/../../experiments/motion-ablation-v2/imu-only-compact")','set(checkpoint "'+str(checkpoint)+'")')
    p.write_text(s)
    p=src/'inference.h';s=p.read_text();s=replace(s,'infer(std::array<float, 14> input)','infer(std::array<float, 18> input)');p.write_text(s)
    p=src/'solution.cpp';s=p.read_text()
    s=replace(s,'    bool started = false;','    bool started = false;\n    std::array<float, 3> memory_velocity{};\n    bool memory_valid = false;')
    enabled='true' if arm.startswith('feedback') else 'false'
    injection='''            std::array<float, 18> model_input{};
            std::copy(preintegrated.input.begin(), preintegrated.input.end(), model_input.begin());
            const double memory_dt = time - state.previous_time;
            const bool use_history = ENABLED && state.memory_valid && memory_dt <= .2;
            if (use_history) {
                std::copy(state.memory_velocity.begin(), state.memory_velocity.end(), model_input.begin() + 14);
                model_input[17] = 1.f;
            }
            const auto motion = infer(model_input);
            const auto next_velocity = preintegrated.gyro_rotation.t() * Vec3d(motion[0], motion[1], motion[2]) * (1. / memory_dt);
            for (int axis = 0; axis < 3; ++axis)
                state.memory_velocity[axis] = float(std::clamp(next_velocity[axis], -2., 2.));
            state.memory_valid = memory_dt > 0. && memory_dt <= .2;'''.replace('ENABLED',enabled)
    if diagnostic:
        injection+='''
            std::fprintf(stderr, "MEMORY");
            for (float x : model_input) std::fprintf(stderr, " %.9g", double(x));
            for (float x : motion) std::fprintf(stderr, " %.9g", double(x));
            for (float x : state.memory_velocity) std::fprintf(stderr, " %.9g", double(x));
            for (double x : preintegrated.gyro_rotation.val) std::fprintf(stderr, " %.17g", x);
            std::fprintf(stderr, "\\n");'''
    s=replace(s,'            const auto motion = infer(preintegrated.input);',injection);p.write_text(s)
    write(src/'provenance.json',{'teacher_commit':cfg['teacher_commit'],'original':original,
        'modified':{p.name:sha(p) for p in src.iterdir() if p.suffix in ('.h','.cpp','.py','.txt')},
        'feedback':arm.startswith('feedback'),'diagnostic_logging':diagnostic,'checkpoint_sha256':sha(checkpoint/'net.onnx')})
    return src

def main():
    cfg=json.loads((ROOT/'configs/velocity-memory-v11.json').read_text());ARCHIVE.mkdir()
    all_arms=cfg['arms']+[a+'-final' for a in cfg['arms']]
    runs={a:json.loads((ROOT/'models'/('velocity-v11-'+a)/'run.json').read_text()) for a in all_arms}
    assert len({r['initial_weights_sha256'] for r in runs.values()})==1
    assert len({r['sampling_plan_sha256'] for r in runs.values()})==1
    assert all(r['parameters']==1606 and r['steps']==1520 for r in runs.values())
    for key in ('model_sha256','best_epoch','best_validation'):assert runs['feedback'][key]==runs['feedback-repeat'][key],key
    for key in ('model_sha256','best_epoch','best_validation'):assert runs['feedback-final'][key]==runs['feedback-repeat-final'][key],key
    for arm in all_arms:
        dst=ARCHIVE/arm;dst.mkdir();source=ROOT/'models'/('velocity-v11-'+arm)
        for filename in ('net.onnx','meta.json','run.json','history.json'):shutil.copy2(source/filename,dst/filename)
    for filename in ('velocity-memory-v11.json','long-validation-v11.json'):shutil.copy2(ROOT/'configs'/filename,ARCHIVE/filename)
    shutil.copy2(ROOT/'data/velocity-memory-v11/audit.json',ARCHIVE/'preparation-audit.json')
    artifacts={}
    v10=json.loads((ROOT/'experiments/joint-balance-v10/artifacts.json').read_text())
    for name,key in (('teacher','teacher'),('v10','joint-v10')):
        artifacts[name]=v10[key];assert sha(artifacts[name]['path'])==artifacts[name]['sha256']
    for arm,diagnostic in [('no-history',False),('feedback',False),('no-history-final',False),('feedback-final',False),('feedback-final',True)]:
        name=arm+('-parity' if diagnostic else '');src=source_tree(cfg,arm,diagnostic);build=OUT/('build-'+name)
        run('configure-'+name,['cmake','-S',str(src),'-B',str(build),
            '-DCMAKE_TOOLCHAIN_FILE='+str(ROOT/'workspace/build/toolchain/wasi-sdk/share/cmake/wasi-sdk-p1.cmake'),
            '-DPython3_EXECUTABLE='+sys.executable,'-DVIOBENCH_NATIVE_DIR='+str(ROOT/'workspace/runtime/native'),
            '-DVIOBENCH_OPENCV_DIR='+str(ROOT/'workspace/build/runtime/opencv-build'),'-DVISUAL_DOWNSAMPLE=2','-DVISUAL_SPARSE_DIRECTION=ON'])
        run('build-'+name,['cmake','--build',str(build),'-j','4'])
        wasm=OUT/('AeroVIO-native-v11-'+name+'.wasm');shutil.copy2(build/'solution.wasm',wasm)
        run('check-'+name,[sys.executable,'-m','viobench','check',str(wasm),'--frames','80','--logs',str(OUT/('check-'+name))])
        if not diagnostic:artifacts[arm]={'path':str(wasm),'sha256':sha(wasm),'bytes':wasm.stat().st_size}
        dst=ARCHIVE/('source-'+name);dst.mkdir()
        for p in src.iterdir():shutil.copy2(p,dst/p.name)
    write(ARCHIVE/'artifacts.json',artifacts)
    write(ARCHIVE/'training-verification.json',{'repeat_exact':True,'same_initialization_targets_steps':True,
        'parameters_each':1606,'steps_each':1520,'chosen_epochs':{a:r['best_epoch'] for a,r in runs.items()}})
    print('BUILD COMPLETE',flush=True)
if __name__=='__main__':main()
