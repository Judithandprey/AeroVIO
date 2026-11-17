"""Run the whole multi-image C++ test batch in Wasmtime, retaining fuel usage.

The old V8 helper's 200M budget covers small geometry tests. These tests include
five 48-frame detector/controller sequences. The 2B budget is for this entire
test batch only; estimator per-frame limits and measured fuel are unchanged.
"""
import argparse
from pathlib import Path
import wasmtime
from viobench.tools.runner.engine import Engine

def main():
    p=argparse.ArgumentParser();p.add_argument('wasm',type=Path);args=p.parse_args()
    engine=Engine();module=engine.module(args.wasm.read_bytes())
    store=wasmtime.Store(engine.wasmtime);budget=2_000_000_000
    store.set_fuel(budget);store.set_epoch_deadline(engine.ticks(60))
    wasi=wasmtime.WasiConfig();wasi.inherit_stdout();wasi.inherit_stderr();store.set_wasi(wasi)
    linker=wasmtime.Linker(engine.wasmtime);linker.define_wasi()
    instance=linker.instantiate(store,module);exports=instance.exports(store)
    exports['_initialize'](store);result=exports['run_tests'](store)
    print('whole multi-image test batch fuel:',budget-store.get_fuel())
    return result
if __name__=='__main__':raise SystemExit(main())
