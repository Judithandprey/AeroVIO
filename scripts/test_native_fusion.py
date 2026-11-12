"""Run actual C++ geometry tests inside the pinned Wasmtime engine."""
import argparse
from pathlib import Path
import wasmtime
from viobench.tools.runner.engine import Engine


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('wasm',type=Path)
    args=p.parse_args()
    engine=Engine()
    module=engine.module(args.wasm.read_bytes())
    store=wasmtime.Store(engine.wasmtime)
    store.set_fuel(200_000_000)
    store.set_epoch_deadline(engine.ticks(60))
    wasi=wasmtime.WasiConfig(); wasi.inherit_stdout(); wasi.inherit_stderr()
    store.set_wasi(wasi)
    linker=wasmtime.Linker(engine.wasmtime); linker.define_wasi()
    instance=linker.instantiate(store,module)
    exports=instance.exports(store)
    exports['_initialize'](store)
    result=exports['run_tests'](store)
    print('geometry fuel:',200_000_000-store.get_fuel())
    return result


if __name__=='__main__':
    raise SystemExit(main())
