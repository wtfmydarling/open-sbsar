# open-sbsar

An open, clean-room implementation of the runtime behind compiled Substance archives
(`.sbsar`): a reader and renderer for the compiled programs inside them (`.sbsasm`), a
bit-level specification of the format, and a writer that produces `.sbsar` files.

Status: research software, under active development. Not affiliated with, endorsed by or
derived from code of Adobe. "Substance" is a trademark of its owner; it is used here only to
name the file format this project interoperates with.

## How it was made

Black-box only. We wrote `.sbsar` files ourselves, rendered them with the official command-line
renderer and compared the images pixel by pixel with our engine, then refined the formulas until
they matched. No decompilation, disassembly, shader or API capture, and no reading of the vendor's
binaries or resources was used at any point.

## What is here

| Path | What |
| --- | --- |
| `lab/sdrt/` | `sdrt`, the engine (Rust): container and program parser, node operations, the function VM, FX-Maps, text, bitmaps; CPU reference path and an optional Vulkan (wgpu) backend |
| `lab/sdrt/GPU.md` | state of the Vulkan backend: what runs on the GPU, how bit-exactness with the CPU path is kept, limits (a research log: it refers to test scripts that are not part of this repository) |
| `tools/write.py` | writes `.sbsar` files from scratch (Python 3, standard library; needs `7z`) |
| `docs/` | the format specification, draft 1: [SPEC.md](docs/SPEC.md) (container, program binary, execution model), [SPEC-vm.md](docs/SPEC-vm.md) (functions), [SPEC-ops.md](docs/SPEC-ops.md) (node operations), [SPEC-fxmap.md](docs/SPEC-fxmap.md) (FX-Maps, text, SVG, bitmaps) |

## Accuracy

Measured against Substance's own renderer (version 14.1, its Vulkan engine) at 128² on 227
real-world material files (not part of this repository): 281,402 of 324,070 image nodes (86.8%)
identical, and of 1,575 outputs
1,135 within 1/255, 156 within 0.02, 284 further off. Most of the remaining differences are
floating-point noise that Substance's own CPU and GPU engines also disagree on, amplified by
noise/threshold chains; matching its GPU engine exactly is the purpose of the Vulkan backend.

The Vulkan backend (`--features gpu`, `--backend vulkan`) runs about 95% of the nodes of real
graphs on the GPU and reproduces the CPU path bit for bit, except where it deliberately uses the
GPU's own transcendental functions (pixel processors, levels' gamma), which measure closer to
Substance's own GPU engine. Whole files at 1024² render 1.3–56× faster than on the CPU path.

## Build and run

```sh
cargo build --release --manifest-path lab/sdrt/Cargo.toml                  # CPU only
cargo build --release --features gpu --manifest-path lab/sdrt/Cargo.toml   # + Vulkan backend
```

`7z` must be on `PATH` (archives are read with it).

```sh
sdrt render FILE.sbsar OUTDIR            # every output as a 16-bit PNG named after its identifier
sdrt node FILE.sbsar INDEX OUT.png       # one node's image (OUT.f32: raw RGBA f32, every bit)
sdrt nodes FILE.sbsar OUTDIR K...        # several nodes from one run
sdrt inputs FILE.sbsar                   # the graph's parameters
sdrt stats FILE.sbsar                    # every node: op, size, value ranges
    --set '$outputsize=9,9'              # set a parameter (log2 sizes here: 512²)
    --backend vulkan                     # hybrid GPU execution (build with --features gpu)
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
See [NOTICE](NOTICE) for the clean-room statement, trademarks and acknowledgements.

Unless you explicitly state otherwise, any contribution you submit for inclusion in this work
shall be dual licensed as above, without any additional terms or conditions.
