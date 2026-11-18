# open-sbsar

An open, clean-room engine for compiled Substance archives (`.sbsar`): a Rust library that opens
`.sbsar` files, exposes their parameters and renders their outputs on the GPU (wgpu/Vulkan) or on
the CPU; a command-line tool; a writer that produces `.sbsar` files; and a bit-level
specification of the format.

Not affiliated with, endorsed by or derived from code of Adobe. "Substance" is a trademark of its
owner; it is used here only to name the file format this project interoperates with.

## Status

Version 0.1: usable, under active development.

- **Accuracy**, measured against Substance 3D Designer 14.1's own renderer (its Vulkan engine) on
  227 real materials at 128², every internal node compared: 88.6% of the nodes are identical
  within one 8-bit code; of 1,571 outputs, 71.2% are identical within 1/255, 14.3% differ by
  less than 0.02 and 14.4% by more (usually a few pixels: by eye, most of them cannot be told
  apart from Substance's output). Most remaining differences are floating-point noise that
  Substance's own CPU and GPU engines also disagree on, amplified by threshold chains.
- **Every node type** of the format is implemented: every node and function instruction in all 1,388
  files of the Substance community library is understood (`examples/scan_ops.rs`); all 1,388 of
  them were rendered in the last full check.
- **GPU**: about 95% of the nodes of real graphs run on the GPU, bit-exact with the CPU path
  (except pixel-processor transcendentals and levels' gamma, which use the GPU's own functions,
  as Substance's GPU engine does). Validated on an NVIDIA RTX 4070 SUPER (Vulkan). A material at
  512² re-renders with new parameters in about 0.5 s once its shaders are compiled.
- **Image inputs** (curvature, ambient occlusion, position, normal maps…) are evaluated: pass
  images with `Parameters::set_image` (CLI: `--set-image IDENT=path`).
- **Pure Rust**: the `.sbsar` archive is read by the crate's own 7z reader; no external tools.
- **Not yet**: WebAssembly/WebGPU.
- **Known limits**: 17 of the 1,388 library files need more GPU memory than a 12 GB card has
  (some of their nodes have fixed sizes up to 8192²); the GPU render then returns an error, and
  the CPU path can render them, slowly (one took 7 minutes at 128²). Malformed archives and programs return errors instead of
  crashing (fuzz-tested), but a hostile file can still make a render take minutes or several GB
  (as large valid files do): run untrusted files with a time and memory limit.

## Library

```toml
[dependencies]
open-sbsar = "0.1"            # default feature "gpu" (wgpu 30); default-features = false: CPU only
```

```rust
use open_sbsar::{Package, Parameters, Renderer, Value};

let package = Package::open("material.sbsar")?;            // or Package::from_bytes(&bytes)
let graph = &package.graphs()[0];
for input in graph.inputs() {                               // identifier, kind, default, range, label…
    println!("{} {:?} {:?}", input.identifier, input.kind, input.default);
}
let mut params = Parameters::new();
params.set("$outputsize", Value::Int(vec![10, 10]));        // log2 sizes: 1024 x 1024
params.set("$randomseed", Value::Int(vec![42]));

let mut renderer = Renderer::new_gpu()?;                    // or Renderer::new_cpu()
for out in renderer.render(graph, &params, None)? {         // None: every image output
    // out.format: U8 / U16 / F16 / F32, the format the graph computes the output in;
    // out.data holds those bytes; out.as_f32(), to_rgba_f32(), to_rgba16(), to_rgba8() convert
    println!("{} {}x{} {} channel(s) {:?}", out.identifier, out.width, out.height, out.channels, out.format);
}
```

- **Your own GPU device**: `Renderer::with_device(device, queue)` renders on the `wgpu::Device`
  and `wgpu::Queue` your application already uses (`open_sbsar::wgpu` is the wgpu version the
  crate is built with, currently 30). Nothing is requested from the device; an error names any
  limit that is too small.
- **Image inputs**: `graph.inputs()` lists them with `kind == InputKind::Image`; connect one with
  `params.set_image("curvature", InputImage { width, height, channels, data, storage })`, where
  `data` holds `width * height * channels` f32 values in 0..1 (1 or 4 channels) and `storage`
  is the source format (`U8`/`U16`/`F16`/`F32`). An image input that is not set uses the
  graph's default (or transparent black).
- **Incremental re-rendering**: an `Instance` re-renders a graph after parameter changes,
  recomputing only the nodes whose result can change; its outputs are byte-identical to a full
  render. `instance.render(&mut renderer, graph, &params, None)`, then
  `instance.last_stats()` gives the nodes recomputed and reused. Results are kept within a
  memory budget (`set_cache_budget`, default 512 MiB; on the GPU it is device memory).
- **Reuse one renderer**: it keeps the GPU context and the compiled shaders between renders; the
  first render of a material compiles its shaders, later ones do not.
- Errors (also internal failures) are returned as `open_sbsar::Error`; `renderer.warnings` lists
  ignored inputs and unsupported features of the last render.
- Examples: `lab/sdrt/examples/render.rs` (all outputs to 16-bit PNGs),
  `lab/sdrt/examples/host_device.rs` (rendering on a host device),
  `lab/sdrt/examples/incremental_check.rs` (incremental renders checked against full ones).

## Command-line tool

```sh
cargo build --release --manifest-path lab/sdrt/Cargo.toml   # target/release/open-sbsar
open-sbsar render FILE.sbsar OUTDIR            # every output as a 16-bit PNG named after its identifier
open-sbsar inputs FILE.sbsar                   # the graph's parameters
open-sbsar node FILE.sbsar INDEX OUT.png       # one internal node (OUT.f32: raw RGBA f32)
open-sbsar stats FILE.sbsar                    # every node: op, size, value ranges
    --set '$outputsize=10,10'            # set a parameter (log2 sizes here: 1024²)
    --set-image IDENT=image.png          # connect an image input (PNG, JPEG, TIFF, EXR)
    --backend vulkan                     # the GPU path (the default is the CPU path)
```

## What is here

| Path | What |
| --- | --- |
| `lab/sdrt/` | the `open-sbsar` crate: library (`src/lib.rs`, `src/api.rs`), engine (program parser, node operations, function VM, FX-Maps, text, bitmaps, CPU and GPU paths) and the `sdrt` tool |
| `lab/sdrt/GPU.md` | the GPU backend: what runs on the GPU and how bit-exactness with the CPU path is kept (a research log) |
| `tools/write.py` | writes `.sbsar` files from scratch (Python 3, standard library; needs `7z`) |
| `docs/` | the format specification, draft 1: [SPEC.md](docs/SPEC.md) (container, program binary, execution model), [SPEC-vm.md](docs/SPEC-vm.md) (functions), [SPEC-ops.md](docs/SPEC-ops.md) (node operations), [SPEC-fxmap.md](docs/SPEC-fxmap.md) (FX-Maps, text, SVG, bitmaps) |

## How it was made

Black-box only. We wrote `.sbsar` files ourselves, rendered them with the official command-line
renderer and compared the images pixel by pixel with our engine, then refined the formulas until
they matched. No decompilation, disassembly, shader or API capture, and no reading of the vendor's
binaries or resources was used at any point.

## Support

This project is independently developed and funded by a student. If you find it useful, consider
supporting its development. Donations help cover AI-assisted research, testing, and
infrastructure costs.

**[Support open-sbsar](https://app.lava.top/wtfmydarling)**

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
See [NOTICE](NOTICE) for the clean-room statement, trademarks and acknowledgements.

Unless you explicitly state otherwise, any contribution you submit for inclusion in this work
shall be dual licensed as above, without any additional terms or conditions.
