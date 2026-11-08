# Vulkan backend — checkpoint 2026-11-08 (transformation port)

This is a working **hybrid** backend, not a completed GPU runtime or a full-compatibility claim.
CPU v28 remains frozen at `scratch/vers/sdrt_v28` and is the specification: GPU kernels must
reproduce it bit for bit. Work and validation stay in research/sbsar.
Previous checkpoint: `scratch/vers/sdrt_gpu_2026-11-07_pp1` (resident buffers + PP shaders).
This checkpoint: `scratch/vers/sdrt_gpu_2026-11-08_xf1` (+ `_src.tar.gz`, `.json`).

## Run

From research/sbsar, use disk TMPDIR, 128² probes and the existing memory scope:

```sh
export TMPDIR="$PWD/scratch/t"
export SDCMP_OUTPUTSIZE=7
systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --expand-environment=no cargo build --offline --release --features gpu --manifest-path lab/sdrt/Cargo.toml
systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --expand-environment=no lab/sdrt/target/release/sdrt node FILE NODE OUT.png --backend vulkan --set '$outputsize=7,7'
```

`--backend cpu` remains the default; CPU-only builds need no GPU feature (`gpu_fma`, the probe
binary, has `required-features = ["gpu"]`). Explicit Vulkan fails if hardware initialization
fails or a software CPU adapter is selected. Dispatch/compilation errors are reported, never
silently converted into CPU execution. Unsupported cases take the CPU path *before* any GPU work.

Diagnostics:
- `SDRT_GPU_LOG=1`: every PP / transformation decision (shader or `CPU fallback (reason)`), PP
  WGSL size, value count and form (`control flow`, with its WGSL lines), PP pipeline creation
  time, and the final dispatch/transfer counts.
- `SDRT_GPU_DUMP=DIR`: writes each generated PP shader to `DIR/pp_<node>.wgsl`, its per-node
  values (`.lits`: buffer word and value id) and the function's instructions (`.ins`).
- `SDRT_GPU_PIN` (research): `1`/`all` embeds every per-node value (each node's text as before
  the value buffer), `none` reads every value from the buffer (results change, see
  `cx/gpu_ppfast/REPORT.md`), `N:all`, `N:+i,j`, `N:-i,j` for PP node N only.
  `SDRT_GPU_FLOW_LIMIT=L` overrides the control-flow size limit (20,000 WGSL lines).
- `sdrt nodes` with `SDRT_NODES_F32=1` writes raw `n<K>.f32` (every bit) instead of 16-bit PNGs.

## Implemented

- Optional wgpu 0.19 Vulkan backend (`gpu.rs`), tested on RTX 4070 SUPER, NVIDIA 615.71.09.
- Resident RGBA f32 buffers keyed by graph node; upload only at CPU→GPU boundaries, read back at
  GPU→CPU boundaries, exports and `stats`/keep-all; last-use release follows the graph.
- Shared WGSL prelude (`gpu_prelude.wgsl`): Config layout, UNORM8/UNORM16/half/f32 storage
  quantization (bit-exact with v28), correctly rounded `unorm_float`, `store`, `image_pixel`
  (two-slot or packed inputs) and `barrier()` (below).
- Atomic kernels (`gpu_atomic.wgsl`): uniform colour (`0x0c`), grayscale / alpha matte and
  channel shuffle (`0x06`), blend (`0x02`, incl. opacity input and crop mask), levels (`0x1e`),
  normal (`0x24`), sharpen (`0x1a`). Same-size inputs only (resizing stays on the CPU).
- Pixel Processor (`0x28`) shader generation (`gpu_pp.rs`, `gpu_pp.wgsl`, `gpu_control.wgsl`):
  see the 2026-11-07 notes below; since then also conditional rand / variable mutation / loops
  through a "control-flow" form, and more than two inputs (packed by GPU-to-GPU copies).
  The control-flow form (2026-11-09, `cx/gpu_ppfast`) has no memoized `eval_*` functions: each
  register component lives in a function-scope variable with a validity flag and is computed
  in one place; the generator knows statically which components are computed at each point and
  emits a runtime check only where that depends on the path. A conditional's operands are
  emitted one after the other, each register under the predicate for which the VM computes it
  (taken branch, or the untaken branch's shared components), so nothing is copied per branch;
  registers without side effects are computed in advance. Per-node values (literals, inputs,
  value-processor values, the variable state after size/seed) are read from `config.lits`
  (words 40.. of binding 0) unless the driver could compute differently with them known
  (division/pow operands, conditions and select values, values combined in one float
  arithmetic tree, 0/1/non-finite values, known float expressions): those stay embedded, so
  every result is unchanged.
- **Transformation (op `0x04`, gray and colour; `gpu_xf.wgsl`, `ops.rs try_gpu_transform`)**: a node runs as a
  chain of passes in one submission (`gpu.rs dispatch_transform`), mirroring each CPU step:
  - `sample`: matrix/offset mapping, nearest (tiled axes wrap, others clamp; outside → matte) or
    1/256 bilinear with the coupled cross weight; UNORM sources interpolate 16-bit codes;
    matte border, or edge clamp for identity + manual level;
  - `halve`: mip levels (in the transformation's format; embedded bitmaps: float) and the
    dihedral path's per-axis halvings (node format); 16-bit targets average integer codes;
  - `upsample`: the dihedral path's per-axis 2x bilinear (source format, unquantized);
  - `permute`: the dihedral path's texel permutation (identity: a copy).
  Auto and manual (incl. function) mip levels, the 1x1 end of a chain, seeded parameter
  functions (evaluated with the node seed, exactly as `exec`), gray/colour in and out.

### Bit-exactness technique: `barrier()`

The NVIDIA compiler contracts `a*b + c` into one fused multiply-add (one rounding instead of
two). Rust never does. `barrier(p)` XORs the product's bits with `config.header.y >> 31` — zero at
run time (pixel count < 2^31) but opaque to the compiler — so the product is rounded before the
addition. Wrap every product that feeds an addition. Measured: with `barrier` as identity, 76 of
183 transformation cases differ (1 ULP); with it, all are bit-exact. It also made the grayscale
kernel exact (`gpu_basic.py` 20/20, the old stage-1 residual is gone). Atomic blend/levels/normal/
sharpen and PP sampling do not use it yet (see residuals).

## Fallbacks and explicit limitations

Transformation stays on the CPU for: a missing input; any `SDRT_XF*`, `SDRT_DSROUND`,
`SDRT_DSINT`, `SDRT_MIPROUND` experiment switch; non-power-of-two sources needing a resize or mip
chain; non-finite matrix/offset or coordinates beyond 2^24 source texels; images over one storage
binding (128 MiB, e.g. 4096² f32 bitmaps); a matte outside [0, 1] that UNORM interpolation could
sample (Rust's saturating casts and codes past 65535); a float 1x1 node sampled as a UNORM mip
level. A fallback after parameter evaluation restores the node variables.

PP: inputs over one storage binding take the CPU path (previously a panic). Control-flow
shaders whose main function exceeds 20,000 WGSL lines take the CPU path: runtime checks of
components computed on some paths copy the code below them, and creation time grows about
quadratically (8,350 lines: 1 s; stripe-new-1 node 158, a chain of whole-value sequences under
such checks, 141,856 lines: 256 s). The old guard (100,000 lines once the driver inlines every
`eval_*` call; chaux node 98: 14.5 M) and the 512-instruction limit of that form are gone.

Known, measured driver behaviour: **NVIDIA 615 flushes subnormal results to zero** in shaders
(WGSL allows it; CPU v28 keeps them). `gpu_transform.py subnormal` documents exactly that
difference. Other drivers may also reassociate additions (WGSL allows it; NVIDIA 615 does not,
`order_cancel`): re-run every suite before trusting another GPU/driver.

Other limits (unchanged): one submission per GPU node (a transformation chain is one), no buffer
pool, no tiling/large images, no 8K batches. Timers include CPU work and transfers.

## Reproducible checks

Run from research/sbsar under the memory scope; `python3 scratch/gpu_xf2/regression.py` runs all
of them (logs in `scratch/gpu_xf2/regress/`). Current CPU must equal v28 byte for byte.

| Script in lab/py | Coverage | Result 2026-11-08 |
| --- | --- | --- |
| `gpu_basic.py` | 20 uniform/gray/matte/shuffle outputs, all formats | 20/20 bit-exact |
| `gpu_resident.py` | 11 residency/boundary cases, 100-node chain | 11/11 exact |
| `gpu_pp.py` (+`SDRT_GPU_DYNAMIC=1`) | 104 PP cases each | pass; native transcendental cases within their tolerances |
| `gpu_control.py` | 108 branch/loop/variable/shared-rand cases | 108/108 exact |
| `gpu_sampling.py` | 32 PP sampling cases | pass; float bilinear 1 ULP (no barrier yet) |
| `gpu_many_inputs.py` | 20 packed-input cases | 20/20 exact |
| `gpu_pp_boundaries.py` | 67 size/disconnected/NaN/Inf UV cases | 67/67 exact |
| `gpu_atomic.py` | 269 blend/levels cases | pass within tolerances (blend 176/241 exact, max 1 UNORM8 code; levels 13/28, pow: relative 3e-6 or one half step) |
| `gpu_neighbors.py` | 96 normal/sharpen cases | pass within tolerances (68/96 exact) |
| `gpu_transform.py` | 1,066 transformation cases | 1,065 bit-exact + 1 documented FTZ case; 1,057 on GPU |
| `gpu_transform_real.py` | real transformation nodes (below) | all isolated nodes bit-exact |
| `gpu_cache.py PREFIX` | real PP nodes vs v28 and the Adobe 14.1 vk cache | unchanged |

`gpu_transform.py` covers: every storage format in and out, gray/colour, all tilings, matte,
random full-mantissa matrices/offsets, auto/manual mips (incl. past the 1x1 end), the dihedral
resize chain (halve + upsample mixes, rotations of non-square sources), seeded function matrices,
GPU→GPU chains, signed zeros, review findings (HDR/negative/huge matte), switches.

`gpu_transform_real.py` dumps transformation nodes and their sources as raw f32 with both
backends; a node counts as isolated when its source is bit-identical (upstream GPU kernels have
their own residuals). 133 cache files (all with cached render ≤ 60 s), ≤ 80 transformations each:
6,510 nodes, 6,501 on the GPU (9 over the binding limit), **4,746 isolated, 4,746 bit-exact**.
Transformations are ~25% of all real nodes (82,657 of 324,070 in the 227-file survey).

Speed: the same sweep took 197 s on CPU and 1,025 s hybrid, 797 s of which was the NVIDIA
compiler creating PP pipelines. Those slow shaders were the memoized control-flow form (test-2-twm:
16 shaders of 88 instructions, 68,756 lines once inlined, 25-28 s each; feathergen: 108
instructions, 88,160 lines, 90-100 s). With the flat form (2026-11-09, cold driver cache):
test-2-twm 489 s -> 17 s, feathergen 844 s -> 7.6 s (CPU: 5.1 s, 2.3 s). The remaining
creation time is large static shaders whose ~100 `sample_image` calls are each inlined
(2,641-2,788 instructions: 4.5-8.9 s).

## Next work

1. PP compile cost: done for the control-flow form (`cx/gpu_ppfast/REPORT.md`). Left: smaller
   sampler code for big static shaders, and a dispatcher or implied-flag tracking for chains of
   whole values under runtime checks (the 20,000-line limit).
2. Apply `barrier()` to the atomic blend/levels/normal/sharpen kernels and PP sampling; residual
   divisions/sqrt/pow need correctly rounded implementations to become exact.
3. Port the next frequent kernels (blur, warp, gradient map, FX-Map, distance...), then command
   batching, buffer reuse and representative timings; optionally `DenormPreserve` via SPIR-V
   passthrough if subnormal parity matters.
4. Then the broad writer/readback round-trip campaign and the full cached survey on the GPU.

No Adobe implementation, shader extraction, decompilation or API capture was used.

---

## 2026-11-07 notes (still valid)

- Pixel Processor scalar-demand shader generation: constants, system variables, interface/VP
  values, typed float/int/bool arithmetic and conversion, vector construction/swizzles,
  comparisons, pure conditionals, dot, lerp, trig/transcendentals, Jenkins pixel hash + LCG rand,
  sequence/write, getvar/setvar and shared scalar-component memoization. Variables snapshot whole
  vectors on the first getvar component demand. Size/seed functions stay on CPU, with their final
  variable state embedded into the shader after seed evaluation. Unsupported functions are
  rejected before seed side effects.
- Sampling: nearest and bilinear; tiled/clamped edges; coupled 1/256 weights; UNORM interpolation
  in 16-bit integer space; manual buffer sampling (no hardware sampler). Sampler NaN/Inf checks
  precede floating-point coordinate operations.
- Signed integer division/modulo edge guards, floor-to-int, saturated float-to-int, measured pow
  domain guards. Functions over 4,096 instructions (both forms since 2026-11-09) or expression depth
  256 stay on CPU. Value processors stay on CPU.
- Real cached comparisons (red-yellow-material, plaid-fabric-01, variable-reading nodes):
  `scratch/gpu_start/real/*/results.json`; summary in the 11-07 checkpoint json.
