# open-sbsar draft 1 — node operations

Read the [core specification](SPEC.md) for archive structure, record envelope conventions, numeric types, format conversion, and sampler definitions. Read the [function specification](SPEC-vm.md) for bytecode instruction decoding and execution environments. Read the [FX-Map and graphics specification](SPEC-fxmap.md) for FX-Map tree traversal, embedded TrueType rendering, and SVG command streams.

This document specifies every node operation supported by the compiled program format. In accordance with the project rules, the CPU evaluation path with default configuration is normative. Experimental switches (prefixed with `SDRT_*` in the reference implementation) represent rejected alternatives and are noted only where necessary to prevent implementer divergence.

---

## Conventions and Common Structures

### Record Header and Bitfields

Every node record begins with a 32-bit unsigned little-endian word (`opword`):

| Bits | Field | Meaning |
|---|---|---|
| 0 | `colour` | 0 = grayscale node ($C = 1$ channel); 1 = colour node ($C = 4$ channels RGBA). |
| 1–7 | `opcode` | Node operation code ($0\times 00$ to $0\times 2\text{C}$, even values). |
| 8–11 | `log2w` | Base-2 logarithm of static width: $W = 1 \ll \text{log2w}$. |
| 12–15 | `log2h` | Base-2 logarithm of static height: $H = 1 \ll \text{log2h}$. |
| 16–23 | `B` | Node configuration byte 2: format, seed slot, size function flag. |
| 24–31 | `D` | Node configuration byte 3: tiling flags, format extensions, operation-specific bits. |

In byte `B`:
* Bit 0 (`B & 1`): If set, a size-function offset word follows the seed slot.
* Bits 4–5 (`(B >> 4) & 3`): Storage format for the node output:
  * `0`: UNORM8 (8-bit unsigned normalized, divisor 255.0).
  * `1`: UNORM16 (16-bit unsigned normalized, divisor 65535.0).
  * `2`: Half (16-bit IEEE binary16 floating-point).
  * `3`: Float (32-bit IEEE binary32 floating-point).
* Bits 6–7 (`(B >> 6) & 3`): Slot encoding for the node random seed.

In byte `D`:
* For filter and sampling nodes, bit 0 (`D & 1`) indicates horizontal wrapping (wrap X), and bit 1 (`D & 2`) indicates vertical wrapping (wrap Y). Other node types repurpose these bits as specified in their subsections.

### Parameter Slots

Parameter slots are consumed sequentially according to the two-bit descriptor $S(\text{field}, \text{bit}, n) = (\text{field} \gg \text{bit}) \& 3$:
* `0` (None): Parameter is absent; the operation default is used. Consumes 0 words from the record stream.
* `1` (Const): Parameter is inline constant data; consumes $n$ 32-bit words.
* `2` (Fn): Parameter is driven by a dynamic function; consumes 1 32-bit word specifying the block-relative byte offset of the function.
* `3` (Input): Parameter is an image input reference; consumes 1 32-bit word specifying the index of the source node in the node table.

Boolean parameter slots are exceptions: kind 1 encodes `true` without consuming a literal word; kind 2 consumes 1 word for a function pointer; kinds 0 and 3 encode `false` and consume 0 words.

The common trailing suffix `K` parsed for almost all node records consists of:
1. Seed slot: $S(B, 6, 1)$ (consumes 0, 1, or 1 word).
2. Size function: if $B \& 1 \neq 0$, 1 word specifying the function offset ($S(2, 1)$).

The common pixel size parameter `P` consists of $S(D, 2, 2)$, consuming 0, 2 words, or 1 function pointer.

### Input Resampling and Samplers

Images in memory are stored as row-major RGBA `f32` vectors. Grayscale images replicate channel 0 across RGB with alpha fixed to 1.0: `[g, g, g, 1.0]`.

* `input_at(index, W, H)`: Resolves node `index`. If the input's dimensions $(w_i, h_i)$ match the node's dimensions $(W, H)$, the image buffer is borrowed directly. If dimensions differ, the image is resampled to $(W, H)$ using clamped separable 1/256 bilinear sampling (`resized`) at destination pixel centres $((x + 0.5) / W, (y + 0.5) / H)$.
* Coupled texture sampling (`bilinear_or`): Used when sampling images with specific wrap modes, border colors, and format-dependent tie breaking. The 4 bilinear weights are derived from rounded 1/256 fractions $f_x, f_y$ and cross-term $w_{\text{cross}} = \text{round}(f_x \cdot f_y \cdot 256.0) / 256.0$. For UNORM inputs (formats 0 and 1), texels are converted to 16-bit integer codes, accumulated via integer arithmetic `((sum + 128) >> 8) / 65535.0`, preventing floating-point accumulation drift.
* Disconnected inputs: An index equal to `0xffffffff` is permitted only on opcode $0\times 28$ processors and evaluates as transparent black. Any other disconnected input produces a diagnostic and substitutes a 1×1 `[0.0, 0.0, 0.0, 0.0]` placeholder.

---

## 0x00 — gradient map

Maps scalar driver values from an input image through a multi-stop colour or grayscale gradient transfer function.

### Record Layout and Parameter Slots

```
[opword]
[input_node: u32]
[key_count: u32]
[v8word: u32]           (Present only if program version >= 8 and (D & 2) != 0)
[keys_ptr: u32]         (Block byte offset to gradient keys blob)
[K: seed and size slots]
```

Byte `D` bit assignments depend on the archive format version:
* Version $< 8$: Bit 0 (`D & 1`): stored midpoint fields present; Bit 1 (`D & 2`): float keys format; Bit 2 (`D & 4`): repeat addressing mode.
* Version $\ge 8$: Bit 0 (`D & 1`): stored midpoint fields present; Bit 1 (`D & 2`): `v8word` (smoothness parameter) present; Bit 2 (`D & 4`): float keys format; Bit 3 (`D & 8`): repeat addressing mode.

### Keys Blob Encoding

The blob at `keys_ptr` contains `key_count` contiguous key records:
* **Float keys** (`floats_keys` is true):
  * Position: 1 word `f32`.
  * Values: $C$ words `f32` (4 words for colour, 1 word for gray). For gray, the single value $g$ is expanded to `[g, g, g, 1.0]`.
  * Midpoint: 1 word `f32` if bit 0 (`D & 1`) is set.
  * Total size per key: $((C == 4 \text{ ? } 5 : 2) + (\text{mid} \text{ ? } 1 : 0)) \times 4$ bytes.
* **Packed keys** (`floats_keys` is false):
  * Position: `u16` little-endian, divided by 65535.0.
  * Values: For colour, 4 bytes `u8` (R, G, B, A), each divided by 255.0. For gray, 1 `u16` little-endian, divided by 65535.0 (replicated to RGB with alpha 1.0).
  * Midpoint: `u16` little-endian if bit 0 is set.
  * Total size per key: $(C == 4 \text{ ? } 6 : 4) + (\text{mid} \text{ ? } 2 : 0)$ bytes, padded to 4-byte alignment at the blob end.

Keys are stable-sorted by position using IEEE `f32` total ordering (`total_cmp`). Stored midpoint fields are skipped by the evaluator.

### Sizing and Coordinate Extraction

The driver image is resized to node dimensions $(W, H)$ via `input_at`.
For each pixel, the coordinate $x$ is:
* Colour source: alpha channel `px[3]`.
* Grayscale source: red channel `px[0]`.
* Nonfinite coordinate guard: If $x$ is NaN or infinite, $x = 0.0$.

### Evaluation and LUT Construction

Parameter `v8word` defines smoothness $w$ (default 0.0). Smoothstep interpolation applies when `smooth` is true:
* Version $\ge 8$: `smooth = (D & 1) != 0` (smoothstep when midpoint flag is present; linear when absent).
* Version $< 8$: `smooth = (D & 1) == 0` (smoothstep when midpoint flag is absent; linear when present).

There are two primary LUT evaluation paths:

#### 1. 1024-Texel Float Path (Version $\ge 8$ with `v8word`, or target format half / float)

A 1024-texel unclamped `f32` LUT is constructed:
* If `v8word` is present: Constructed using the cubic Bézier algorithm `build_v8_lut(keys, w)` (see [Appendix A](#a-exact-smooth-gradient-lut-algorithm)).
* If `v8word` is absent: 1024 samples evaluated at $t = i / 1023.0$ ($i \in [0, 1023]$) in `f64` using `lookup64(keys, t, smooth)`, cast to `f32`.

Sampling the 1024-texel table (`sample_f32_lut`):
$$\text{coord} = \begin{cases} \text{rem\_euclid}\left(\text{fma}\left(x, \frac{1023.0}{1024.0}, \frac{0.5}{1024.0}\right), 1.0\right) & \text{if repeat} \\ \text{fma}\left(\text{clamp}(x, 0.0, 1.0), \frac{1023.0}{1024.0}, \frac{0.5}{1024.0}\right) & \text{otherwise} \end{cases}$$
$$u = \text{coord} \cdot 1024.0 - 0.5, \quad i = \lfloor u \rfloor$$
$$\text{wt} = \frac{\lfloor (u - i) \cdot 256.0 + 0.5 \rfloor}{256.0}$$
$$\text{idx}(k) = \begin{cases} \text{rem\_euclid}(k, 1024) & \text{if repeat} \\ \text{clamp}(k, 0, 1023) & \text{otherwise} \end{cases}$$
Let $P_0 = \text{LUT}[\text{idx}(i)]$ and $P_1 = \text{LUT}[\text{idx}(i + 1)]$. If $\text{wt} \le 0.0$, return $P_0$; if $\text{wt} \ge 1.0$, return $P_1$. Otherwise blend in `f64`:
$$\text{out}[c] = \text{f32}(P_0[c] \cdot (1.0 - \text{wt}) + P_1[c] \cdot \text{wt})$$

If the node's target format is half (format 2), each sampled channel is truncated using `half_trunc` before final storage conversion.

#### 2. 256-Texel UNORM16 Path (Target format UNORM8 or UNORM16 without `v8word`)

A 256-texel UNORM16 LUT is built by evaluating `lookup64(keys, i / 255.0, smooth)` in `f64`, multiplying by 65535.0 in `f32`, clamping to $[0.0, 65535.0]$, and rounding via $\lfloor \text{val} + 0.5 \rfloor$ to `u32`.

Sampling uses integer fixed-point blending:
$$\text{coord} = \begin{cases} \text{rem\_euclid}\left(\text{fma}\left(x, \frac{255.0}{256.0}, \frac{0.5}{256.0}\right), 1.0\right) & \text{if repeat} \\ \text{fma}\left(\text{clamp}(x, 0.0, 1.0), \frac{255.0}{256.0}, \frac{0.5}{256.0}\right) & \text{otherwise} \end{cases}$$
$$u = \text{coord} \cdot 256.0 - 0.5$$
* If repeat:
  * If $u < 0.0$: $(i_0, i_1) = (255, 0), \quad f = u + 1.0$
  * If $u \ge 255.0$: $(i_0, i_1) = (255, 0), \quad f = u - 255.0$
  * Otherwise: $i = \lfloor u \rfloor, \quad (i_0, i_1) = (i, i + 1), \quad f = u - i$
* Otherwise (clamp):
  $$i = \min(\max(\lfloor u \rfloor, 0), 254), \quad (i_0, i_1) = (i, i + 1), \quad f = u - i$$
$$\text{wt} = \text{clamp}\left(\lfloor f \cdot 256.0 + 0.5 \rfloor, 0, 256\right)$$
$$\text{out}[c] = \frac{(T[i_0][c] \cdot (256 - \text{wt}) + T[i_1][c] \cdot \text{wt} + 128) \gg 8}{65535.0}$$

### Edge Cases and Defaults

* Zero keys: Returns `[0.0, 0.0, 0.0, 1.0]` everywhere.
* Single key: Returns that key's color across the entire domain.
* Repeated positions: If an interval has $p_1 \le p_0$, evaluation uses $t = 1.0$.

[References: `ops.rs::gradient_map`, `gradient_keys`, `gradient_lut_f32`, `gradient_lut16`, `sample_f32_lut`, `lookup64`, `build_v8_lut`; `program.rs::parse_node`.]

---

## 0x02 — blend

Combines background and foreground images using standard compositing formulas, opacity weighting, and an optional bounding crop box.

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]
[background: u32]       (Background node index)
[foreground: u32]       (Foreground node index)
[K: seed and size slots]
[opacity: S(flags, 4, 1)]
[opacitymult: S(flags, 9, 1)] (Present only if opacity slot kind == 3, i.e. an input image)
[mask: 2 words]         (Present only if flags & 0x40 != 0)
```

Bit allocations in `flags`:
* Bits 0–3: Blend mode ($0 \dots 11$).
* Bit 6 (`0x40`): Crop rectangle enabled.
* Bits 11–12: Alpha compositing mode ($0 \dots 3$).

### Inputs and Sizing

Both background (input 0) and foreground (input 1) images are resized to the node's dimensions $(W, H)$ via `input_at`. If opacity is driven by an image (slot kind 3), that image is also resized to $(W, H)$ via `input_at`.

### Parameters and Crop Mask

1. Opacity: Default 1.0. If driven by an image, opacity multiplier (`opacitymult`) defaults to 1.0. Per-pixel effective opacity:
   $$\text{op} = \text{clamp}\left(\text{op\_const} \cdot (\text{op\_img} \text{ ? } \text{img\_px}[0] : 1.0), 0.0, 1.0\right)$$
2. Crop Rectangle (`mask`): Encoded as two words `[v0, v1]`:
   $$x_0 = \lfloor (v_0 \& 0\text{xFFFF}) / 32768.0 \cdot W \rfloor, \quad y_0 = \lfloor (v_0 \gg 16) / 32768.0 \cdot H \rfloor$$
   $$x_1 = \lfloor (v_1 \& 0\text{xFFFF}) / 32768.0 \cdot W \rfloor, \quad y_1 = \lfloor (v_1 \gg 16) / 32768.0 \cdot H \rfloor$$
   For pixels where $x < x_0 \lor x \ge x_1 \lor y < y_0 \lor y \ge y_1$, blending is bypassed and the background pixel $P_b$ is retained directly.

### Blend Formulas

The mode function $g(b, f)$ operates per channel:

| Mode | Name | Formula $g(b, f)$ |
|---:|---|---|
| 0, 7 | Copy / Switch | $f$ |
| 1 | Add | $b + f$ |
| 2 | Subtract | $b - f$ |
| 3 | Multiply | $b \cdot f$ |
| 4 | AddSub | $b + 2.0 \cdot f - 1.0$ |
| 5 | Max | $\max(b, f)$ |
| 6 | Min | $\min(b, f)$ |
| 8 | Divide | $d = b / f$; if $d$ is NaN, return 1.0; else return $d$ |
| 9 | Overlay | $b < 0.5 \text{ ? } 2.0 \cdot b \cdot f : 1.0 - 2.0 \cdot (1.0 - b) \cdot (1.0 - f)$ |
| 10 | Screen | $1.0 - (1.0 - b) \cdot (1.0 - f)$ |
| 11 | Soft Light | $f < 0.5 \text{ ? } 2.0 \cdot b \cdot f + b^2 \cdot (1.0 - 2.0 \cdot f) : 2.0 \cdot b \cdot (1.0 - f) + \sqrt{b} \cdot (2.0 \cdot f - 1.0)$ |
| other | Fallback | $f$ |

Linear interpolation evaluates with separate products: $\text{mix}(a, b, t) = a \cdot (1.0 - t) + b \cdot t$.

### Compositing Rules

Let $b_a = P_b[3]$, $f_a = P_f[3]$, and $o = \text{op}$.

* **Grayscale Nodes ($C = 1$):**
  $$\text{out} = \text{Image::gray}(\text{mix}(P_b[0], g(P_b[0], P_f[0]), o))$$
* **Colour Nodes ($C = 4$):**
  * If Mode == 7 (Switch): Mixes all 4 channels identically:
    $$C[ch] = \text{mix}(P_b[ch], P_f[ch], o) \quad \text{for } ch \in [0, 3]$$
  * Otherwise, composite according to alpha mode:
    * **Alpha Mode 1 (Ignore Alpha):**
      $$C[ch] = \text{mix}(P_b[ch], g(P_b[ch], P_f[ch]), o) \quad \text{for } ch \in [0, 2]$$
      $$C[3] = b_a + o \cdot f_a \cdot (1.0 - b_a)$$
    * **Alpha Mode 2 (Straight / W3C Compositing):**
      $$a_{\text{full}} = f_a + b_a \cdot (1.0 - f_a), \quad a = b_a \cdot (1.0 - o) + a_{\text{full}} \cdot o$$
      $$\text{mixed}[ch] = g(P_b[ch], P_f[ch]) \cdot f_a \cdot b_a + P_f[ch] \cdot f_a \cdot (1.0 - b_a) + P_b[ch] \cdot b_a \cdot (1.0 - f_a)$$
      $$\text{pre}[ch] = P_b[ch] \cdot b_a \cdot (1.0 - o) + \text{mixed}[ch] \cdot o$$
      $$C[ch] = a \neq 0.0 \text{ ? } (\text{pre}[ch] / a) : 0.0 \quad \text{for } ch \in [0, 2], \quad C[3] = a$$
    * **Alpha Mode 3 with Mode == 0 (Premultiplied Copy):**
      $$C[ch] = P_f[ch] \cdot o + P_b[ch] \cdot (1.0 - f_a \cdot o) \quad \text{for } ch \in [0, 2]$$
      $$C[3] = f_a \cdot o + b_a \cdot (1.0 - f_a \cdot o)$$
    * **Alpha Mode 0 (and default fallback):**
      $$C[ch] = \text{mix}(P_b[ch], g(P_b[ch], P_f[ch]), f_a \cdot o) \quad \text{for } ch \in [0, 2]$$
      $$C[3] = b_a + o \cdot f_a \cdot (1.0 - b_a)$$

[References: `ops.rs::blend`; `program.rs::parse_node`.]

---

## 0x04 — transformation

Applies 2D affine transformations, manual or automatic mip selection, and tiling/border handling.

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]
[input: u32]            (Source node index)
[K: seed and size slots]
[miplevel: S(2, 1)]     (Present only if (flags & 0x1f) == 0x1e)
[matrix: S(flags, 6, 4)]
[offset: S(flags, 25, 2)]
[mattecolor: S(flags, 28, 4)]
```

Bit allocations in `flags`:
* Bits 0–4: Mip level selection mode: `0x1f` automatic LOD; `0x1e` driven by dynamic function `miplevel`; $0 \dots 29$ manual constant level.
* Bit 5 (`0x20`): Filtering mode: 1 = bilinear; 0 = nearest.
* Bits 6–7: Matrix slot kind.
* Bits 25–26: Offset slot kind.
* Bits 28–29: Matte color slot kind.

Byte `D` bits: Bit 0 wrap X, bit 1 wrap Y.

### Parameter Evaluation and Defaults

Evaluate in order: `matrix` (default `[1.0, 0.0, 0.0, 1.0]`), `offset` (default `[0.0, 0.0]`), `mattecolor` (default `[0.0, 0.0, 0.0, 0.0]`), then `miplevel` if present (default 0.0).

### Coordinate Mapping

Destination pixel coordinates $(x, y) \in [0, W-1] \times [0, H-1]$ map to source UV coordinates via column-major affine mapping:
$$dx = \frac{x + 0.5}{W} - 0.5, \quad dy = \frac{y + 0.5}{H} - 0.5$$
$$u = m_0 \cdot dx + m_2 \cdot dy + 0.5 + \text{offset}_x$$
$$v = m_1 \cdot dx + m_3 \cdot dy + 0.5 + \text{offset}_y$$

### Fast Dihedral / Identity Path

When auto-mip (`flags & 0x1f == 0x1f`) and bilinear filtering are selected with zero offset, and the matrix is a signed permutation matrix ($[\pm 1, 0, 0, \pm 1]$ or $[0, \pm 1, \pm 1, 0]$), the node executes an exact multi-resolution resize pass:
1. Target dimensions $(tw, th)$ are $(H, W)$ if $m_0 == 0.0$, else $(W, H)$.
2. Halve oversized source axes down to $(tw, th)$ using pair averaging stored in node format $n.\text{format}()$.
3. Double undersized axes up to $(tw, th)$ using coupled texture sampling in the original source format.
4. If the matrix is identity, return the resized buffer directly. Otherwise, flip/transpose texels according to matrix signs.

### General Mip Generation and Sampling

If the fast dihedral path does not apply:
1. Differential footprints:
   $$du = \sqrt{\left(\frac{m_0 \cdot sw}{W}\right)^2 + \left(\frac{m_2 \cdot sw}{H}\right)^2}, \quad dv = \sqrt{\left(\frac{m_1 \cdot sh}{W}\right)^2 + \left(\frac{m_3 \cdot sh}{H}\right)^2}$$
   $$\text{lod}_{\text{raw}} = \log_2(\max(du, dv))$$
2. Level selection:
   $$\text{level} = \begin{cases} \lfloor \max(\text{manual}, 0.0) + 0.5 \rfloor & \text{if manual mip} \\ \lfloor \max(\text{lod}_{\text{raw}}, 0.0) + 0.5 + 1.0 / 2048.0 \rfloor & \text{if auto mip} \end{cases}$$
3. Mip levels $\ge 1$ are generated in node format $n.\text{format}()$ (except opcode $0\times 20$ bitmap sources, which use float format 3). Level 0 is the unscaled source image. A 1×1 image acts as its own smallest level.
4. Border and matte:
   * Identity with manual mip clamps borders to the edge texel.
   * Otherwise, texels outside $[0, 1]$ on non-wrapped axes take matte color (`[matte[0..3]]` for colour, `Image::gray(matte[0])` for gray).
   * Bilinear sampling past a non-tiled edge blends with the matte border using coupled weights and UNORM 16-bit integer accumulation (`bilinear_or`). Nearest returns the matte color if non-wrapped coordinates fall outside $[0.0, 1.0]$.

[References: `ops.rs::transformation`, `mip_levels_fmt`; `image.rs::bilinear_or`, `downsample_axes`, `upsample_axes_fmt`.]

---

## 0x06 — grayscale conversion / channel shuffle

Converts colour images to grayscale using weighted channel combination, or shuffles channels between two colour sources.

### Record Layout and Parameter Slots

* **Grayscale Conversion (`colour == 0`):**
  ```
  [opword]
  [input: u32]
  [K: seed and size slots]
  [weights: S(D, 0, 4)]
  [mattelevel: S(D, 2, 1)]
  ```
  Byte `D` bit 4 (`D & 0x10`): Matte flattening enable.
* **Channel Shuffle (`colour == 1`):**
  ```
  [opword]
  [flags: u32]
  [input0: u32]
  [input1: u32]
  [K: seed and size slots]
  ```

### Grayscale Conversion Execution

Input 0 is resized to $(W, H)$ via `input_at`.
1. Evaluate parameters: `weights` (default `[1/3, 1/3, 1/3, 0]`), `mattelevel` (default 1.0).
2. Per-pixel luminance:
   $$g = P[0] \cdot w_0 + P[1] \cdot w_1 + P[2] \cdot w_2 + P[3] \cdot w_3$$
3. If $D \& 0\times 10 \neq 0$:
   $$g = \text{matte} \cdot (1.0 - P[3]) + g \cdot P[3]$$
4. Output: Store `Image::gray(g)`.

### Channel Shuffle Execution

Both inputs are resized to $(W, H)$ via `input_at`.
`flags` is unpacked into 4 bytes: `sel = flags.to_le_bytes()`.
For each output channel $c \in [0, 3]$:
$$s = \text{sel}[c] \text{ as usize}$$
$$\text{out}[c] = \begin{cases} P_a[s] & \text{if } s < 4 \\ P_b[(s - 4) \& 3] & \text{otherwise} \end{cases}$$

[References: `ops.rs::exec`; `program.rs::parse_node`.]

---

## 0x08 — FX-Map

Evaluates procedural pattern iteration and quadrant hierarchies. Complete details on pattern rendering, state vectors, random streams, and compositing are specified in [SPEC-fxmap.md](SPEC-fxmap.md).

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]
[tree_ptr: u32]
[inputs: u32 * ((flags >> 10) & 0x3f)]
[K: seed and size slots]
[background: input index if (flags & 3) == 3, else S(flags, 0, C)]
[hurst: S(flags, 2, 1)]
[opacitymult: S(flags, 4, 1)]
[renderroi: S(flags, 6, 4)]
[tilingroi: S(flags, 8, 4)]
```

[References: `fxmap.rs::render`; `program.rs::parse_node`.]

---

## 0x0a — SVG command stream

Renders vector paths, stroked polylines, and filled triangle strips from a binary command stream into an image buffer. Detailed command structures, triangle rasterization, and stroke rules are specified in [SPEC-fxmap.md](SPEC-fxmap.md).

### Record Layout and Parameter Slots

```
[opword]
[stream_ptr: u32]
[background_input: u32] (Present only if D & 4 != 0)
[K: seed and size slots]
[color: C words]        (Present only if D & 2 != 0)
```

[References: `ops.rs::svg`; `program.rs::parse_node`.]

---

## 0x0c — uniform

Fills the entire node extent with a constant or function-generated colour or grayscale value.

### Record Layout and Parameter Slots

```
[opword]
[K: seed and size slots]
[color: S(D, 0, C)]
```

### Execution

1. Evaluate parameter `color` with the node's seed. Default for colour is `[0.0, 0.0, 0.0, 1.0]`; default for grayscale is `0.0`.
2. Scalar function values broadcast across all 4 channels during parameter evaluation.
3. Every pixel in the $W \times H$ buffer is initialized to this colour.
4. Output undergoes node storage conversion $n.\text{format}()$.

[References: `ops.rs::exec`; `program.rs::parse_node`.]

---

## 0x0e — warp

Displaces the source image along the local backward-difference gradient of a secondary driver image.

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]            (Present only if program version >= 9)
[source: u32]
[gradient: u32]
[K: seed and size slots]
[P: pixelsize S(D, 2, 2)]
[intensity: S(D, 5, 1)]
```

Byte `D` bits: Bit 0 wrap X, bit 1 wrap Y.

### Inputs and Sizing

Both source and gradient retain their native dimensions $(sw, sh)$ and $(gw, gh)$.

### Evaluation and Math

1. Evaluate parameters: `intensity` (default 1.0), `pixelsize` (default `(1.0, 1.0)`).
2. Gradient value extraction: $gv = P_{\text{grad}}[3]$ for colour, $gv = P_{\text{grad}}[0]$ for gray.
3. Backward difference:
   * If gradient dimensions match node dimensions ($gw == W \land gh == H$):
     Direct texel fetches:
     $$dx = gv(x, y) - gv(x - 1, y), \quad dy = gv(x, y) - gv(x, y - 1)$$
   * If gradient dimensions differ:
     Coupled bilinear samples in gradient format `gfmt` at destination centre $(u_0, v_0)$:
     $$c = gs(u_0, v_0)$$
     $$dx = c - gs(u_0 - 1.0 / gw, v_0), \quad dy = c - gs(u_0, v_0 - 1.0 / gh)$$
4. Displacement calculation:
   $$kx = \text{intensity} \cdot \frac{gw}{256.0} \cdot \text{ps}_x, \quad ky = \text{intensity} \cdot \frac{gh}{256.0} \cdot \text{ps}_y$$
   $$du = kx \cdot dx, \quad dv = ky \cdot dy$$
5. Source sampling:
   Sample the native source at $(u_0 + du, v_0 + dv)$ using coupled texture sampling `bilinear_or` in source storage format `fmt`, wrapping or clamping per byte `D`.

[References: `ops.rs::warp`; `program.rs::parse_node`.]

---

## 0x10 — emboss

Simulates directional surface shading from a height map using backward differences, illumination vectors, and highlight/shadow coloration.

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]
[source: u32]
[height: u32]
[K: seed and size slots]
[P: pixelsize S(D, 2, 2)]
[intensity: S(flags, 1, 1)]
[lightangle: S(flags, 3, 1)]
[highlight: S(flags, 5, 4)]
[shadow: S(flags, 7, 4)]
```

Byte `D` bits: Bit 0 wrap X, bit 1 wrap Y.

### Inputs and Sizing

Source is resized to $(W, H)$ via `input_at`. Height retains native dimensions $(hw, hh)$.

### Evaluation and Math

1. Evaluate parameters: `intensity` (default 5.0), `lightangle` in turns (default 0.0), `highlight` (default `[1.0, 1.0, 1.0, 1.0]`), `shadow` (default `[0.0, 0.0, 0.0, 1.0]`), `pixelsize` (default `(1.0, 1.0)`).
2. Angle: $\text{ang} = \text{lightangle} \cdot 2\pi$.
3. Height value: $hv = P_{\text{hgt}}[3]$ for colour, $hv = P_{\text{hgt}}[0]$ for gray.
4. Backward differences $(gx, gy)$ evaluate identically to warp (direct fetch if dimensions match; coupled bilinear if dimensions differ).
5. Shading slope:
   $$kx = \text{intensity} \cdot \frac{hw}{256.0} \cdot \text{ps}_x, \quad ky = \text{intensity} \cdot \frac{hh}{256.0} \cdot \text{ps}_y$$
   $$s = kx \cdot gx \cdot \cos(\text{ang}) + ky \cdot gy \cdot \sin(\text{ang})$$
6. Color application:
   For each channel $ch \in [0, 2]$:
   $$\text{col}[ch] = s > 0.0 \text{ ? } \text{highlight}[ch] : (1.0 - \text{shadow}[ch])$$
   $$Q[ch] = P_{\text{src}}[ch] + s \cdot \text{col}[ch]$$
   Grayscale nodes use channel 0 of highlight/shadow. Source alpha is preserved unchanged.

[References: `ops.rs::emboss`; `program.rs::parse_node`.]

---

## 0x12 — legacy motion blur

Performs fixed 6-tap directional motion blur driven by the gradient of a secondary driver image.

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]
[source: u32]
[gradient: u32]
[K: seed and size slots]
[P: pixelsize S(D, 2, 2)]
[intensity: S(flags, 0, 1)]
[shiftangle: S(flags, 2, 1)]
```

Byte `D` bits: Bit 0 wrap X, bit 1 wrap Y.

### Inputs and Sizing

Both source and gradient retain native dimensions. If the gradient input index is absent or invalid, the operation returns the source resized to $(W, H)$ prior to parameter evaluation.

### Evaluation and Math

1. Evaluate parameters: `intensity` (default 1.0), `shiftangle` in turns (default 0.0), `pixelsize` (default `(1.0, 1.0)`).
2. Angle: $\text{shift} = \text{shiftangle} \cdot 2\pi$.
3. Backward differences $(dx, dy)$ evaluate identically to warp.
4. Displacement vector:
   $$gx = dx \cdot gw \cdot \text{ps}_x, \quad gy = dy \cdot gh \cdot \text{ps}_y$$
   $$\text{scale} = \text{intensity} \cdot \frac{1.005}{128.0}$$
   $$ux = (gx \cdot \cos(\text{shift}) - gy \cdot \sin(\text{shift})) \cdot \text{scale}$$
   $$uy = (gx \cdot \sin(\text{shift}) + gy \cdot \cos(\text{shift})) \cdot \text{scale}$$
5. 6-Tap Accumulation:
   Evaluate $i \in [0, 5]$ with $t = (i + 0.5) / 6.0 - 0.5$:
   Sample the native source at $(u_0 + t \cdot ux, v_0 + t \cdot uy)$ via coupled texture sampling in source format `sfmt`.
   Accumulate samples and divide by 6.0. Grayscale outputs replicate channel 0 with alpha 1.0.

[References: `ops.rs::legacy_motion_blur`; `program.rs::parse_node`.]

---

## 0x14 — blur

Executes a two-pass separable box filter with GPU-texture-unit tap pairing and sub-pixel fractional rounding.

### Record Layout and Parameter Slots

```
[opword]
[source: u32]
[K: seed and size slots]
[P: pixelsize S(D, 2, 2)]
[intensity: S(D, 4, 1)]
```

Byte `D` bits: Bit 0 wrap X, bit 1 wrap Y.

### Inputs and Sizing

If native source dimensions $(sw, sh)$ differ from $(W, H)$ and both $sw \ge W$ and $sh \ge H$, the horizontal pass fetches from the native source directly at node-pixel offsets; the vertical pass reads the intermediate buffer. Otherwise, the source is resized to $(W, H)$ via `input_at` before filtering.

### Evaluation and Math

1. Evaluate parameters: `intensity` (default 10.0), `pixelsize` (default `(1.0, 1.0)`).
2. Radii:
   $$r_x = \text{intensity} \cdot \frac{W}{256.0} \cdot \text{ps}_x, \quad r_y = \text{intensity} \cdot \frac{H}{256.0} \cdot \text{ps}_y$$
3. Box taps for radius $r$:
   If $r \le 0.5$, single tap at offset 0 with weight 1.0.
   Otherwise, let $k = \lceil r + 0.5 \rceil$. For $t \in [-k, k]$:
   $$\text{overlap} = \frac{\max(0.0, \min(t + 0.5, r) - \max(t - 0.5, -r))}{2.0 \cdot r}$$
4. Parity-dependent paired taps (`blur_taps`):
   * If $k == 2$: Pair $(1, 2)$ and $(-1, -2)$.
   * If $k > 2$: Pair consecutive offsets $(t, t + 1)$ when $(\text{parity} + t) \text{ rem\_euclid } 2 == 0$.
   * For each pair with weights $w_a, w_b$:
     $$s = w_a + w_b, \quad q = \frac{\text{round}(w_b / s \cdot 256.0)}{256.0}$$
     Replace weights with $s \cdot (1.0 - q)$ at offset $a$, and $s \cdot q$ at offset $b$.
5. Two-pass filtering:
   Horizontal box pass produces an `f32` intermediate buffer. Vertical box pass filters the intermediate into the destination buffer. Both passes wrap or clamp per byte `D`.

[References: `ops.rs::blur`, `box_taps`, `blur_taps`, `blur_fetches`, `box_pass_parity`; `program.rs::parse_node`.]

---

## 0x16 — directional motion blur

Applies linear motion blur along an arbitrary angle using dominant-axis shearing or direct multi-tap sampling.

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]
[source: u32]
[K: seed and size slots]
[P: pixelsize S(D, 2, 2)]
[intensity: S(flags, 0, 1)]
[angle: S(flags, 2, 1)]
```

Byte `D` bits: Bit 0 wrap X, bit 1 wrap Y.

### Inputs and Sizing

Source image is resized to $(W, H)$ via `input_at`.

### Evaluation and Math

1. Evaluate parameters: `intensity` (default 10.0), `angle` in turns (default 0.0), `pixelsize` (default `(1.0, 1.0)`).
2. Major axis determination:
   $$r = \text{intensity} \cdot \frac{W}{256.0}, \quad dx = \cos(\text{angle} \cdot 2\pi) \cdot \text{ps}_x, \quad dy = \sin(\text{angle} \cdot 2\pi) \cdot \text{ps}_y$$
   Major axis is Y if $|dy| \ge |dx|$ (ties choose Y); otherwise X. Let $\text{major} = \max(|dx|, |dy|)$. If $\text{major} == 0.0$, return source.
   $$\text{radius} = 0.5 + (r - 0.5) \cdot \text{major}$$
   Taps are derived from `box_taps(radius)`.
3. Small / Axis-Aligned Kernel:
   If $\text{taps}.\text{len}() \le 3 \lor \min(|dx|, |dy|) < 1\text{e-}6$:
   Sample directly at destination centre plus $(t \cdot dx / \text{major}, t \cdot dy / \text{major})$. Separable bilinear sampling applies for axis-aligned angles; separable 1/256 (`bilinear_256`) applies for off-axis angles.
4. Large Off-Axis Sheared Kernel (Y-major shown):
   * Shear displacement: $sh(y) = (y + 0.5 - H \cdot 0.5) \cdot dx / dy$.
   * If X is untiled, pad intermediate by $ep = \lceil H \cdot 0.5 \cdot |dx / dy| \rceil + 2$. Intermediate dimensions are $(W + 2 \cdot ep, H)$.
   * Sheared pass: For each intermediate pixel, sample source at unwrapped continuing position:
     $$u = \frac{x - ep + 0.5 + sh(y + t)}{W}, \quad v = \frac{y + t + 0.5}{H}$$
     using coupled texture sampling in source format `fmt`, accumulate weighted taps.
   * Unshear pass: Sample blurred intermediate at:
     $$u_{\text{unshear}} = \frac{x + ep + 0.5 - sh(y)}{W + 2 \cdot ep}, \quad v_{\text{unshear}} = \frac{y + 0.5}{H}$$

[References: `ops.rs::motion_blur`; `program.rs::parse_node`.]

---

## 0x18 — directional warp

Displaces the source image along a uniform angular direction weighted by a scalar amount driver image.

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]
[source: u32]
[amount: u32]
[K: seed and size slots]
[P: pixelsize S(D, 2, 2)]
[intensity: S(flags, 1, 1)]
[angle: S(flags, 3, 1)]
[v9third: S(flags, 7, 1)] (Amount offset parameter)
```

Byte `D` bits: Bit 0 wrap X, bit 1 wrap Y.

### Inputs and Sizing

Source retains native dimensions. Amount image is resized to $(W, H)$ via `input_at`.

### Evaluation and Math

1. Evaluate parameters: `intensity` (default 10.0), `angle` in turns (default 0.0), `pixelsize` (default `(1.0, 1.0)`), `v9third` (default 0.0).
2. Scalar driver: $a = P_{\text{amount}}[3]$ for colour, $P_{\text{amount}}[0]$ for gray.
   $$m = a - \text{v9third}$$
3. Displacement vector:
   $$\text{ang} = \text{angle} \cdot 2\pi$$
   $$dx = \cos(\text{ang}) \cdot \frac{\text{intensity}}{256.0} \cdot \text{ps}_x, \quad dy = \sin(\text{ang}) \cdot \frac{\text{intensity}}{256.0} \cdot \text{ps}_y$$
4. Source sampling:
   Sample native source at $(u + dx \cdot m, v + dy \cdot m)$ using coupled texture sampling in source format `fmt`, wrapping or clamping per byte `D`.

[References: `ops.rs::directional_warp`; `program.rs::parse_node`.]

---

## 0x1a — sharpen

Accentuates high-frequency edges using a 5-tap Laplacian kernel weighted by anisotropic pixel sizes.

### Record Layout and Parameter Slots

```
[opword]
[source: u32]
[K: seed and size slots]
[P: pixelsize S(D, 2, 2)]
[intensity: S(D, 4, 1)]
```

Byte `D` bits: Bit 0 wrap X, bit 1 wrap Y.

### Inputs and Sizing

If native source dimensions differ from $(W, H)$, neighbors are sampled at native texel spacing $(1/sw, 1/sh)$ using coupled bilinear sampling; otherwise, neighbors are fetched directly from the resized buffer.

### Evaluation and Math

1. Evaluate parameters: `intensity` (default 1.0), `pixelsize` (default `(1.0, 1.0)`).
2. Second differences per channel:
   $$Q[ch] = C[ch] + \text{intensity} \cdot \left((2.0 \cdot C[ch] - L[ch] - R[ch]) \cdot \text{ps}_x + (2.0 \cdot C[ch] - U[ch] - D[ch]) \cdot \text{ps}_y\right)$$
3. Grayscale nodes store only channel 0. Wrap/clamp follows byte `D`.

[References: `ops.rs::sharpen`; `program.rs::parse_node`.]

---

## 0x1c — HSL

Modifies hue, saturation, and luminosity in colour images using branchless GPU-matched color space transformations.

### Record Layout and Parameter Slots

```
[opword]
[source: u32]
[K: seed and size slots]
[hue: S(D, 0, 1)]
[saturation: S(D, 2, 1)]
[luminosity: S(D, 4, 1)]
```

### Inputs and Sizing

Colour source image is resized to $(W, H)$ via `input_at`.

### Evaluation and Math

1. Evaluate parameters: `hue`, `saturation`, `luminosity`, each defaulting to 0.5.
2. RGB to HSL:
   $$mx = \max(R, G, B), \quad mn = \min(R, G, B), \quad l = (mx + mn) / 2.0$$
   If $mx == mn$, $h = 0.0$ and $s = 0.0$. Otherwise:
   $$d = mx - mn, \quad \varepsilon = 1.0 / 4096.0$$
   $$s = \frac{d}{1.0 - |2.0 \cdot l - 1.0| + \varepsilon}$$
   $$\text{raw} = \begin{cases} (G - B) + (G < B \text{ ? } 6.0 \cdot d : 0.0) & \text{if } mx == R \\ (B - R) + 2.0 \cdot d & \text{if } mx == G \\ (R - G) + 4.0 \cdot d & \text{otherwise} \end{cases}$$
   $$\text{bias} = \begin{cases} -4.0 \cdot \varepsilon & \text{if } \text{raw} == 0.0 \\ -2.0 \cdot \varepsilon & \text{if } \text{raw} < 2.0 \cdot d \\ 0.0 & \text{if } \text{raw} \le 4.0 \cdot d \\ 2.0 \cdot \varepsilon & \text{otherwise} \end{cases}$$
   $$h = \text{rem\_euclid}\left(\frac{\text{raw} + \text{bias}}{6.0 \cdot (d + \varepsilon)}, 1.0\right)$$
3. Parameter adjustments:
   $$hh = \text{rem\_euclid}(h + 2.0 \cdot (\text{hue} - 0.5), 1.0)$$
   $$s = \text{clamp}(s + 2.0 \cdot (\text{saturation} - 0.5), 0.0, 1.0)$$
   $$l = \text{clamp}(l + 2.0 \cdot (\text{luminosity} - 0.5), 0.0, 1.0)$$
4. HSL to RGB:
   If $s == 0.0$, return $(l, l, l)$. Otherwise:
   $$q = l < 0.5 \text{ ? } l \cdot (1.0 + s) : l + s - l \cdot s, \quad p = 2.0 \cdot l - q$$
   Define helper $f(t)$ for $t = \text{rem\_euclid}(t, 1.0)$:
   $$f(t) = \begin{cases} p + (q - p) \cdot 6.0 \cdot t & \text{if } t < 1/6 \\ q & \text{if } t < 1/2 \\ p + (q - p) \cdot (2/3 - t) \cdot 6.0 & \text{if } t < 2/3 \\ p & \text{otherwise} \end{cases}$$
   $$R = f(hh + 1/3), \quad G = f(hh), \quad B = f(hh - 1/3)$$
   Alpha is preserved unchanged.

[References: `ops.rs::hsl`, `rgb_to_hsl`, `hsl_to_rgb`; `program.rs::parse_node`.]

---

## 0x1e — levels

Performs per-channel histogram remapping with black point, white point, and midtone gamma curve adjustments.

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]
[source: u32]
[K: seed and size slots]
[inlow: S(flags, 0, C)]
[inhigh: S(flags, 2, C)]
[inmid: S(flags, 4, C)]
[outlow: S(flags, 6, C)]
[outhigh: S(flags, 8, C)]
```

Bit allocations in `flags`:
* Bit 10 (`0x400`): Range clamp bypass. When set, intermediate and output clamps to $[0.0, 1.0]$ are disabled.

### Inputs and Sizing

Source image is resized to $(W, H)$ via `input_at`.

### Evaluation and Math

1. Evaluate parameters in order: `inlow` (default 0.0), `inhigh` (default 1.0), `inmid` (default 0.5), `outlow` (default 0.0), `outhigh` (default 1.0).
2. **Every parameter vector is unconditionally clamped to $[0.0, 1.0]$**, regardless of the flag clamp bit.
3. For each channel $c \in [0, C-1]$:
   * Effective input span:
     $$d = \text{inhigh}[c] \neq \text{inlow}[c] \text{ ? } (\text{inhigh}[c] - \text{inlow}[c]) : 1\text{e-}4$$
   * Normalization:
     $$x = (P[c] - \text{inlow}[c]) / d$$
     If `flags & 0x400 == 0`, $x = \text{clamp}(x, 0.0, 1.0)$.
   * Midtone gamma:
     Let $m = \text{inmid}[c]$.
     $$e = m \ge 0.5 \text{ ? } 1.0 + 16.0 \cdot (m - 0.5) : \frac{1.0}{1.0 + 16.0 \cdot (0.5 - m)}$$
     If $e \neq 1.0$, $x = (\max(x, 0.0))^e$.
   * Output scaling:
     $$y = \text{outlow}[c] + x \cdot (\text{outhigh}[c] - \text{outlow}[c])$$
     If `flags & 0x400 == 0`, $y = \text{clamp}(y, 0.0, 1.0)$.
4. Grayscale nodes replicate channel 0 across RGB with alpha 1.0.

[References: `ops.rs::levels`; `program.rs::parse_node`.]

---

## 0x20 — graph image input / bitmap

Supplies external image inputs or decodes embedded raw bitmaps and JPEG resources.

### Record Layout and Parameter Slots

```
[opword]
[data: u32]             (Input UID if D == 0; block offset to bitmap/JPEG otherwise)
[data2: u32]            (Present only if D & 8 != 0 and D & 1 != 0: alpha JPEG offset)
[K: seed and size slots]
```

Byte `D` bit assignments:
* `D == 0`: External graph image input.
* Bit 0 (`D & 1`): Split alpha JPEG present (when combined with bit 3).
* Bit 2 (`D & 4`): 16-bit integer raw components.
* Bit 3 (`D & 8`): JPEG compressed stream.
* Bit 4 (`D & 0x10`): 16-bit half-float raw components.
* Bit 5 (`D & 0x20`): 32-bit float raw components.
* Bits 0–1 (`D & 3`): Raw channel count: 1 = Gray, 2 = RGB (alpha set to 1.0), other = RGBA.

### Execution

1. **Graph Image Input ($D == 0$):**
   `data` contains the numeric interface UID. If bound in interface defaults, takes the default RGBA float vector. If unbound, returns transparent black (gray alpha 1.0).
2. **Null Resource Pointer ($D \neq 0 \land \text{data} == 0$):**
   Returns transparent black (gray alpha 1.0).
3. **Embedded JPEG ($D \& 8 \neq 0$):**
   Reads `u32` byte length at `data`, decodes colour JPEG. If `data2 != 0`, decodes secondary JPEG and substitutes its channel 0 into alpha. Resizes to $(W, H)$ via `resized`.
4. **Raw Embedded Bitmap:**
   Format precedence:
   * If $D \& 0\times 20 \neq 0$: 4 bytes/comp (`f32` little-endian).
   * Else if $D \& 0\times 10 \neq 0$: 2 bytes/comp (half float, decoded via `f16_to_f32`).
   * Else if $D \& 4 \neq 0$: 2 bytes/comp (`u16` little-endian, divided by 65535.0).
   * Otherwise: 1 byte/comp (`u8`, divided by 255.0).
   Single-channel data is expanded to `Image::gray`.

[References: `ops.rs::bitmap`, `decode_jpeg`, `from_dynamic`, `f16_to_f32`; `program.rs::parse_node`.]

---

## 0x22 — text

Renders anti-aliased typographic strings from embedded TrueType fonts. Font parsing, glyph outline decomposition, and rasterization are specified in [SPEC-fxmap.md](SPEC-fxmap.md).

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]
[discarded: u32]
[data: u32]             (String offset or input UID)
[data2: u32]            (Font file offset)
[K: seed and size slots]
[background: u32]       (Present only if (flags >> 2) & 3 == 3)
[color: S(flags, 4, C)]
[matrix: S(flags, 10, 4)]
[position: S(flags, 6, 2)]
[fontsize: S(flags, 8, 1)]
[alignment: S(2, 1)]     (Present only if (flags >> 12) & 3 == 3)
```

[References: `ops.rs::text`; `text.rs`; `program.rs::parse_node`.]

---

## 0x24 — normal

Computes tangent-space normal vectors from a height map using backward differences and customizable coordinate handedness.

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]
[height: u32]
[K: seed and size slots]
[P: pixelsize S(D, 2, 2)]
[intensity: S(flags, 0, 1)]
[inversedy: boolean S(flags, 2, 1)]
[input2alpha: boolean S(flags, 4, 1)]
```

Byte `D` bits: Bit 0 wrap X, bit 1 wrap Y.

### Inputs and Sizing

Height input retains native dimensions $(sw, sh)$ and reads channel 0 ($P[0]$).

### Evaluation and Math

1. Evaluate parameters: `intensity` (default 1.0), `inversedy` (default false), `input2alpha` (default false for version $\ge 5$), `pixelsize` (default `(1.0, 1.0)`).
2. Version rule for alpha channel:
   * Program version $< 5$: Alpha channel unconditionally outputs the source height $c$.
   * Program version $\ge 5$: Alpha channel outputs source height $c$ only if `input2alpha` is true; otherwise outputs 1.0.
3. Backward differences $(dx, dy)$ evaluate identically to warp.
4. Normal vector:
   $$kx = \text{intensity} \cdot \frac{sw}{256.0} \cdot \text{ps}_x, \quad ky = \text{intensity} \cdot \frac{sh}{256.0} \cdot \text{ps}_y$$
   $$vx = -dx \cdot kx, \quad vy = -dy \cdot ky \cdot (\text{inversedy ? } -1.0 : 1.0)$$
   $$L = \sqrt{vx^2 + vy^2 + 1.0}$$
   $$\text{out} = \left[\frac{vx}{L} \cdot 0.5 + 0.5, \; \frac{vy}{L} \cdot 0.5 + 0.5, \; \frac{1.0}{L} \cdot 0.5 + 0.5, \; (\text{height\_alpha ? } c : 1.0)\right]$$

[References: `ops.rs::normal`; `program.rs::parse_node`.]

---

## 0x26 — dynamic gradient

Samples a 2D gradient texture using a 1D scalar driver input mapped along either the horizontal or vertical axis.

### Record Layout and Parameter Slots

```
[opword]
[driver: u32]
[gradient: u32]
[K: seed and size slots]
[coordinate: S(D, 1, 1)]
```

Byte `D` bit assignments:
* Bit 0 (`D & 1`): Repeat addressing mode.
* Bit 3 (`D & 8`): Vertical orientation flag.

### Inputs and Sizing

Driver input is resized to $(W, H)$ via `input_at`. Gradient image retains native dimensions $(gw, gh)$.

### Evaluation and Math

1. Evaluate parameter `coordinate` (default 0.0).
2. Driver coordinate extraction: $t = P_{\text{driver}}[3]$ for colour, $P_{\text{driver}}[0]$ for gray.
   $$t = \text{repeat ? rem\_euclid}(t, 1.0) : \text{clamp}(t, 0.0, 1.0)$$
3. Axis mapping:
   $$(a, b) = \text{vertical ? } (\text{coordinate}, t) : (t, \text{coordinate})$$
   $$u = \frac{a \cdot (gw - 1.0) + 0.5}{gw}, \quad v = \frac{b \cdot (gh - 1.0) + 0.5}{gh}$$
4. Gradient sampling:
   Sample gradient with coupled texture sampling `bilinear_or` in gradient format `gfmt`, wrapping both axes if repeat is true, else clamping. Grayscale outputs take channel 0.

[References: `ops.rs::dyn_gradient`; `program.rs::parse_node`.]

---

## 0x28 — pixel and value processor

Executes dynamic bytecode functions per pixel (pixel processor) or once per graph node (value processor).

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]
[inputs: u32 * (flags & 0xffff)]
[K: seed and size slots]
[func_ptr: u32]         (Block byte offset to bytecode function)
```

Bit allocations in `flags`:
* Bits 0–15: Input count $k$.
* Bits 16–31: Value processor selector. Any non-zero value selects value processing; zero selects pixel processing.

### Value Processor Execution

1. Executed during dependency traversal with environment `NodeEnv`.
2. RNG state initialized with $\text{pixel\_seed}(\text{seed}, [0.0, 0.0])$.
3. Shared node variables are updated by the function. Writes emitted via opcode 6 are committed to the runtime value processor table in execution order.
4. Returns a 1×1 placeholder image `[0.0, 0.0, 0.0, 0.0]`, subsequently quantized to node format (grayscale alpha becomes 1.0).

### Pixel Processor Execution

1. Evaluated once per pixel centre $((x + 0.5) / W, (y + 0.5) / H)$ with environment `PixEnv`.
2. RNG state initialized per pixel:
   $$h = 0$$
   $$\text{for } w \in [\text{seed}, \text{bits}(pos_x), \text{bits}(pos_y)]: \quad h = h + w; \quad h = h + (h \ll 10); \quad h = h \oplus (h \gg 6)$$
   $$h = h + (h \ll 3); \quad h = h \oplus (h \gg 11); \quad h = h + (h \ll 15)$$
   $$\text{state} = h \& 0\text{xffffff}$$
3. Inputs retain native dimensions and are sampled via VM sampler instructions 51/52 with wrapping enabled on both axes.
4. Function result conversion:
   * Grayscale nodes (`colour == 0`): Output `Image::gray(c[0])` (`[c[0], c[0], c[0], 1.0]`) regardless of the number of components returned by the function.
   * Colour nodes (`colour == 1`):
     * $\ge 4$ components: `[c[0], c[1], c[2], c[3]]`.
     * 3 components: `[c[0], c[1], c[2], 1.0]`.
     * $\le 2$ components: `[c[0], c[0], c[0], 1.0]` (broadcast $c[0]$ across RGB with alpha 1.0).

[References: `ops.rs::pixel_processor`, `pixel_seed`; `SPEC-vm.md`.]

---

## 0x2a — distance

Computes the distance transform from a binary threshold mask with metric selection, falloff, and optional source combining.

### Record Layout and Parameter Slots

```
[opword]
[flags: u32]
[mask: u32]
[source: u32]           (Present only if flags & 1 != 0)
[K: seed and size slots]
[P: pixelsize S(D, 2, 2)]
[distance: S(flags, 2, 1)]
[mode: S(2, 1)]         (Present only if flags & 0x40 != 0)
```

Bit allocations in `flags`:
* Bit 0 (`0x1`): Source image present.
* Bit 1 (`0x2`): Combine mode enabled.
* Bits 4–5: Default metric mode.
* Bit 6 (`0x40`): Function mode override present.

Byte `D` bits: Bit 0 wrap X, bit 1 wrap Y.

### Inputs and Sizing

Mask is resized to $(W, H)$ via `input_at`. If source is present, it is also resized to $(W, H)$ via `input_at`. Mask inside is strictly:
$$\text{inside} = (P_{\text{mask}}[3] \text{ if colour else } P_{\text{mask}}[0]) > 0.5$$

### Evaluation Order and Quantization (14.1 Profile)

1. Parameters evaluate in observable order: `distance` (default 10.0), `pixelsize` (default `(1.0, 1.0)`), `mode` (default `(flags >> 4) & 3`), then `distance` is evaluated a second time.
2. Effective distance and scale:
   $$d_{\max} = \frac{\text{second\_distance}}{256.0}, \quad \text{inv} = \frac{1.0}{d_{\max}}$$
3. Seed coordinate quantization:
   $$\text{code}(s, n) = \text{f32}\left(\text{round}\left(\frac{s + 0.5}{n} \cdot 65535.0\right)\right)$$
4. Delta per axis between query pixel $p$ and seed $s$:
   $$\Delta = \frac{p + 0.5}{n} - \frac{\text{code}(s, n)}{65535.0}$$
   If wrap is enabled and $|\Delta| > 0.5$, $\Delta = \Delta - \text{signum}(\Delta)$.
   $$du = \Delta_x \cdot \frac{1.0}{\text{ps}_x}, \quad dv = \Delta_y \cdot \frac{1.0}{\text{ps}_y}$$
5. Metric selection:
   $$\text{metric} = \begin{cases} \text{fma}(du, du, dv^2) & \text{if mode } 0 \text{ (Euclidean)} \\ |du| + |dv| & \text{if mode } 1 \text{ (Manhattan)} \\ \max(|du|, |dv|) & \text{otherwise (Chebyshev)} \end{cases}$$
   Final distance $d = \sqrt{\text{metric}}$ for mode 0, else $\text{metric}$.

### Jump Flooding Search

* Seeds are initialized to $k$ if $\text{inside}[k]$, else $\infty$.
* Step size begins at $\max(W, H) / 2$, halving each iteration down to 1.
* Evaluates 8 neighbor directions $(dx, dy)$ in order: $(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)$.
* Neighbor coordinates are computed in normalized space from pixel center: for axis coordinate $p \in \{x, y\}$, axis size $n \in \{W, H\}$, and direction component $d \in \{dx, dy\}$:
  $$\text{uv} = \frac{p + 0.5}{n} + \frac{d \cdot \text{step}}{\max(W, H)}$$
  $$\text{coord} = \begin{cases} \lfloor \text{uv} \cdot n \rfloor \text{ rem\_euclid } n & \text{if wrapped} \\ \text{clamp}(\lfloor \text{uv} \cdot n \rfloor, 0, n - 1) & \text{otherwise} \end{cases}$$
* Neighbor seed at $(\text{coord}_x, \text{coord}_y)$ is tested against current winner using $\text{qd2}(k, q)$. Best seed replaces winner only on strictly smaller metric.

### Falloff and Combine Output

If a valid seed winner exists:
* If query pixel is inside mask, force $d = 0.0$.
* $\text{within} = d < d_{\max}$.
* $\text{falloff} = \max(\text{fma}(-d, \text{inv}, 1.0), 0.0)$.

Outputs:
* **Without source:** Grayscale outputs `Image::gray(falloff)`; colour outputs `[falloff, falloff, falloff, 1.0]`.
* **With source, Combine clear (`flags & 2 == 0`):**
  Output nearest seed pixel $P_{\text{src}}[\text{winner}]$ if within radius; original pixel $P_{\text{src}}[\text{query}]$ otherwise.
* **With source, Combine set (`flags & 2 != 0`):**
  * Grayscale: $P_{\text{src}}[\text{winner}][0] \cdot \text{falloff}$.
  * Colour: RGB is nearest winner RGB if within radius, else query RGB; Alpha is $P_{\text{src}}[\text{winner}][3] \cdot \text{falloff}$.
* If no seed exists across the entire image: candidate is query pixel, falloff is 0.0.

[References: `ops.rs::distance`; `program.rs::parse_node`.]

---

## 0x2c — curves

Maps image color channels through 1D piecewise cubic Bézier curves evaluated via high-precision lookup tables.

### Record Layout and Parameter Slots

```
[opword]
[source: u32]
[count: u32]            (Maximum point count)
[curve_ptr: u32]        (Block byte offset to curve data blob)
[K: seed and size slots]
```

Byte `D` bit 1 (`D & 2`): When set, the input image is not read; the curve is evaluated along the horizontal coordinate $(x + 0.5) / W$ across all channels.

### Inputs and Curve Data Encoding

Source image is resized to $(W, H)$ via `input_at`.
Blob at `curve_ptr` contains 1 curve for grayscale nodes, 5 curves for colour nodes (master luminance, R, G, B, A).
Each curve begins with `u32` point count $k = \min(\text{blob\_count}, \text{count})$. Then follow $k$ points of 6 words `f32`:
$$[x, y, \text{left}_x, \text{left}_y, \text{right}_x, \text{right}_y]$$

### Segment Evaluation and LUT Construction

An empty curve evaluates as the identity function $f(x) = x$.
For non-empty curves, a 1025-texel LUT is constructed at sample positions $t = i / 1024.0$ ($i \in [0, 1024]$):
1. Endpoint extension:
   * If $x \le \text{pts}[0].x$: evaluate synthetic cubic Bézier segment from $P_0 = [0.0, \text{pts}[0].y]$ to $P_3 = [\text{pts}[0].x, \text{pts}[0].y]$, with control handles $P_1 = [0.0, \text{pts}[0].y]$ and $P_2 = [\text{pts}[0].\text{left}_x, \text{pts}[0].\text{left}_y]$.
   * If $x \ge \text{last}.x$: evaluate synthetic cubic Bézier segment from $P_0 = [\text{last}.x, \text{last}.y]$ to $P_3 = [1.0, \text{last}.y]$, with control handles $P_1 = [\text{last}.\text{right}_x, \text{last}.\text{right}_y]$ and $P_2 = [1.0, \text{last}.y]$.
2. In-between segment: $P_0 = a.\text{pos}$, $P_1 = a.\text{right}$, $P_2 = b.\text{left}$, $P_3 = b.\text{pos}$.
3. Cubic equation:
   $$\text{cubic}(t) = (1 - t)^3 \cdot P_0 + 3(1 - t)^2 t \cdot P_1 + 3(1 - t) t^2 \cdot P_2 + t^3 \cdot P_3$$
4. 40 bisection iterations find $t$ such that $\text{cubic}(t).x = x$. Midpoint $y$ is evaluated at $t$.

Sampling the LUT:
$$t = \text{clamp}(x, 0.0, 1.0) \cdot 1024.0, \quad i = \min(\lfloor t \rfloor, 1023)$$
$$f = \frac{\text{round}((t - i) \cdot 256.0)}{256.0}$$
$$\text{val} = \text{LUT}[i] + (\text{LUT}[i + 1] - \text{LUT}[i]) \cdot f$$

### Colour Application Order

For colour nodes:
1. Master curve (curve 0) is applied independently to channels R, G, B:
   $$C[ch] = \text{eval}(0, C[ch]) \quad \text{for } ch \in [0, 2]$$
2. Individual channel curves (curves 1, 2, 3, 4) are applied to channels R, G, B, A:
   $$C[ch] = \text{eval}(ch + 1, C[ch]) \quad \text{for } ch \in [0, 3]$$

[References: `ops.rs::curve`, `curve_points`, `bezier_curve`; `program.rs::parse_node`.]

---

## A. Exact smooth-gradient LUT algorithm

The following self-contained algorithm specifies the normative cubic construction for gradient maps ($0\times 00$) with `v8word` ($w$). It fixes iteration bounds, EPS threshold, and floating-point evaluation order. [Reference: `ops.rs::build_v8_lut`.]

```rust
fn build_v8_lut(sorted_keys: &[(f32, [f32; 4])], w: f32) -> [[f32; 4]; 1024] {
    const STEP: f32 = 1.0 / 1023.0;
    const EPS: f32 = 0.5 * STEP;

    #[inline]
    fn mix(a: f32, b: f32, t: f32) -> f32 {
        a * (1.0 - t) + b * t
    }
    #[inline]
    fn cubic(p: [f32; 4], t: f32) -> f32 {
        let a = mix(p[0], p[1], t);
        let b = mix(p[1], p[2], t);
        let c = mix(p[2], p[3], t);
        mix(mix(a, b, t), mix(b, c, t), t)
    }
    #[inline]
    fn derivative(p: [f32; 4], t: f32) -> f32 {
        let a = 3.0 * (p[1] - p[0]);
        let b = 3.0 * (p[2] - p[1]);
        let c = 3.0 * (p[3] - p[2]);
        mix(mix(a, b, t), mix(b, c, t), t)
    }
    fn invert(p: [f32; 4], x: f32, mut t: f32) -> f32 {
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for outer in 0..32 {
            let guess = t;
            for _ in 0..100 {
                let err = cubic(p, t) - x;
                if err.abs() < EPS {
                    return t;
                }
                let d = derivative(p, t);
                if d.abs() < 1.0e-4 {
                    break;
                }
                t -= err / d;
                if !t.is_finite() {
                    break;
                }
            }
            t = guess;
            if outer != 0 && cubic(p, hi) - cubic(p, lo) < EPS {
                return t;
            }
            if cubic(p, t) < x {
                lo = t;
            } else {
                hi = t;
            }
            t = (lo + hi) * 0.5;
        }
        t
    }

    if sorted_keys.is_empty() {
        return [[0.0, 0.0, 0.0, 1.0]; 1024];
    }
    if sorted_keys.len() == 1 {
        return [sorted_keys[0].1; 1024];
    }
    let first = sorted_keys[0];
    let last = sorted_keys[sorted_keys.len() - 1];
    let mut keys = Vec::with_capacity(sorted_keys.len() + 2);
    if first.0 > 0.0 {
        keys.push((0.0, first.1));
    }
    keys.extend_from_slice(sorted_keys);
    if last.0 < 1.0 {
        keys.push((1.0, last.1));
    }
    let n = keys.len();
    let mut left = keys.clone();
    let mut right = keys.clone();
    if n > 2 {
        for j in 1..n - 1 {
            let span = keys[j + 1].0 - keys[j - 1].0;
            let dl = w * (keys[j].0 - keys[j - 1].0) * 0.5;
            let dr = w * (keys[j + 1].0 - keys[j].0) * 0.5;
            left[j].0 -= dl;
            right[j].0 += dr;
            for c in 0..4 {
                let slope = if span != 0.0 { (keys[j + 1].1[c] - keys[j - 1].1[c]) / span } else { 0.0 };
                left[j].1[c] -= dl * slope;
                right[j].1[c] += dr * slope;
            }
        }
        right[0].0 = keys[0].0 + (left[1].0 - keys[0].0) * w * 0.5;
        left[n - 1].0 = keys[n - 1].0 + (right[n - 2].0 - keys[n - 1].0) * w * 0.5;
        for c in 0..4 {
            right[0].1[c] = keys[0].1[c] + (left[1].1[c] - keys[0].1[c]) * w * 0.5;
            left[n - 1].1[c] = keys[n - 1].1[c] + (right[n - 2].1[c] - keys[n - 1].1[c]) * w * 0.5;
        }
    }
    let (mut j, mut t) = (0usize, 0.5f32);
    std::array::from_fn(|i| {
        let x = i as f32 * STEP;
        if j < n - 2 && x > keys[j + 1].0 + EPS {
            j += 1;
            t = 0.5;
        }
        let (a, b) = (keys[j], keys[j + 1]);
        if (x - a.0).abs() <= EPS {
            return a.1;
        }
        if (x - b.0).abs() <= EPS {
            return b.1;
        }
        if x <= first.0 {
            return first.1;
        }
        if x >= last.0 {
            return last.1;
        }
        if w == 0.0 || n == 2 || a.0 == b.0 {
            return std::array::from_fn(|c| a.1[c] + (x - a.0) * ((b.1[c] - a.1[c]) / (b.0 - a.0)));
        }
        let px = [a.0, right[j].0, left[j + 1].0, b.0];
        t = invert(px, x, t);
        std::array::from_fn(|c| cubic([a.1[c], right[j].1[c], left[j + 1].1[c], b.1[c]], t))
    })
}
```
