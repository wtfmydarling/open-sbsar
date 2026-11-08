# open-sbsar draft 1 — node operations

[Core specification](SPEC.md) defines record layouts, numeric rules, storage and sampler primitives. Every node receives final output storage conversion. The parameter orders given below are **evaluation** orders; record parsing order remains the core table. `resize` means clamped separable 1/256 sampling. `tex` means coupled texture sampling in the sampled node's format, with wrap bits from D and edge clamp unless otherwise stated. All dimensions are positive. Missing required graph inputs are not implicit black images: valid compiled programs normally supply uniform nodes. Disconnected processor inputs are the explicit exception.

## 0x00 — gradient map

Input: image resized to node dimensions; lookup coordinate is alpha for colour, R for gray. Parameter: optional v8 smoothness word, read as f32 without clamping. D bit 0 indicates stored midpoint fields. Float keys: position, C values, optional midpoint, each f32. Packed keys: u16 position/65535, then RGBA bytes/255 for colour or u16 gray/65535, then optional u16 midpoint. Pad the complete blob to four bytes. Stored midpoint values are skipped by the current evaluator. Stable-sort by f32 total order of position. Before v8, float-key flag is D bit 1 and repeat bit 2; from v8 they are bits 2 and 3.

Without the v8 smoothness word, interpolate key intervals in f64 using `a+(b-a)*t`. For v<8 smoothstep t when midpoint flag is absent; for v≥8 smoothstep t when midpoint flag is present. Smoothstep is `t*t*(3-2*t)`. At/before first and after last use endpoint colour; no keys → `(0,0,0,1)`. At a repeated-position interval use t=1 if its upper position is not greater than its lower position.

There are three LUT paths:

| Condition | LUT construction |
|---|---|
| v≥8 and D&2 | 1024 unclamped f32 texels, special cubic algorithm below, all target formats |
| no such word, target half/f32 | 1024 unclamped f32 texels at f64 i/1023, interval evaluation above, cast to f32 |
| no such word, target UNORM8/16 | 256 UNORM16 texels at f64 i/255; cast each interpolated value to f32, multiply by 65535 in f32, clamp to [0,65535], then f64 floor(value+.5) |

Nonfinite input coordinate becomes zero. Set `coord=fma(clamp(x,0,1),(N-1)/N,.5/N)` or, for repeat, `rem_euclid(fma(x,(N-1)/N,.5/N),1)`. Compute `u=coord*N-.5`, floor texel index, round fraction to 1/256 (ties up). Clamp texel indices when not repeating, wrap when repeating. Integer LUT interpolation uses `(a*(256-w)+b*w+128)>>8`, then divide by 65535. Float LUT interpolation returns an endpoint exactly for weight 0 or 1, otherwise computes the weighted sum in f64 and casts once to f32. Half targets truncate to half before final storage, on both float-LUT paths.

The cubic construction listing in §A is normative; in particular, advance at most **one** key interval per LUT texel, preserve the warm-start Newton parameter, do not clamp smoothness, and keep the explicit endpoint snaps. These details permit large extrapolated values at tightly clustered keys. [References: `ops.rs::gradient_map`, `lookup64`, `build_v8_lut`, `sample_f32_lut`; `image.rs::half_trunc`.]

## 0x02 — blend

Inputs: background B, foreground F, each resized. F bits 0–3 select mode; bits 11–12 select alpha mode. Evaluate opacity or, for image opacity, opacity multiplier. Default opacity/multiplier=1. Image opacity uses resized R. Clamp their product to [0,1]. Optional crop words pack `(u16 x0,u16 y0)` then `(u16 x1,u16 y1)`, each divided by 32768; only pixels `floor(x0*W)<=x<floor(x1*W)` and similarly Y blend. Outside retain background.

Define `mix(b,f,t)=b*(1-t)+f*t`, with separate f32 products. The mode function g is:

| Mode | g(b,f) |
|---:|---|
| 0,7 | f |
| 1 | b+f |
| 2 | b−f |
| 3 | b*f |
| 4 | b+2*f−1 |
| 5,6 | max(b,f), min(b,f) |
| 8 | b/f; NaN result becomes 1 |
| 9 | b<.5 ? 2*b*f : 1−2*(1−b)*(1−f) |
| 10 | 1−(1−b)*(1−f) |
| 11 | f<.5 ? 2*b*f+b*b*(1−2*f) : 2*b*(1−f)+sqrt(b)*(2*f−1) |
| other | CPU uses f; unmeasured mode is Open O05 |

Gray result is mix(B.r,g(B.r,F.r),opacity). Colour mode 7 mixes all four channels directly. Otherwise, with o=opacity, ba=B.a, fa=F.a:

* Alpha mode 0, and mode 3 unless blend mode is copy: RGB=mix(B,g(B,F),fa*o); A=ba+o*fa*(1−ba).
* Alpha mode 1: RGB=mix(B,g(B,F),o); same alpha.
* Alpha mode 2: `afull=fa+ba*(1-fa); a=ba*(1-o)+afull*o`; `mixed=g(B,F)*fa*ba+F*fa*(1-ba)+B*ba*(1-fa)` per RGB; `pre=B*ba*(1-o)+mixed*o`; RGB=pre/a when a!=0, else zero; A=a.
* Alpha mode 3 and copy: RGB=F*o+B*(1−fa*o); A=fa*o+ba*(1−fa*o).

No additional intermediate colour clamp precedes final storage. [Reference: `ops.rs::blend`.]

## 0x04 — transformation

Evaluate matrix (identity), offset (0,0), matte (0,0,0,0), then dynamic mip if present. Matrix is column-major output-to-source: `u=m0*dx+m2*dy+.5+offset.x`, `v=m1*dx+m3*dy+.5+offset.y`, with dx/dy destination pixel centres minus .5. F bit 5 chooses bilinear; low five bits select manual mip 0…29, 30 function, 31 automatic. Manual mip=max(0,value) rounded away at half ties; clamp chosen level to the available pyramid. A 1×1 source remains its own last level.

For auto bilinear, zero offset, and signed permutation matrix (diagonal ±1 or off-diagonal ±1), use the special resize path: swap target width/height for an off-diagonal matrix, halve all oversized axes in node format until they fit, then double undersized axes with coupled sampling in the original source node format. Upsample intermediates are not stored/quantized. Finally flip/transpose texels as the matrix dictates. Identity merely copies the resized pixels. Manual mip zero bypasses this path.

Otherwise compute `du=sqrt((m0*sw/W)^2+(m2*sw/H)^2)`, `dv=sqrt((m1*sh/W)^2+(m3*sh/H)^2)`, `lod=log2(max(du,dv))`. Auto level=`floor(max(lod,0)+.5+1/2048)`. General magnification samples level zero directly. Mips ≥1 are built in transformation output format, except opcode-20 sources use f32 mip storage. Level zero sampling uses source format; higher levels use mip format. Nearest also samples the selected mip.

Bilinear uses coupled weights and matte border on nonwrapped axes, including partial contribution just outside UV [0,1]. Exact identity+zero-offset clamps borders, including manual-mip sampling. Gray matte is `(matte.r,matte.r,matte.r,1)`. Nearest returns matte if a nonwrapped UV is outside inclusive [0,1], otherwise fetches with wrap/clamp. [References: `ops.rs::transformation`, `mip_levels_fmt`; `image.rs` sampling and per-axis resizing.]

## 0x06 — grayscale conversion / channel shuffle

Gray: resize input; evaluate weights then matte. Defaults `(1/3,1/3,1/3,0)` and matte 1. Sum `R*w0+G*w1+B*w2+A*w3` in order. D bit 4 replaces g with `matte+(g-matte)*A`. Store gray.

Colour: resize both inputs. Each byte of flags selects corresponding output RGBA channel: 0…3 first input, 4…7 second. The CPU handles larger selectors as second input `(selector−4)&3`; validity of such selectors is **Open O05**. [Reference: `ops.rs::exec`.]

## 0x08 — FX-Map

See [SPEC-fxmap.md](SPEC-fxmap.md), including traversal, state, patterns, draw grouping and per-layer storage. [Reference: `fxmap.rs::render`.]

## 0x0a — SVG command stream

See SPEC-fxmap, SVG section. This is a binary command stream, not embedded XML SVG. [Reference: `ops.rs::svg`.]

## 0x0c — uniform

Evaluate colour; default gray zero or RGBA `(0,0,0,1)`. Fill every pixel then store in node format. Scalar function values broadcast through ordinary parameter evaluation. [Reference: `ops.rs::exec`.]

## 0x0e — warp

Evaluate intensity (1), then pixelsize (1,1). Both source and gradient retain native dimensions. Gradient value is alpha for colour, R for gray. At destination centre compute backward differences `dx=g(u,v)-g(u-1/gw,v)`, `dy=g(u,v)-g(u,v-1/gh)` using direct fetches when gradient dimensions equal output, otherwise coupled sampling in gradient format. Sample source with coupled weights at `(u+intensity*gw/256*ps.x*dx, v+intensity*gh/256*ps.y*dy)`. Wrap/clamp follows D. Nonzero v9 leading flags remain **Open O10**. [Reference: `ops.rs::warp`.]

## 0x10 — emboss

Evaluate intensity (5), lightangle (0 turns), highlight (1,1,1,1), shadow (0,0,0,1), pixelsize. Source is resized; height retains native dimensions. Use height alpha for colour or R for gray and native backward differences as in warp. Let `kx=intensity*hw/256*ps.x`, `ky=intensity*hh/256*ps.y`, `s=kx*dx*cos(TAU*angle)+ky*dy*sin(TAU*angle)`. RGB adds `s*highlight` when s>0, otherwise `s*(1-shadow)`; gray uses first highlight/shadow component. Preserve source alpha. [Reference: `ops.rs::emboss`.]

## 0x12 — legacy motion blur

Inputs are source and gradient driver; retain both at native sizes. Evaluate intensity (1), shiftangle (0 turns), then pixelsize. At destination pixel centre `(u,v)`, read gradient alpha for colour or R for gray and compute `dx=g(u,v)-g(u-1/gw,v)`, `dy=g(u,v)-g(u,v-1/gh)`: direct fetches when gradient and output dimensions match, otherwise coupled samples in gradient storage format. Set `gx=dx*gw*ps.x`, `gy=dy*gh*ps.y`, `shift=TAU*shiftangle`, `scale=intensity*1.005/128`, then `ux=(gx*cos(shift)-gy*sin(shift))*scale`, `uy=(gx*sin(shift)+gy*cos(shift))*scale`. Accumulate six coupled source-format samples at `(u+t*ux,v+t*uy)`, `t=(j+.5)/6-.5`, j=0…5, in order; divide all four channels by 6. Colour retains the averaged alpha; gray uses `Image::gray(acc[0]/6)`. All fetches use wrap/clamp per D. The fitted constant 1.005 is the normative CPU value. [Reference: `ops.rs::legacy_motion_blur`.]

The operation helper returns resized source before evaluating parameters if its gradient entry is absent or out of image-table bounds. Parsed records always supply two indices, and normal runtime scheduling rejects a disconnected `0xffffffff` gradient for this opcode before reaching that helper, producing the SPEC.md §5 forward-input placeholder instead. Thus the helper fallback does not authorize disconnected gradients in the binary format (**Open O08**). [References: `program.rs::parse_node`; `ops.rs::legacy_motion_blur`, `run_inner`.]

## 0x14 — blur

Evaluate intensity (10), pixelsize. Radii `rx=intensity*W/256*ps.x`, `ry=intensity*H/256*ps.y`. Apply horizontal then vertical box passes; intermediates are f32 without node-format storage. A radius ≤.5 gives a single centre tap. Otherwise for integer t from `-ceil(r+.5)` through `ceil(r+.5)`, overlap `max(0,min(t+.5,r)-max(t-.5,-r))/(2*r)` is its weight.

Adjust weights to model paired fetches. Let k be largest absolute nonzero tap offset. k<2: unchanged. k=2: pair (1,2) and (−1,−2). k>2: pair consecutive offsets (t,t+1) when `(output_coordinate+t).rem_euclid(2)=0`. For each pair weights a,b, set `q=round(b/(a+b)*256)/256`, and replace weights by `(a+b)*(1-q),(a+b)*q`. The ordinary pass accumulates RGBA from direct integer fetches in ascending tap order with these adjusted weights; it does not replace pairs with bilinear fetches.

For a native source with at least output dimensions on both axes and differing dimensions, the horizontal pass instead samples that native image. Emit pairs first in the order above, each as one coupled sample at offset `first+q*(second-first)`, weight a+b; then emit unpaired integer taps in ascending order. Offsets are measured in **output** pixels (`offset/W`). The vertical pass uses the node-size f32 intermediate and parity-adjusted taps. Otherwise resize first. Boundary fetches wrap per D or clamp. Remaining exceptional filter edge parity is **Open O23**. [References: `ops.rs::blur`, `box_taps`, `blur_taps`, `blur_fetches`, `box_pass_parity`.]

## 0x16 — directional motion blur

Resize source. Evaluate intensity (10), angle (0 turns), pixelsize. Define `r=intensity*W/256`, `(dx,dy)=(cos(TAU*angle)*ps.x,sin(TAU*angle)*ps.y)`. Major axis is Y when abs(dy)>=abs(dx), including ties. If major magnitude is zero return source. Box radius=`.5+(r-.5)*major`. Use unpaired box taps from 0x14.

If there are ≤3 taps or `min(abs(dx),abs(dy))<1e-6`, sample directly at destination centre plus `(t*dx/major,t*dy/major)` in output pixels, using separable 1/256 bilinear, weight by each tap and accumulate. For the larger off-axis case use the following continued-shear algorithm (Y-major shown; swap X/Y for X-major):

1. Shear displacement at row y: `(y+.5-H*.5)*dx/dy` in X pixels.
2. If X is untiled, extend each side of the intermediate by `ep=ceil(H*.5*abs(dx/dy))+2` pixels; otherwise ep=0. Intermediate dimensions are `(W+2*ep,H)`.
3. For each intermediate coordinate `(bx,y)`, let x=bx−ep. For every box tap (t,wt), set yy=y+t and sh=`(yy+.5-H*.5)*dx/dy`. Coupled-sample the resized source at `((x+.5+sh)/W,(yy+.5)/H)` and accumulate sample*wt. Do not wrap the shear coordinate before calculating sh. Do not quantize the f32 intermediate.
4. Unshear at output (x,y) by sampling the intermediate at `((x+ep+.5-sh(y))/(W+2*ep),(y+.5)/H)`.

Both coupled sampling stages use the original source-format argument, even though the intermediate is f32. This detail is part of CPU behavior. Wrap/clamp is applied at fetches. The continuing shear prevents seam artifacts from wrapping a finite sheared buffer. Remaining external-renderer precision is **Open O23**. [Reference: `ops.rs::motion_blur`.]

## 0x18 — directional warp

Retain native source; resize amount input. Evaluate intensity (10), angle (0 turns), pixelsize, then optional amount-offset (0). Let `m=(amount alpha if colour else R)-offset`. Coupled-sample native source at `UV+(cos(angle*TAU)*intensity/256*ps.x, sin(angle*TAU)*intensity/256*ps.y)*m`. Flags bits 7–8 describe the offset slot observed in v9; it is a subtraction, not an angle or intensity multiplier. [Reference: `ops.rs::directional_warp`.]

## 0x1a — sharpen

Evaluate intensity (1), pixelsize. When input size differs, sample at native texel spacing through coupled sampling; otherwise direct fetch. For each RGBA channel `out=c+intensity*((2*c-left-right)*ps.x+(2*c-up-down)*ps.y)`. Gray stores only R. Neighbors wrap/clamp per D. [Reference: `ops.rs::sharpen`.]

## 0x1c — HSL

Resize colour input. Evaluate hue, saturation, luminosity, all default .5. RGB→HSL: mx=max(R,G,B), mn=min, l=(mx+mn)/2. If mx=mn, h=s=0. Otherwise d=mx−mn, E=1/4096; `s=d/(1-abs(2*l-1)+E)`. Raw hue numerator is `(G-B)+(G<B?6*d:0)` for mx=R; `(B-R)+2*d` for mx=G; `(R-G)+4*d` otherwise. Bias is −4E if raw=0, −2E if raw<2d, zero if raw<=4d, +2E otherwise. `h=rem_euclid((raw+bias)/(6*(d+E)),1)`.

Set `h=rem_euclid(h+2*(hue-.5),1)`, `s=clamp(s+2*(saturation-.5),0,1)`, `l=clamp(l+2*(luminosity-.5),0,1)`. Convert: if s=0 RGB=l. Otherwise q=l*(1+s) if l<.5 else l+s−l*s; p=2*l−q. For t wrapped to [0,1), channel f(t)=p+(q−p)*6*t for t<1/6, q for t<.5, p+(q−p)*(2/3−t)*6 for t<2/3, p otherwise. RGB=f(h+1/3),f(h),f(h−1/3). Preserve alpha. [References: `ops.rs::hsl`, `rgb_to_hsl`, `hsl_to_rgb`.]

## 0x1e — levels

Resize input. Evaluate inlow(0), inhigh(1), inmid(.5), outlow(0), outhigh(1), per C channels. Clamp **all parameters** to [0,1] regardless of F bit 10. This bit disables intermediate/result clamps when set. Let d=inhigh−inlow except exactly equal endpoints use 1e−4; x=(input−inlow)/d; clamp x if enabled. Exponent e is `1+16*(mid-.5)` for mid>=.5, else `1/(1+16*(.5-mid))`. If e!=1 set x=max(x,0)^e. Return `outlow+x*(outhigh-outlow)`, clamped when enabled. Negative nonzero input ranges remain negative. [Reference: `ops.rs::levels`.]

## 0x20 — graph image input / bitmap

D=0 means data word is input UID. With no bound image the CPU fills with the interface numeric default (v8+ RGBA) or transparent black; gray uses R. External image binding is **Open O04**. D!=0 and data pointer zero returns solid zero (gray alpha one).

D bit 3 selects JPEG: at each pointer is u32 byte length followed by JPEG bytes; bit 0 adds a second JPEG carrying alpha in its first channel. Decode colour JPEG, replace alpha where second pixels are available, then resize. Baseline decoder is included in SPEC-fxmap §B. Unsupported JPEGs use an image-library fallback whose bit-exact semantics are **Open O21**.

Raw channels from D&3: 1 gray, 2 RGB, otherwise RGBA. D bit 5 takes precedence and selects f32, else bit 4 selects half, else bit 2 selects u16, otherwise u8. Pixels are packed, top-to-bottom row-major, little-endian component words, padded to four bytes. u8/u16 normalize by 255/65535; half expands sign/exponent/mantissa (including subnormals, infinities and NaNs). Missing RGB alpha is one; single-channel input replicates gray with alpha one. No sRGB transfer function is applied by the CPU. Unmeasured colour-space tags are **Open O21**. [References: `ops.rs::bitmap`, `decode_jpeg`, `from_dynamic`, `f16_to_f32`; `reader.py::bitmap_span`.]

## 0x22 — text

See SPEC-fxmap text section and font/rasterizer listings. [Reference: `ops.rs::text`; `text.rs`.]

## 0x24 — normal

Keep native height image; use R. Evaluate intensity (1), inversedy (false), input2alpha when version≥5, then pixelsize. Version<5 always emits height in alpha; newer versions do so only for true boolean slot, otherwise alpha one. D wrap bits apply. At output centre get native backward differences as in warp. `vx=-dx*intensity*sw/256*ps.x`, `vy=-dy*intensity*sh/256*ps.y`, with Y sign reversed when inversedy. `L=sqrt(vx*vx+vy*vy+1)`. Output `(vx/L*.5+.5,vy/L*.5+.5,1/L*.5+.5,alpha)`. [Reference: `ops.rs::normal`.]

## 0x26 — dynamic gradient

Resize first input; retain gradient native dimensions. Evaluate coordinate (0). Input t is alpha for colour or R for gray; D bit 0 wraps t with rem_euclid(t,1), otherwise clamps t to [0,1]. D bit 3 selects vertical: `(a,b)=(coordinate,t)`, otherwise `(t,coordinate)`. Sample gradient at `((a*(gw-1)+.5)/gw,(b*(gh-1)+.5)/gh)` with coupled weights, wrapping both axes if repeat else clamping. Gray output takes sampled R. Broader repeat-edge evidence remains **Open O23**; the preceding CPU rule is definite. [Reference: `ops.rs::dyn_gradient`.]

## 0x28 — pixel and value processor

F low 16 bits count input indices; any nonzero high 16 bits selects value processing. Evaluate embedded function in the environments specified in SPEC-vm. Pixel processing invokes once per destination centre with independent variables and RNG; input samples retain native sizes. Value processing invokes once with NodeEnv, commits pending write slots in execution order after the function returns, and returns a 1×1 zero placeholder image, subsequently subject to normal node storage conversion. See SPEC-vm §4 for visibility and typing. Invariant-register precomputation is permitted only when it preserves demand order and side effects; it is an optimization, not extra evaluations observable to the program. [Reference: `ops.rs::pixel_processor`.]

## 0x2a — distance

Resize mask and optional source. Mask inside is `(alpha if colour else R)>.5`, strictly. Evaluate distance (10), pixelsize, mode, then **distance again**; the first distance evaluation's scalar is unused on the default path but its variable effects remain observable. Mode defaults to F bits 4–5; F bit 6 supplies a function override. D independently wraps X/Y.

Set dmax=second_distance/256 and inv=1/dmax. Seed coordinates are `f32(round(f64((pixel+.5)/dimension)*65535))` with the fraction computed in f64. Query coordinates remain f32 exact pixel centres. Delta per axis is query−seed_code/65535; when wrapped and abs(delta)>.5 subtract sign(delta); multiply by reciprocal pixelsize. Mode 0 metric is `fma(du,du,dv*dv)`, mode 1 `abs(du)+abs(dv)`, other modes max(abs(du),abs(dv)). Mode 0 final distance is sqrt(metric); other modes use metric directly.

Find candidate seeds by jump flooding, not an exact Euclidean transform. Initialize each inside pixel to itself and others invalid. Starting step=max(W,H)/2, repeatedly process all pixels from the previous array, then halve integer step until zero. Keep current seed as first candidate; try directions `(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)` in that order. Neighbor coordinate is `floor((query_uv+direction*step/max(W,H))*axis_dimension)`, wrapped or clamped per axis. Replace winner only on strictly smaller metric. No extra refinement pass.

For a valid winner, force final distance zero if the query mask is inside. `within=d<dmax`; falloff=`max(fma(-d,inv,1),0)`. Invalid winner gives within=false and falloff zero. Without source output gray falloff, or colour `(v,v,v,1)`. With source and combine clear (F bit 1=0), output nearest seed source within radius, original source otherwise. With combine set, gray outputs nearest seed R*falloff; colour uses nearest RGB inside radius, original RGB outside, and nearest seed alpha*falloff. If no seed exists, the candidate is the original pixel and falloff is zero: gray evaluates `original_R*0` and colour outputs `(original_RGB,original_alpha*0)`. For finite channels these products are zero (with IEEE signed-zero behavior); nonfinite channels follow ordinary multiplication and can yield NaN. Gray then replicates R and sets alpha one. No lower bound is imposed on distance or pixelsize in CPU arithmetic. Zero/NaN/extreme parameter parity is **Open O12**. [Reference: `ops.rs::distance`.]

## 0x2c — curves

Resize input. Blob contains one gray curve or five colour curves (master RGB, R, G, B, A). Each starts with u32 own point count followed by six f32 per point `(x,y,left_x,left_y,right_x,right_y)`. CPU uses min(count,record maximum), advancing by this clamped count. Empty curve is identity.

For input x between adjacent points, use cubic Bézier controls `(a.position,a.right,b.left,b.position)`. Before first point append synthetic `(0,first.y)` with identical outgoing control; after last append `(1,last.y)` with identical incoming control. Solve x(t) with 40 f32 bisections in [0,1], comparing cubic x(mid)<x, then evaluate y at midpoint. No key sorting occurs. The cubic is `u*u*u*p0+3*u*u*t*p1+3*u*t*t*p2+t*t*t*p3` in order.

Build 1025 f32 samples at i/1024. For a nonempty curve clamp input to [0,1], set t=x*1024, i=min(floor(t),1023), fraction=round((t−i)*256)/256, return `lut[i]+(lut[i+1]-lut[i])*fraction`. Empty curves return x directly. Colour first applies master to RGB independently then individual RGBA curves. D bit 1 replaces input with `(x_pixel+.5)/W` for every channel. D bit 0 is parsed as repeat metadata but does not change the CPU LUT clamp; unmeasured repeat behavior is **Open O23**. [References: `ops.rs::curve`, `bezier_curve`.]

## A. Exact smooth-gradient LUT algorithm

The following self-contained Rust algorithm fixes arithmetic order and iteration limits. Inputs are stable-sorted `(position,RGBA)` keys and unconstrained smoothness w. It requires only ordinary arrays/vectors and the numeric conventions in SPEC.md. [Reference: `ops.rs::build_v8_lut`.]

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
