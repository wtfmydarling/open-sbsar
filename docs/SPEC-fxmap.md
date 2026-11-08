# open-sbsar draft 1 — FX-Maps, text, SVG and embedded decoders

This chapter supplements [SPEC.md](SPEC.md) and uses its numeric conventions and Open register. Source citations refer to the default CPU implementations. Detailed decoder listings at the end are part of this specification so that external font/JPEG implementations are not required to reproduce the supported profile.

## 1. FX-Map record and tree encoding

Node record layout is in SPEC.md. The flags' pattern-input count is **six bits**, 10–15. Background kind 3 is a node index; other kinds use C f32 values/function. Hurst and opacitymult default to 1; background defaults to zero (transparent for colour). Evaluation order is background, hurst, opacitymult, renderroi, and tilingroi only if renderroi exists. Render/tiling ROI vectors are `(x0,x1,y0,y1)`, not `(x0,y0,x1,y1)`. Tiling ROI defaults `(0,1,0,1)`. Flags bit 26 culling is not used by the CPU (**Open O17**). [References: `program.rs::parse_node`; `fxmap.rs::render`.]

Every tree record begins with u32 W: bits 0–2 kind; bit 3 seed inheritance; bits 4–5 seed-slot kind. Pointers are block-relative, zero means absent child. Tree kinds supported are 0 quadrant, 1 switch, 3 iterate; others are **Open O22**.

### 1.1 Quadrant

W bits 6–7 select source: 0 none, 1 generated pattern, 2 input image (3 unmeasured, **Open O17**). Bits 8–11 select generated pattern or static input index. Bit 16 means four distinct child pointers; bit 17 means one pointer reused for all four children (bit 16 wins if both set). Bit 18 selects max for gray / alpha blending for colour. Two-bit slots start at 19 opacity, 21 branch offset, 23 frame offset, 25 pattern size, 27 rotation, 29 supplement. Bit 31 adds an image-flags word immediately after W.

Image flags bits 0–1 describe dynamic image index (CPU consumes a function when bit 1 is set); bits 2–3 filter: 0 bilinear+mips, 1 bilinear, 2 nearest. Filter 3 currently falls through to level-zero bilinear (**Open O17**). Bit 4 marks image premultiplication but is not applied by the CPU (**Open O17**).

After optional image flags, a parameter-block pointer is present when seed kind is nonzero, any of the six ordinary slot kinds is nonzero, or image-flags bit 1 is set. Child pointers follow. At the parameter block consume seed, opacity C words, branch float2, frame float2, size float2, rotation float, supplement float, dynamic image-index function pointer, in that order and only for present slots. Ordinary tree slot kind 1 consumes constants, 2 consumes function pointer, 0/3 consume nothing. [Reference: `fxmap.rs::Ctx::quadrant`, `slot_with`, `seed_slot`.]

### 1.2 Switch and iterate

After W consume seed slot when present, then selector/count function pointer when W bit 7 is set. Switch static selector is bit 6; append two child pointers for false/true respectively. Iterate static count is the entire upper **24 bits** (`W>>8`), not 16 bits. Bit 6 means two child pointers stored in order output1, output0; otherwise only output1. Dynamic count is float first component max(0,value), truncating/saturating to u32. [Reference: `fxmap.rs::Ctx::node_body`.]

## 2. Traversal and random state

Start depth 0, centre (.5,.5), branch (0,0), seed node_seed&0xffffff, number zero. All tree functions share variables initialized by the enclosing node. Each parameter function restarts its own stream at the current seed; parameter evaluation order is record order. Tree `$pos` is the current traversal centre **before** the current quadrant applies its branch/frame offsets. `$number` is iteration number, `$depth` counts quadrant descent, `$depthpow2=2^depth`.

Seed-slot literal is raw u32. A seed function uses the node LCG, all 24 random bits. An **integer** result is numerically converted to f32 and then its **float bit pattern** becomes seed. A float result currently converts f32→i64→u32; external meaning is **Open O17**. On a quadrant with a parameter block, absent seed means zero; replace seed by k when inheritance bit is clear, or add k when set, masking to 24 bits. A quadrant without a parameter block leaves seed unchanged. Switch/iterate replace/add only when a seed slot is present. Selector/count functions see the resulting seed. [References: `fxmap.rs::seed_slot`, `quadrant`, `node_body`.]

All tree RNG recurrences are `s=(A*s+C)&0xffffff`. Ordinary draws are `(s>>1)/8388608`; seed-slot draws are s/16777216.

| Function slot | A | C |
|---|---:|---:|
| opacity | 3626811 | 51137 |
| branch offset | 14154961 | 50989 |
| frame offset | 1823937 | 50707 |
| pattern size | 5812301 | 50411 |
| rotation | 3470325 | 50131 |
| supplement | 12382630 | 50647 |
| image index | 1664525 | 7271263 |
| iterate count | 9501597 | 50591 |
| switch selector | 14464988 | 50111 |
| seed function | 16294893 | 10493233 |

A switch evaluates only the selected child with unchanged traversal geometry. An iterate runs output0 once first, if present, using its own seed/number; then output1 count times. Before each output1 call advance seed by `(8532665,14533)` and set number to zero-based iteration index. Nested iterates replace number rather than flattening it. [References: `fxmap.rs` RNG constants, `QuadEnv::rand`, `Ctx::node_body`.]

For a quadrant, apply branch offset by adding it to centre **immediately in f32**, before children; it is not scaled by cell size. Frame offset affects only this draw. Draw centre=`(updated_centre+branch)+frame`; default frame zero. Pattern size defaults cell size `2^-depth`; explicit size components multiply by cell size (scalar size broadcasts). Rotation and supplement default zero, opacity one. A missing source still traverses children.

After recording current draw, children visit top-left, top-right, bottom-left, bottom-right. Child depth d=parent depth+1 and centre offsets are `(±2^-d/2,±2^-d/2)`. Advance the following recurrence before each child, even if its pointer is zero, starting from parent seed:

| Child depth | A | C |
|---:|---:|---:|
| 1 | 1691414 | 52627 |
| 2 | 5547014 | 52673 |
| 3 | 1289321 | 52711 |
| 4 | 10611682 | 52747 |
| 5 | 2603697 | 52807 |
| 6 | 13013189 | 52859 |
| 7 | 14482087 | 52249 |
| 8 | 4839483 | 52289 |
| 9 | 3906728 | 52321 |
| 10 | 4193611 | 52379 |
| 11 | 4656553 | 52453 |
| ≥12 | 3558434 | 52511 |

The ≥12 reuse is CPU behavior; arbitrary deeper graphs and shift limits remain **Open O08**. [Reference: `fxmap.rs::Ctx::quadrant`, `CHILD`.]

Compute structural maximum quadrant depth over every tree branch, including untaken switch branches. Switch and iterate edges do not increment depth. Weight at depth d is 1 when hurst=0, otherwise `hurst^d / sum(k=0..max_depth,hurst^k)` with f32 summation ascending. Multiply by opacitymult. Negative/extreme hurst is not clamped in the CPU.

Evaluate functions in traversal order, but replay draws in stable groups: generated patterns ascending by kind, then images. Image draws retain traversal order unless **any** image quadrant anywhere in the tree (including untaken branches) has a dynamic index; then stable-sort images by clamped input index. Variable mutations are not reordered with draws. Negative dynamic image indices saturate to zero; indices beyond the last input clamp to the last. No connected image inputs means no image draw. [References: `fxmap.rs::max_depth`, `weight_calc`, `scan_groups`, `render`, `draw_inner`.]

## 3. Pattern geometry and coverage

Skip a stamp when either centre or size component or rotation is nonfinite, or either size is exactly zero. The active code does not impose a 1e−7 minimum. Use f32 angle `a=-rotation*TAU`, ca=cos(a), sa=sin(a) for pattern coordinates. Relative to centre, `u=(ca*dx-sa*dy)/(size.x/2)`, `v=(sa*dx+ca*dy)/(size.y/2)`. Negative size flips an axis.

Coverage uses a separate f64 rotation `a64=-f64(rotation)*TAU64`, half sizes hx/hy and extents `ex=abs(ca64*hx)+abs(sa64*hy)`, `ey=abs(sa64*hx)+abs(ca64*hy)`. Compute f64 local coordinates Lx/Ly and bounding coordinates Bx=dx/ex, By=dy/ey. With T=1e−6, include a sample iff `Bx>-1+T && Bx<=1+T && By>-1+T && By<=1+T && abs(Lx)<=1+T && abs(Ly)<=1+T`. Thus bounding-box left/top ties are excluded and right/bottom included. Pattern values still use f32 local coordinates. Hardware raster and nearest-image tie residuals are **Open O16**. [References: `fxmap.rs::Cover::new`, `outside64`, `Ctx::draw_inner`.]

Without ROI, enumerate unwrapped pixel coordinates around the stamp, compute their centres relative to stamp, test coverage, then wrap destination indices on tiled axes or discard outside untiled axes. Preserve row-first order and repeated wrapped contributions for stamps spanning multiple tiles. The CPU bounds use `ext=max(abs(size.x),abs(size.y))*.7072`, X from `floor((cx-ext)*W-1)` through `ceil((cx+ext)*W+1)` inclusive, similarly Y.

With ROI R, output pixel samples FX space `(R.x0+(x+.5)/W*(R.x1-R.x0), R.y0+(y+.5)/H*(R.y1-R.y0))`. For each tiled axis with positive tiling period, enumerate stamp copies from `floor((tile_lo-(centre+ext))/period)` through `ceil((tile_lo+period-(centre-ext))/period)`, ascending; Y copies outside, X inside. Untiled axes use only offset zero. Reject pixel FX coordinates outside the half-open tiling ROI on a tiled axis. Compute copy centres in f32; coverage pixel coordinates are recomputed in f64 using f32 ROI span cast to f64. Zero/negative ROI spans and extreme tiling remain **Open O12**. [Reference: `fxmap.rs::Ctx::draw_inner`.]

### 3.1 Generated patterns

For covered local u,v, define r²=u²+v², r=sqrt(r²), B=max(1−r²,0), S(x)=clamp(x,0,1)²*(3−2*clamp(x,0,1)), supplement s.

| Kind | Pattern value |
|---:|---|
| 0 square | 1 |
| 1 disc | r<=1 ? 1 : 0 |
| 2 paraboloid | B |
| 3 bell | B^4 |
| 4 gaussian | max(exp(−11.1*r²)−exp(−11.1),0) |
| 5 thorn | max(1−r,0)^4 |
| 6 pyramid | 1−max(abs(u),abs(v)) |
| 7 brick | s<=0 ? 1 : S((1−abs(u))/s)*S((1−abs(v))/s) |
| 8 gradation | (u+1)*.5*(1−max(s,0)) |
| 9 waves | formula below |
| 10 half bell | B^4*factor below |
| 11 ridged bell | B^4*(1−abs(u))²; supplement ignored |
| 12 crescent | 6.75*r²*B²*factor below |
| 13 capsule | max(1−20*(max(abs(u)−.77*max(s,0),0)²+v²),0) |
| 14 cone | max(1−r,0) |
| 15 gradation offset | (u+1)*.5+s |

Waves: `waves=3+12*max(s,0); cycle=rem_euclid(.5*waves*u,1); half=min(cycle,1-cycle); x=4*min(half,.5-half); a=1.223936f32; wave=1-a*x*x+(a-1)*x*x*x*x; carrier=(half<.25?wave:-wave); p=B*(carrier+1)*.5`. This is not ordinary cosine.

Half bell: d=1−max(s,0); factor=1−S(u/d) for d>0, S((u−d)/−d) for d<0, or `(u<=0)` for d=0. Crescent uses same d: factor=0 for d=0, otherwise 1−S((u+1)/(2*d)). [Reference for all patterns: `fxmap.rs::pattern`.]

### 3.2 Image patterns and mip selection

Build each input's mip chain in its own node storage format. Sample selected mip at `((u+1)/2,(v+1)/2)` against transparent black. Filter 2 is nearest at level zero; filter 1 is coupled bilinear level zero; filter 0 chooses a mip then coupled bilinear. Use original input format for interpolation even at higher mip levels. Gray in-bounds image samples have alpha one. Half border sampling applies half ties-away rounding described in SPEC.md.

The default mip model is a measured fit, **Open O15**, especially for elongated rotated stamps. To reproduce the CPU, set `scale=1` without ROI, else max absolute ROI spans. `tx=inputW/(abs(size.x)/scale*W)`, `ty=inputH/(abs(size.y)/scale*H)`. Let c=abs(cos(a)), s=abs(sin(a)). Define H(a,b)=max(abs(a),abs(b))*P(min/max), zero if max=0. P is piecewise linear through:

```
(0,1),(.04366,1.01432),(.08749,1.02998),(.13165,1.04505),
(.17633,1.06010),(.22169,1.07573),(.26795,1.09200),(.31530,1.10717),
(.36397,1.12431),(.41421,1.14231),(.46631,1.16005),(.52057,1.17761),
(.57735,1.19769),(.63707,1.21922),(.70021,1.23976),(.76733,1.26835),
(.83910,1.30082),(.91633,1.33629),(1,1.37383)
```

Set cols=max(H(c*tx,s*ty),H(s*tx,c*ty)), rows=H(c*max(tx,ty),s*max(tx,ty)). Blend weight at folded angle atan2(min(c,s),max(c,s)) in degrees is piecewise linear through `(0,0),(27,0),(30,.3),(35,.7),(40,.89),(42.5,.94),(45,1)`. LOD=log2(cols+blend*(rows−cols)); level=max(round(LOD),0), clamped to last available mip. [References: `fxmap.rs::HW_LOD_P`, `ROW_BLEND`, `hw_len`, `row_blend`, `Ctx::draw_inner`, `splat`.]

## 4. FX compositing and storage

Generated pattern sample is `(1,1,1,p)`; image sample is RGBA. Let op be quadrant opacity vector, weight=depth_weight*opacitymult. Gray colour-image input contributes zero; gray image input uses **two f32 subtractions** `p=1-(1-sample.r)`. Generated gray uses pattern p. Gray contribution=`op.r*p*weight`, except gradation offset contribution=`op.r*(p-supplement)*weight+supplement`. NaN contribution skips draw. Max blend takes max(destination,contribution); additive sums.

UNORM8 gray prequantizes background, converts each contribution with fixed-point UNORM8 storage, and maxes integer codes or adds codes saturating to 255. Other gray formats accumulate in f32; half truncates once at end. All then receive final node-format conversion.

Colour uses alpha factor `a=op.a*sample.a*weight` (absent opacity alpha defaults 1). NaN alpha, opacity RGB or sample channel skips the draw. Higher-precision colour additive RGB is `dst+op.rgb*sample.rgb*a`; alpha-blend RGB is `dst+(op.rgb*sample.rgb-dst)*a`; both alpha=`dst.a+a*(1-dst.a)`. No clamp for floating accumulation. Higher-precision additive generated gradation offset instead uses `core=op.a*(p-supplement)*weight`, RGB+=op.rgb*core+supplement, alpha+=(core+supplement)*(1−alpha). Remaining combinations/nonfinite parity are **Open O18**.

UNORM8 colour prequantizes background and, per stamp, premultiplied fragment `(op.rgb*sample.rgb*a,a)` using fixed-point storage. For each channel d=dst*(1−src.a) if alpha blend or alpha channel, otherwise d=dst. Store `round(clamp(src+d,0,1)*255)/255` after each layer. This layer round is ordinary nearest, unlike fragment conversion. Half colour also accumulates in f32 then truncates to half at end. [References: `fxmap.rs::Ctx::splat`, `render`; `image.rs::unorm`, `half_trunc`.]

## 5. SVG stream

At stream pointer read words: command=word&7, count_or_payload=word>>3. Command 0 terminates; command 3 sets opaque colour with low payload byte red, next green, next blue, divided by 255 (initial colour white). Gray uses arithmetic mean RGB. Command 1 consumes count packed points and draws a **triangle strip**, each sliding window of three points; command 2 consumes points and draws consecutive line segments. Point word packs u16 X then u16 Y divided by 65536. Snap each coordinate to 1/256 output pixel using nearest ties-away rounding, in f64. Initial background comes from optional image, literal colour, or zero.

For each fill triangle, discard zero area, orient edges so interior cross products are positive, and test pixel centres. A zero edge test is owned iff edge dy<0 or dy=0 and dx>0. Covered pixels replace background with current colour; no fill antialiasing. Strokes are one-pixel diamond-exit line strips: minimum L1 distance from segment to pixel centre must be <.5, and endpoint must not itself lie at L1 distance <.5. Compute the minimum at segment endpoints and parameters where segment X or Y equals centre, retaining parameters in [0,1].

Command 4 consumes count*8 gradient-stop bytes, command 5 count*4 geometry bytes. The CPU skips both without rendering their gradient semantics: **Open O19**. Unknown command reports unsupported and stops. A one-million-command guard is an implementation limit (**Open O08**). Old stream dialects and complete raster parity are also **Open O19**. [Reference: `ops.rs::svg`; `reader.py::parse_record`.]

## 6. Text and fonts

Text flags low two bits equal 2 select a string input UID; otherwise string data is u32 code-point count followed by u32 UTF-32 code points. CPU limits embedded strings to 4096 points. Font data is u32 opaque ID, u32 byte length, then embedded font. Missing font, unsupported font or missing string input leaves background and records unsupported. Start with optional resized background or zero. Evaluate colour (ones), position (0,0), matrix (identity), fontsize (.25), then alignment if dynamic. Alignment static 0 left,1 centre,2 right; dynamic rounds then clamps to 0…2.

Font tables use **big-endian** fields internally. The decoder in §A supports TrueType glyf and OTTO/CFF, cmap formats 4/12, hmtx advances and horizontal format-0 kern subtables. It does not apply GPOS shaping. Collections select their last font; whether every renderer selects last rather than a particular face is **Open O20**. Decoder limits and unsupported outline/charstring variants are likewise **Open O20**, not universal format constraints.

Let sc=fontsize/(hhea.ascent−hhea.descent), in UV per font unit. Baseline anchor is `.5+position`; split at LF, ignore CR. Baseline step=`fontsize*(ascent-descent+lineGap)/(ascent-descent)`. Each line's pen advances with hmtx plus unrounded pair kern. Compute union of glyph ink X bounds at these pens (empty glyph bounds=(0,0)); line width=(hi−lo)*sc. For centre/right subtract half/all width from the **pen origin**, without also subtracting lo.

Matrix maps output displacement to text displacement, column-major. Transform outlines by its inverse about the anchor. If abs(det)<=1e−12 substitute identity inverse. For glyph (fontX,fontY), text position is `(line_origin+pen*sc+fontX*sc,baseline-fontY*sc)`, then inverse-transform and scale by output W/H. Flatten curves in font units at tolerance `.35 / (sc * max(min(W*hypot(inv0,inv2),H*hypot(inv1,inv3)),1e-12))`; use the recursive outline algorithms in §A, without hinting or pixel snapping.

For whole-block clipping, union all nonempty lines' **unaligned** ink bounds lo_all,hi_all, align that union as one block, and pair its X endpoints with first-line ascent and last-line descent Y bounds. Transform all four corners; take their output bounding box. Inclusive clip columns floor(left*W)…ceil(right*W)−1; inclusive rows floor(top*H)…floor(bottom*H). This is not simply clipping each line to its own width.

Rasterize each glyph independently. For tiling, process original then copies with dy=−1,0,1 and dx=−1,0,1 ascending, omitting original and disallowed axes, retaining copies intersecting output; offset the clip box with each copy. Further-away copies are not generated (**Open O20**). Intersect each glyph raster rectangle with the clip box **before rasterizing**, so local Y origin changes. Bounds: floor(min glyph) limited by clip lower; ceil(max glyph)+1 limited by clip exclusive upper.

Coverage uses f64 signed-area accumulation with nonzero fill, specified by §A `rasterize`. In particular an edge starting above local Y=0 initializes X at edge Y=−1, not at zero; preserve this measured top-clip behavior. Quantize each glyph coverage by `min(floor(a*255+.5),255)/255`, then combine glyphs `coverage=coverage+a*(1-coverage)`. Finally each output channel is `background+(colour-background)*coverage`. This mixes colour alpha as another channel; it is not an extra alpha-over operation. Tiny-font width differences and unsupported shaping are **Open O20**. [References: `ops.rs::text`; `text.rs::Font`, `flatten_stb`, `rasterize`.]

## A. Embedded font decoder and coverage algorithms

The following Rust listings are self-contained decoder algorithms. They define byte layout, subroutine biases, outline flags, recursion limits, curve flattening and raster-area accumulation precisely. Comments referring to experiments are omitted; only the default raster branch is retained. Unused alternative flattening algorithms are omitted. Source: `text.rs::Font`, `cff_index`, `cff_dict`, `flatten_stb`, `rasterize`. The use of checked/default reads in individual helpers does not define general malformed-font acceptance (Open O08/O20).

```rust
use std::collections::HashMap;

pub struct Font<'a> {
    d: &'a [u8],
    tables: HashMap<[u8; 4], (usize, usize)>,
    pub upem: f64,
    loca_long: bool,
    nglyphs: usize,
    nhm: usize,
    cmap: Option<(usize, u16)>,
    pub ascent: f64,
    pub descent: f64,
    pub line_gap: f64,

    cff: Option<Cff>,
}

struct Cff {
    charstrings: Vec<(usize, usize)>,
    gsubrs: Vec<(usize, usize)>,
    lsubrs: Vec<(usize, usize)>,
}

fn cff_index(d: &[u8], o: usize) -> (Vec<(usize, usize)>, usize) {
    let n = u16be(d, o) as usize;
    if n == 0 { return (vec![], o + 2); }
    let osz = d.get(o + 2).copied().unwrap_or(1) as usize;
    let off = |i: usize| (0..osz).fold(0usize, |a, k| a << 8 | d.get(o + 3 + i * osz + k).copied().unwrap_or(0) as usize);
    let data = o + 3 + (n + 1) * osz - 1;
    ((0..n).map(|i| (data + off(i), data + off(i + 1))).collect(), data + off(n))
}

fn cff_dict(d: &[u8], a: usize, b: usize) -> HashMap<u16, Vec<f64>> {
    let (mut m, mut st, mut i) = (HashMap::new(), vec![], a);
    while i < b {
        let b0 = d[i];
        match b0 {
            0..=21 => {
                let op = if b0 == 12 { i += 1; 1200 + d.get(i).copied().unwrap_or(0) as u16 } else { b0 as u16 };
                m.insert(op, std::mem::take(&mut st));
                i += 1;
            }
            28 => { st.push(i16be(d, i + 1) as f64); i += 3; }
            29 => { st.push(u32be(d, i + 1) as i32 as f64); i += 5; }
            30 => {

                i += 1;
                while i < b && d[i] & 0x0F != 0x0F && d[i] >> 4 != 0x0F { i += 1; }
                i += 1;
                st.push(0.0);
            }
            32..=246 => { st.push(b0 as f64 - 139.0); i += 1; }
            247..=250 => { st.push((b0 as f64 - 247.0) * 256.0 + d.get(i + 1).copied().unwrap_or(0) as f64 + 108.0); i += 2; }
            251..=254 => { st.push(-(b0 as f64 - 251.0) * 256.0 - d.get(i + 1).copied().unwrap_or(0) as f64 - 108.0); i += 2; }
            _ => i += 1,
        }
    }
    m
}

#[derive(Clone, Copy, Debug)]
enum Vx { Move(i32, i32), Line(i32, i32), Cubic(i32, i32, i32, i32, i32, i32) }

fn u16be(d: &[u8], o: usize) -> u16 { d.get(o..o + 2).map_or(0, |b| u16::from_be_bytes([b[0], b[1]])) }
fn i16be(d: &[u8], o: usize) -> i16 { u16be(d, o) as i16 }
fn u32be(d: &[u8], o: usize) -> u32 { d.get(o..o + 4).map_or(0, |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]])) }

impl<'a> Font<'a> {
    pub fn parse(d: &'a [u8]) -> Option<Font<'a>> {

        let base = if u32be(d, 0) == 0x7474_6366 {
            let nf = u32be(d, 8) as usize;
            u32be(d, 12 + 4 * nf.saturating_sub(1)) as usize
        } else { 0 };
        let ver = u32be(d, base);

        if ver != 0x0001_0000 && ver != 0x7472_7565 && ver != 0x4F54_544F { return None; }
        let n = u16be(d, base + 4) as usize;
        let mut tables = HashMap::new();
        for i in 0..n {
            let r = base + 12 + 16 * i;
            let tag = [d[r], d[r + 1], d[r + 2], d[r + 3]];
            tables.insert(tag, (u32be(d, r + 8) as usize, u32be(d, r + 12) as usize));
        }
        let head = tables.get(b"head")?.0;
        let maxp = tables.get(b"maxp")?.0;
        let hhea = tables.get(b"hhea")?.0;
        let mut f = Font {
            d, upem: u16be(d, head + 18) as f64, loca_long: i16be(d, head + 50) != 0,
            nglyphs: u16be(d, maxp + 4) as usize, nhm: u16be(d, hhea + 34) as usize, cmap: None,
            ascent: i16be(d, hhea + 4) as f64, descent: i16be(d, hhea + 6) as f64, line_gap: i16be(d, hhea + 8) as f64, tables, cff: None,
        };
        if let Some(&(c, _)) = f.tables.get(b"CFF ") {

            let hsize = d.get(c + 2).copied().unwrap_or(4) as usize;
            let (_, o) = cff_index(d, c + hsize);
            let (tops, o) = cff_index(d, o);
            let (_, o) = cff_index(d, o);
            let (gsubrs, _) = cff_index(d, o);
            if let Some(&(ta, tb)) = tops.first() {
                let top = cff_dict(d, ta, tb);
                let cs = top.get(&17).and_then(|v| v.first()).map(|&x| c + x as usize);
                let mut lsubrs = vec![];
                if let Some(p) = top.get(&18).filter(|v| v.len() >= 2) {
                    let (psz, poff) = (p[0] as usize, c + p[1] as usize);
                    let pd = cff_dict(d, poff, poff + psz);
                    if let Some(sub) = pd.get(&19).and_then(|v| v.first()) {
                        lsubrs = cff_index(d, poff + *sub as usize).0;
                    }
                }
                if let Some(cs) = cs {
                    f.cff = Some(Cff { charstrings: cff_index(d, cs).0, gsubrs, lsubrs });
                }
            }
        }
        if let Some(&(c, _)) = f.tables.get(b"cmap") {
            let nt = u16be(d, c + 2) as usize;
            let mut best: Option<(i32, usize, u16)> = None;
            for i in 0..nt {
                let r = c + 4 + 8 * i;
                let (pid, eid, off) = (u16be(d, r), u16be(d, r + 2), u32be(d, r + 4) as usize);
                let fmt = u16be(d, c + off);
                let score = match (pid, eid, fmt) { (3, 10, 12) => 4, (0, _, 12) => 3, (3, 1, 4) => 2, (0, _, 4) => 1, _ => 0 };
                if score > 0 && best.map_or(true, |b| score > b.0) { best = Some((score, c + off, fmt)); }
            }
            f.cmap = best.map(|b| (b.1, b.2));
        }
        Some(f)
    }

    pub fn glyph_index(&self, cp: u32) -> usize {
        let d = self.d;
        let Some((t, fmt)) = self.cmap else { return 0 };
        if fmt == 12 {
            let ng = u32be(d, t + 12) as usize;
            for i in 0..ng {
                let g = t + 16 + 12 * i;
                let (s, e, sg) = (u32be(d, g), u32be(d, g + 4), u32be(d, g + 8));
                if cp >= s && cp <= e { return (sg + cp - s) as usize; }
            }
            return 0;
        }

        if cp > 0xFFFF { return 0; }
        let segx2 = u16be(d, t + 6) as usize;
        let ends = t + 14;
        let starts = ends + segx2 + 2;
        let deltas = starts + segx2;
        let ranges = deltas + segx2;
        for i in 0..segx2 / 2 {
            let end = u16be(d, ends + 2 * i) as u32;
            if cp > end { continue; }
            let start = u16be(d, starts + 2 * i) as u32;
            if cp < start { return 0; }
            let delta = u16be(d, deltas + 2 * i) as u32;
            let ro = u16be(d, ranges + 2 * i) as usize;
            if ro == 0 { return ((cp + delta) & 0xFFFF) as usize; }
            let gi = u16be(d, ranges + 2 * i + ro + 2 * (cp - start) as usize) as u32;
            return if gi == 0 { 0 } else { ((gi + delta) & 0xFFFF) as usize };
        }
        0
    }

    pub fn advance(&self, g: usize) -> f64 {
        let Some(&(h, _)) = self.tables.get(b"hmtx") else { return self.upem * 0.5 };
        let i = g.min(self.nhm.saturating_sub(1));
        u16be(self.d, h + 4 * i) as f64
    }

    pub fn kern(&self, left: usize, right: usize) -> f64 {
        let d = self.d;
        let Some(&(k, _)) = self.tables.get(b"kern") else { return 0.0 };
        if u16be(d, k) != 0 { return 0.0; }
        let nt = u16be(d, k + 2) as usize;
        let (mut o, mut v) = (k + 4, 0.0);
        let key = (left as u32) << 16 | right as u32;
        for _ in 0..nt {
            let (len, cov) = (u16be(d, o + 2) as usize, u16be(d, o + 4));

            if cov >> 8 == 0 && cov & 7 == 1 {
                let np = u16be(d, o + 6) as usize;
                let (mut lo, mut hi) = (0usize, np);
                while lo < hi {
                    let m = (lo + hi) / 2;
                    let r = o + 14 + 6 * m;
                    let km = u32be(d, r);
                    if km == key { v += i16be(d, r + 4) as f64; break; }
                    if km < key { lo = m + 1 } else { hi = m }
                }
            }
            if len == 0 { break; }
            o += len;
        }
        v
    }

    fn cff_vertices(&self, g: usize) -> Option<Vec<Vx>> {
        let cff = self.cff.as_ref()?;
        let &(a, b) = cff.charstrings.get(g)?;
        let d = self.d;
        let bias = |n: usize| if n < 1240 { 107 } else if n < 33900 { 1131 } else { 32768 };
        let (gb, lb) = (bias(cff.gsubrs.len()), bias(cff.lsubrs.len()));
        let mut v: Vec<Vx> = vec![];
        let (mut x, mut y, mut fx, mut fy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut started = false;
        let mut st: Vec<f32> = vec![];
        let mut nstems = 0usize;
        let mut calls: Vec<(usize, usize)> = vec![];
        let (mut i, mut end) = (a, b);
        let close = |v: &mut Vec<Vx>, x: f32, y: f32, fx: f32, fy: f32, started: bool| {
            if started && (fx != x || fy != y) { v.push(Vx::Line(fx as i32, fy as i32)); }
        };
        macro_rules! line { ($dx:expr, $dy:expr) => {{ x += $dx; y += $dy; v.push(Vx::Line(x as i32, y as i32)); }} }
        macro_rules! curve { ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr) => {{
            let (c1x, c1y) = (x + $a, y + $b);
            let (c2x, c2y) = (c1x + $c, c1y + $d);
            x = c2x + $e; y = c2y + $f;
            v.push(Vx::Cubic(c1x as i32, c1y as i32, c2x as i32, c2y as i32, x as i32, y as i32));
        }} }
        macro_rules! moveto { ($dx:expr, $dy:expr) => {{
            close(&mut v, x, y, fx, fy, started);
            x += $dx; y += $dy; fx = x; fy = y; started = true;
            v.push(Vx::Move(x as i32, y as i32));
        }} }
        let mut guard = 0;
        loop {
            guard += 1;
            if guard > 200_000 { break; }
            if i >= end {
                match calls.pop() { Some((ri, re)) => { i = ri; end = re; continue; } None => break }
            }
            let b0 = d[i];
            i += 1;
            match b0 {
                32..=246 => st.push(b0 as f32 - 139.0),
                247..=250 => { st.push((b0 as f32 - 247.0) * 256.0 + d[i] as f32 + 108.0); i += 1; }
                251..=254 => { st.push(-(b0 as f32 - 251.0) * 256.0 - d[i] as f32 - 108.0); i += 1; }
                28 => { st.push(i16be(d, i) as f32); i += 2; }
                255 => { st.push(u32be(d, i) as i32 as f32 / 65536.0); i += 4; }
                1 | 3 | 18 | 23 => { nstems += st.len() / 2; st.clear(); }
                19 | 20 => { nstems += st.len() / 2; st.clear(); i += (nstems + 7) / 8; }
                21 => { let n = st.len(); if n >= 2 { moveto!(st[n - 2], st[n - 1]); } st.clear(); }
                22 => { if let Some(&dx) = st.last() { moveto!(dx, 0.0); } st.clear(); }
                4 => { if let Some(&dy) = st.last() { moveto!(0.0, dy); } st.clear(); }
                5 => { for k in (0..st.len() / 2 * 2).step_by(2) { line!(st[k], st[k + 1]); } st.clear(); }
                6 | 7 => {
                    let mut horiz = b0 == 6;
                    for k in 0..st.len() { if horiz { line!(st[k], 0.0) } else { line!(0.0, st[k]) } horiz = !horiz; }
                    st.clear();
                }
                8 => { let mut k = 0; while k + 5 < st.len() { curve!(st[k], st[k + 1], st[k + 2], st[k + 3], st[k + 4], st[k + 5]); k += 6; } st.clear(); }
                24 => {
                    let n = st.len(); let mut k = 0;
                    while k + 5 < n.saturating_sub(2) { curve!(st[k], st[k + 1], st[k + 2], st[k + 3], st[k + 4], st[k + 5]); k += 6; }
                    if k + 1 < n { line!(st[k], st[k + 1]); }
                    st.clear();
                }
                25 => {
                    let n = st.len(); let mut k = 0;
                    while k + 1 < n.saturating_sub(6) { line!(st[k], st[k + 1]); k += 2; }
                    if k + 5 < n { curve!(st[k], st[k + 1], st[k + 2], st[k + 3], st[k + 4], st[k + 5]); }
                    st.clear();
                }
                26 | 27 => {
                    let n = st.len(); let mut k = 0; let mut f = 0.0;
                    if n % 2 == 1 { f = st[0]; k = 1; }
                    while k + 3 < n {
                        if b0 == 27 { curve!(st[k], f, st[k + 1], st[k + 2], st[k + 3], 0.0) } else { curve!(f, st[k], st[k + 1], st[k + 2], 0.0, st[k + 3]) }
                        f = 0.0; k += 4;
                    }
                    st.clear();
                }
                30 | 31 => {
                    let n = st.len(); let mut k = 0; let mut vert = b0 == 30;
                    while k + 3 < n {
                        let last = if n - k == 5 { st[k + 4] } else { 0.0 };
                        if vert { curve!(0.0, st[k], st[k + 1], st[k + 2], st[k + 3], last) } else { curve!(st[k], 0.0, st[k + 1], st[k + 2], last, st[k + 3]) }
                        k += 4; vert = !vert;
                    }
                    st.clear();
                }
                10 | 29 => {
                    let Some(n) = st.pop() else { break };
                    let subrs = if b0 == 10 { &cff.lsubrs } else { &cff.gsubrs };
                    let idx = n as i64 + if b0 == 10 { lb } else { gb };
                    let Some(&(sa, sb)) = (idx >= 0).then(|| subrs.get(idx as usize)).flatten() else { break };
                    calls.push((i, end));
                    i = sa; end = sb;
                }
                11 => { match calls.pop() { Some((ri, re)) => { i = ri; end = re; } None => break } }
                14 => { close(&mut v, x, y, fx, fy, started); started = false; break; }
                12 => {
                    let op = d.get(i).copied().unwrap_or(0); i += 1;
                    let s = &st;
                    match op {
                        35 if s.len() >= 13 => { let s = s.clone(); curve!(s[0], s[1], s[2], s[3], s[4], s[5]); curve!(s[6], s[7], s[8], s[9], s[10], s[11]); }
                        34 if s.len() >= 7 => { let s = s.clone(); curve!(s[0], 0.0, s[1], s[2], s[3], 0.0); curve!(s[4], 0.0, s[5], -s[2], s[6], 0.0); }
                        36 if s.len() >= 9 => { let s = s.clone(); curve!(s[0], s[1], s[2], s[3], s[4], 0.0); curve!(s[5], 0.0, s[6], s[7], s[8], -(s[1] + s[3] + s[7])); }
                        37 if s.len() >= 11 => {
                            let s = s.clone();
                            let dx = s[0] + s[2] + s[4] + s[6] + s[8];
                            let dy = s[1] + s[3] + s[5] + s[7] + s[9];
                            let (dx6, dy6) = if dx.abs() > dy.abs() { (s[10], -dy) } else { (-dx, s[10]) };
                            curve!(s[0], s[1], s[2], s[3], s[4], s[5]); curve!(s[6], s[7], s[8], s[9], dx6, dy6);
                        }
                        _ => {}
                    }
                    st.clear();
                }
                _ => st.clear(),
            }
        }
        close(&mut v, x, y, fx, fy, started);
        Some(v)
    }

    pub fn flatten_cff(&self, g: usize, flat: f64) -> Option<Vec<Vec<(f64, f64)>>> {
        fn cubic(poly: &mut Vec<(f64, f64)>, p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32), f2: f32, n: u32) {
            let (dx0, dy0, dx1, dy1, dx2, dy2) = (p1.0 - p0.0, p1.1 - p0.1, p2.0 - p1.0, p2.1 - p1.1, p3.0 - p2.0, p3.1 - p2.1);
            let (dx, dy) = (p3.0 - p0.0, p3.1 - p0.1);
            let long = (dx0 * dx0 + dy0 * dy0).sqrt() + (dx1 * dx1 + dy1 * dy1).sqrt() + (dx2 * dx2 + dy2 * dy2).sqrt();
            let short = (dx * dx + dy * dy).sqrt();
            let fl = long * long - short * short;
            if n > 16 { return; }
            if fl > f2 {
                let m = |a: (f32, f32), b: (f32, f32)| ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
                let (p01, p12, p23) = (m(p0, p1), m(p1, p2), m(p2, p3));
                let (pa, pb) = (m(p01, p12), m(p12, p23));
                let mm = m(pa, pb);
                cubic(poly, p0, p01, pa, mm, f2, n + 1);
                cubic(poly, mm, pb, p23, p3, f2, n + 1);
            } else {
                poly.push((p3.0 as f64, p3.1 as f64));
            }
        }
        let vs = self.cff_vertices(g)?;
        let f2 = (flat * flat) as f32;
        let mut out: Vec<Vec<(f64, f64)>> = vec![];
        let mut cur = (0.0f32, 0.0f32);
        for v in vs {
            match v {
                Vx::Move(x, y) => { out.push(vec![(x as f64, y as f64)]); cur = (x as f32, y as f32); }
                Vx::Line(x, y) => { if let Some(p) = out.last_mut() { p.push((x as f64, y as f64)); } cur = (x as f32, y as f32); }
                Vx::Cubic(ax, ay, bx, by, x, y) => {
                    if let Some(p) = out.last_mut() { cubic(p, cur, (ax as f32, ay as f32), (bx as f32, by as f32), (x as f32, y as f32), f2, 0); }
                    cur = (x as f32, y as f32);
                }
            }
        }
        Some(out)
    }

    fn cff_xbounds(&self, g: usize) -> Option<(f64, f64)> {
        let vs = self.cff_vertices(g)?;
        let (mut lo, mut hi) = (i32::MAX, i32::MIN);
        for v in vs {
            let xs: &[i32] = &match v { Vx::Move(x, _) | Vx::Line(x, _) => [x, x, x], Vx::Cubic(a, _, b, _, x, _) => [x, a, b] };
            for &x in xs { lo = lo.min(x); hi = hi.max(x); }
        }
        (lo <= hi).then_some((lo as f64, hi as f64))
    }

    pub fn xbounds(&self, g: usize) -> Option<(f64, f64)> {
        if self.cff.is_some() { return self.cff_xbounds(g); }
        let (o, _) = self.glyph_range(g)?;
        Some((i16be(self.d, o + 2) as f64, i16be(self.d, o + 6) as f64))
    }

    fn glyph_range(&self, g: usize) -> Option<(usize, usize)> {
        if g >= self.nglyphs { return None; }
        let (loca, _) = *self.tables.get(b"loca")?;
        let (glyf, _) = *self.tables.get(b"glyf")?;
        let (a, b) = if self.loca_long { (u32be(self.d, loca + 4 * g) as usize, u32be(self.d, loca + 4 * g + 4) as usize) }
                     else { (u16be(self.d, loca + 2 * g) as usize * 2, u16be(self.d, loca + 2 * g + 2) as usize * 2) };
        if b <= a { return None; }
        Some((glyf + a, glyf + b))
    }

    pub fn outline(&self, g: usize, depth: u32) -> Vec<Vec<(f64, f64, bool)>> {
        let d = self.d;
        let Some((o, _)) = self.glyph_range(g) else { return vec![] };
        let nc = i16be(d, o);
        if nc >= 0 {
            let nc = nc as usize;
            let ends: Vec<usize> = (0..nc).map(|i| u16be(d, o + 10 + 2 * i) as usize).collect();
            let npts = ends.last().map_or(0, |&e| e + 1);
            let il = u16be(d, o + 10 + 2 * nc) as usize;
            let mut p = o + 12 + 2 * nc + il;
            let mut flags = Vec::with_capacity(npts);
            while flags.len() < npts {
                let f = d[p]; p += 1;
                flags.push(f);
                if f & 8 != 0 { let r = d[p]; p += 1; for _ in 0..r { flags.push(f); } }
            }
            flags.truncate(npts);
            let mut xs = Vec::with_capacity(npts);
            let mut v = 0i32;
            for &f in &flags {
                if f & 2 != 0 { let dx = d[p] as i32; p += 1; v += if f & 16 != 0 { dx } else { -dx }; }
                else if f & 16 == 0 { v += i16be(d, p) as i32; p += 2; }
                xs.push(v);
            }
            let mut ys = Vec::with_capacity(npts);
            v = 0;
            for &f in &flags {
                if f & 4 != 0 { let dy = d[p] as i32; p += 1; v += if f & 32 != 0 { dy } else { -dy }; }
                else if f & 32 == 0 { v += i16be(d, p) as i32; p += 2; }
                ys.push(v);
            }
            let mut out = vec![];
            let mut s = 0;
            for &e in &ends {
                out.push((s..=e).map(|i| (xs[i] as f64, ys[i] as f64, flags[i] & 1 != 0)).collect());
                s = e + 1;
            }
            return out;
        }
        if depth > 8 { return vec![]; }

        let mut out = vec![];
        let mut p = o + 10;
        loop {
            let fl = u16be(d, p); let gi = u16be(d, p + 2) as usize; p += 4;
            let (dx, dy) = if fl & 1 != 0 { let r = (i16be(d, p) as f64, i16be(d, p + 2) as f64); p += 4; r }
                           else { let r = (d[p] as i8 as f64, d[p + 1] as i8 as f64); p += 2; r };
            let (mut a, mut b, mut c, mut e) = (1.0, 0.0, 0.0, 1.0);
            let f2 = |o: usize| i16be(d, o) as f64 / 16384.0;
            if fl & 8 != 0 { a = f2(p); e = a; p += 2; }
            else if fl & 0x40 != 0 { a = f2(p); e = f2(p + 2); p += 4; }
            else if fl & 0x80 != 0 { a = f2(p); b = f2(p + 2); c = f2(p + 4); e = f2(p + 6); p += 8; }
            for cont in self.outline(gi, depth + 1) {
                out.push(cont.into_iter().map(|(x, y, on)| (a * x + c * y + dx, b * x + e * y + dy, on)).collect());
            }
            if fl & 0x20 == 0 { break; }
        }
        out
    }
}

pub fn flatten_stb(contours: &[Vec<(f64, f64, bool)>], flat: f64) -> Vec<Vec<(f64, f64)>> {
    fn tess(poly: &mut Vec<(f64, f64)>, a: (f64, f64), q: (f64, f64), b: (f64, f64), f2: f64, n: u32) {
        let (mx, my) = ((a.0 + 2.0 * q.0 + b.0) / 4.0, (a.1 + 2.0 * q.1 + b.1) / 4.0);
        let (dx, dy) = ((a.0 + b.0) / 2.0 - mx, (a.1 + b.1) / 2.0 - my);
        if n <= 16 && dx * dx + dy * dy > f2 {
            tess(poly, a, ((a.0 + q.0) / 2.0, (a.1 + q.1) / 2.0), (mx, my), f2, n + 1);
            tess(poly, (mx, my), ((q.0 + b.0) / 2.0, (q.1 + b.1) / 2.0), b, f2, n + 1);
        } else {
            poly.push(b);
        }
    }
    let f2 = flat * flat;
    let half = |a: f64, b: f64| ((a as i64 + b as i64) >> 1) as f64;
    let mut out = vec![];
    for c in contours {
        if c.len() < 2 { continue; }
        let n = c.len();
        let start = (0..n).find(|&i| c[i].2);
        let pts: Vec<(f64, f64, bool)> = match start {
            Some(s) => (0..n).map(|k| c[(s + k) % n]).collect(),
            None => {
                let m = (half(c[0].0, c[1].0), half(c[0].1, c[1].1), true);
                std::iter::once(m).chain(c.iter().copied().skip(1)).chain(std::iter::once(c[0])).collect()
            }
        };
        let mut poly = vec![(pts[0].0, pts[0].1)];
        let mut cur = (pts[0].0, pts[0].1);
        let mut ctrl: Option<(f64, f64)> = None;
        let m = pts.len();
        for k in 1..=m {
            let p = pts[k % m];
            if p.2 {
                match ctrl.take() {
                    Some(q) => tess(&mut poly, cur, q, (p.0, p.1), f2, 0),
                    None => poly.push((p.0, p.1)),
                }
                cur = (p.0, p.1);
            } else {
                if let Some(q) = ctrl {
                    let mid = (half(q.0, p.0), half(q.1, p.1));
                    tess(&mut poly, cur, q, mid, f2, 0);
                    cur = mid;
                }
                ctrl = Some((p.0, p.1));
            }
        }
        out.push(poly);
    }
    out
}

pub fn rasterize(polys: &[Vec<(f64, f64)>], w: usize, h: usize) -> Vec<f32> {
    let mut acc = vec![0.0f64; w * h + 1];
    let mut line = |p0: (f64, f64), p1: (f64, f64)| {
        if (p0.1 - p1.1).abs() < 1e-12 { return; }
        let (dir, a, b) = if p0.1 < p1.1 { (1.0, p0, p1) } else { (-1.0, p1, p0) };
        let dxdy = (b.0 - a.0) / (b.1 - a.1);
        let y0 = a.1.max(0.0);
        let y1 = b.1.min(h as f64);
        if y0 >= y1 { return; }
        let mut y = y0.floor() as usize;

        let mut x = a.0 + (if a.1 < 0.0 { -1.0 - a.1 } else { y0 - a.1 }) * dxdy;
        while (y as f64) < y1 && y < h {
            let ylo = (y as f64).max(y0);
            let yhi = ((y + 1) as f64).min(y1);
            let dy = yhi - ylo;
            let xnext = x + dxdy * dy;
            let d = dy * dir;
            let (xl, xr) = if x < xnext { (x, xnext) } else { (xnext, x) };
            let row = y * w;
            let x0f = xl.floor();
            let x0i = x0f as i64;
            let x1c = xr.ceil();
            if x1c <= x0f + 1.0 {

                let xm = 0.5 * (x + xnext) - x0f;
                if x0i < 0 { acc[row] += d; }
                else if (x0i as usize) < w {
                    acc[row + x0i as usize] += d - d * xm;
                    if (x0i as usize) + 1 < w { acc[row + x0i as usize + 1] += d * xm; }
                }
            } else {

                let s = 1.0 / (xr - xl);
                let x0r = xl - x0f;
                let a0 = 0.5 * s * (1.0 - x0r) * (1.0 - x0r);
                let x1i = x1c as i64;
                let x1r = xr - x1c + 1.0;
                let am = 0.5 * s * x1r * x1r;
                let put = |acc: &mut Vec<f64>, xi: i64, v: f64| {
                    if xi >= w as i64 { return; }
                    acc[row + xi.max(0) as usize] += v;
                };
                put(&mut acc, x0i, d * a0);
                if x1i == x0i + 2 {
                    put(&mut acc, x0i + 1, d * (1.0 - a0 - am));
                } else {
                    let a1 = s * (1.5 - x0r);
                    put(&mut acc, x0i + 1, d * (a1 - a0));
                    for xi in x0i + 2..x1i - 1 {
                        put(&mut acc, xi, d * s);
                    }
                    let a2 = a1 + (x1i - x0i - 3) as f64 * s;
                    put(&mut acc, x1i - 1, d * (1.0 - a2 - am));
                }
                put(&mut acc, x1i, d * am);
            }
            x = xnext;
            y += 1;
        }
    };
    for poly in polys {
        for i in 0..poly.len() {
            line(poly[i], poly[(i + 1) % poly.len()]);
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let mut s = 0.0;
        for x in 0..w {
            s += acc[y * w + x];
            out[y * w + x] = s.abs().min(1.0) as f32;
        }
    }
    out
}
```

## B. Embedded baseline JPEG decoder

This listing specifies marker parsing, Huffman decoding, restart handling, quantization, zigzag order, integer IDCT, chroma reconstruction and RGB conversion. It accepts 8-bit sequential grayscale/YCbCr JPEG in the implemented profile. Progressive, arithmetic and other unsupported encodings require a fallback whose exact behavior is **Open O21**. Integer IDCT constants are 2841, 2676, 2408, 1609, 1108, 565; row and column shifts and biases must remain as written. Chroma is replicated, with the measured odd-column adjustment for sub-128 values on a horizontally 2× subsampled plane. No transfer-function conversion follows RGB decoding. Source: `jpeg.rs::decode`, `row_idct`, `col_idct`, `Bits`.

```rust
pub struct Decoded {
    pub w: usize,
    pub h: usize,
    pub ch: usize,
    pub px: Vec<u8>,
}

#[derive(Clone, Default)]
struct Comp {
    id: u8,
    ssx: usize,
    ssy: usize,
    width: usize,
    height: usize,
    stride: usize,
    qt: usize,
    ac: usize,
    dc: usize,
    pred: i32,
    px: Vec<u8>,
}

const ZZ: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
];

const W1: i32 = 2841;
const W2: i32 = 2676;
const W3: i32 = 2408;
const W5: i32 = 1609;
const W6: i32 = 1108;
const W7: i32 = 565;

fn clip(x: i32) -> u8 {
    x.clamp(0, 255) as u8
}

fn row_idct(b: &mut [i32]) {
    let (mut x1, mut x2, mut x3, mut x4, mut x5, mut x6, mut x7) = (b[4] << 11, b[6], b[2], b[1], b[7], b[5], b[3]);
    if x1 | x2 | x3 | x4 | x5 | x6 | x7 == 0 {
        let t = b[0] << 3;
        b[..8].fill(t);
        return;
    }
    let mut x0 = (b[0] << 11) + 128;
    let mut x8 = W7 * (x4 + x5);
    x4 = x8 + (W1 - W7) * x4;
    x5 = x8 - (W1 + W7) * x5;
    x8 = W3 * (x6 + x7);
    x6 = x8 - (W3 - W5) * x6;
    x7 = x8 - (W3 + W5) * x7;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2);
    x2 = x1 - (W2 + W6) * x2;
    x3 = x1 + (W2 - W6) * x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    b[0] = (x7 + x1) >> 8;
    b[1] = (x3 + x2) >> 8;
    b[2] = (x0 + x4) >> 8;
    b[3] = (x8 + x6) >> 8;
    b[4] = (x8 - x6) >> 8;
    b[5] = (x0 - x4) >> 8;
    b[6] = (x3 - x2) >> 8;
    b[7] = (x7 - x1) >> 8;
}

fn col_idct(b: &[i32], c: usize, out: &mut [u8], stride: usize) {
    let g = |k: usize| b[8 * k + c];
    let (mut x1, mut x2, mut x3, mut x4, mut x5, mut x6, mut x7) = (g(4) << 8, g(6), g(2), g(1), g(7), g(5), g(3));
    if x1 | x2 | x3 | x4 | x5 | x6 | x7 == 0 {
        let t = clip(((g(0) + 32) >> 6) + 128);
        for y in 0..8 {
            out[y * stride] = t;
        }
        return;
    }
    let mut x0 = (g(0) << 8) + 8192;
    let mut x8 = W7 * (x4 + x5) + 4;
    x4 = (x8 + (W1 - W7) * x4) >> 3;
    x5 = (x8 - (W1 + W7) * x5) >> 3;
    x8 = W3 * (x6 + x7) + 4;
    x6 = (x8 - (W3 - W5) * x6) >> 3;
    x7 = (x8 - (W3 + W5) * x7) >> 3;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2) + 4;
    x2 = (x1 - (W2 + W6) * x2) >> 3;
    x3 = (x1 + (W2 - W6) * x3) >> 3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    let v = [x7 + x1, x3 + x2, x0 + x4, x8 + x6, x8 - x6, x0 - x4, x3 - x2, x7 - x1];
    for y in 0..8 {
        out[y * stride] = clip((v[y] >> 14) + 128);
    }
}

type Vlc = Vec<(u8, u8)>;

struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
    buf: u32,
    n: u32,
}

impl Bits<'_> {
    fn show(&mut self, k: u32) -> u32 {
        while self.n < k {
            let mut byte = 0xFFu32;
            if self.pos < self.d.len() {
                byte = self.d[self.pos] as u32;
                self.pos += 1;
                if byte == 0xFF {
                    let m = self.d.get(self.pos).copied().unwrap_or(0);
                    if m == 0 {
                        self.pos += 1;
                    } else if (0xD0..=0xD7).contains(&m) {

                    } else {
                        self.pos -= 1;
                        byte = 0xFF;
                        self.d = &self.d[..self.pos];
                    }
                }
            }
            self.buf = (self.buf << 8) | byte;
            self.n += 8;
        }
        (self.buf >> (self.n - k)) & ((1 << k) - 1)
    }
    fn skip(&mut self, k: u32) {
        self.show(k);
        self.n -= k;
    }
    fn get(&mut self, k: u32) -> u32 {
        let r = self.show(k);
        self.n -= k;
        r
    }
    fn align(&mut self) {
        self.n &= !7;
    }
    fn vlc(&mut self, t: &Vlc, code: &mut u8) -> Result<i32, String> {
        let v = self.show(16) as usize;
        let (len, c) = t[v];
        if len == 0 {
            return Err("bad huffman code".into());
        }
        self.skip(len as u32);
        *code = c;
        let k = (c & 15) as u32;
        if k == 0 {
            return Ok(0);
        }
        let mut val = self.get(k) as i32;
        if val < (1 << (k - 1)) {
            val += (-1 << k) + 1;
        }
        Ok(val)
    }
}

fn be16(d: &[u8], o: usize) -> usize {
    ((d[o] as usize) << 8) | d[o + 1] as usize
}

pub fn decode(d: &[u8]) -> Result<Decoded, String> {
    if d.len() < 4 || d[0] != 0xFF || d[1] != 0xD8 {
        return Err("not a JPEG".into());
    }
    let mut qt = [[0i32; 64]; 4];
    let mut vlc: Vec<Vlc> = vec![vec![]; 8];
    let mut comps: Vec<Comp> = vec![];
    let (mut width, mut height, mut mbw, mut mbh, mut ssxmax, mut ssymax) = (0, 0, 0, 0, 1, 1);
    let mut rst = 0usize;
    let mut o = 2;
    loop {
        if o + 4 > d.len() || d[o] != 0xFF {
            return Err("bad marker".into());
        }
        let m = d[o + 1];
        let len = be16(d, o + 2);
        let seg = &d[o + 4..(o + 2 + len).min(d.len())];
        match m {
            0xC0 | 0xC1 => {
                if seg[0] != 8 {
                    return Err("not 8-bit".into());
                }
                height = be16(seg, 1);
                width = be16(seg, 3);
                let n = seg[5] as usize;
                if n != 1 && n != 3 {
                    return Err(format!("{n} components"));
                }
                for i in 0..n {
                    let p = &seg[6 + 3 * i..];
                    let c = Comp { id: p[0], ssx: (p[1] >> 4) as usize, ssy: (p[1] & 15) as usize, qt: (p[2] & 3) as usize, ..Default::default() };
                    ssxmax = ssxmax.max(c.ssx);
                    ssymax = ssymax.max(c.ssy);
                    comps.push(c);
                }
                if n == 1 {
                    comps[0].ssx = 1;
                    comps[0].ssy = 1;
                    ssxmax = 1;
                    ssymax = 1;
                }
                let (mbx, mby) = (ssxmax * 8, ssymax * 8);
                mbw = width.div_ceil(mbx);
                mbh = height.div_ceil(mby);
                for c in comps.iter_mut() {
                    c.width = (width * c.ssx + ssxmax - 1) / ssxmax;
                    c.height = (height * c.ssy + ssymax - 1) / ssymax;
                    c.stride = mbw * c.ssx * 8;
                    if (c.width < 3 && c.ssx != ssxmax) || (c.height < 3 && c.ssy != ssymax) {
                        return Err("chroma too small".into());
                    }
                    c.px = vec![0; c.stride * mbh * c.ssy * 8];
                }
            }
            0xC2..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => return Err("not baseline".into()),
            0xC4 => {
                let mut p = 0;
                while p + 17 <= seg.len() {
                    let id = seg[p] as usize;
                    if id & 0xEC != 0 {
                        return Err("bad DHT".into());
                    }
                    let counts = &seg[p + 1..p + 17];
                    let mut t: Vlc = vec![(0, 0); 65536];
                    let (mut code, mut q, mut vi) = (0usize, 0usize, p + 17);
                    for (l, &cnt) in counts.iter().enumerate() {
                        let len = l + 1;
                        let span = 1usize << (16 - len);
                        for _ in 0..cnt {
                            let val = seg[vi];
                            vi += 1;
                            for k in 0..span {
                                t[q + k] = (len as u8, val);
                            }
                            q += span;
                            code += 1;
                        }
                        let _ = code;
                    }
                    vlc[(id >> 4) * 4 + (id & 3)] = t;
                    p = vi;
                }
            }
            0xDB => {
                let mut p = 0;
                while p + 65 <= seg.len() {
                    let i = seg[p] as usize;
                    if i & 0xFC != 0 {
                        return Err("16-bit quantisation table".into());
                    }
                    for k in 0..64 {
                        qt[i][k] = seg[p + 1 + k] as i32;
                    }
                    p += 65;
                }
            }
            0xDD => rst = be16(seg, 0),
            0xDA => {
                let n = seg[0] as usize;
                if n != comps.len() {
                    return Err("non-interleaved scan".into());
                }
                for i in 0..n {
                    let id = seg[1 + 2 * i];
                    let t = seg[2 + 2 * i];
                    let c = comps.iter_mut().find(|c| c.id == id).ok_or("scan component")?;
                    c.dc = (t >> 4) as usize & 3;
                    c.ac = (t & 3) as usize;
                }
                let start = o + 2 + len;
                let mut bits = Bits { d: &d[start..], pos: 0, buf: 0, n: 0 };
                let (mut mbx, mut mby, mut rstcount, mut nextrst) = (0, 0, rst, 0u32);
                let mut block = [0i32; 64];
                loop {
                    for c in comps.iter_mut() {
                        for sby in 0..c.ssy {
                            for sbx in 0..c.ssx {
                                block.fill(0);
                                let mut code = 0u8;
                                c.pred += bits.vlc(&vlc[c.dc], &mut code)?;
                                block[0] = c.pred * qt[c.qt][0];
                                let mut coef = 0usize;
                                loop {
                                    let v = bits.vlc(&vlc[4 + c.ac], &mut code)?;
                                    if code == 0 {
                                        break;
                                    }
                                    if code & 0x0F == 0 && code != 0xF0 {
                                        return Err("syntax".into());
                                    }
                                    coef += (code >> 4) as usize + 1;
                                    if coef > 63 {
                                        return Err("syntax".into());
                                    }
                                    block[ZZ[coef]] = v * qt[c.qt][coef];
                                    if coef >= 63 {
                                        break;
                                    }
                                }
                                for r in 0..8 {
                                    row_idct(&mut block[8 * r..8 * r + 8]);
                                }
                                let off = ((mby * c.ssy + sby) * c.stride + mbx * c.ssx + sbx) * 8;
                                let s = c.stride;
                                for col in 0..8 {
                                    col_idct(&block, col, &mut c.px[off + col..], s);
                                }
                            }
                        }
                    }
                    mbx += 1;
                    if mbx >= mbw {
                        mbx = 0;
                        mby += 1;
                        if mby >= mbh {
                            break;
                        }
                    }
                    if rst != 0 {
                        rstcount -= 1;
                        if rstcount == 0 {
                            bits.align();
                            let i = bits.get(16);
                            if i & 0xFFF8 != 0xFFD0 || i & 7 != nextrst {
                                return Err("restart marker".into());
                            }
                            nextrst = (nextrst + 1) & 7;
                            rstcount = rst;
                            for c in comps.iter_mut() {
                                c.pred = 0;
                            }
                        }
                    }
                }
                break;
            }
            _ => {}
        }
        o += 2 + len;
    }
    let ch = comps.len();
    let mut px = vec![0u8; width * height * ch];
    if ch == 3 {

        let chroma = |c: &Comp, x: usize, y: usize| {
            let (fx, fy) = (ssxmax / c.ssx, ssymax / c.ssy);
            let v = c.px[(y / fy) * c.stride + x / fx] as i32;
            if fx > 1 && x % fx != 0 && v < 128 { v - 1 } else { v }
        };
        for y in 0..height {
            for x in 0..width {
                let yy = (comps[0].px[y * comps[0].stride + x] as i32) << 8;
                let cb = chroma(&comps[1], x, y) - 128;
                let cr = chroma(&comps[2], x, y) - 128;
                let k = (y * width + x) * 3;
                px[k] = clip((yy + 359 * cr + 128) >> 8);
                px[k + 1] = clip((yy - 88 * cb - 183 * cr + 128) >> 8);
                px[k + 2] = clip((yy + 454 * cb + 128) >> 8);
            }
        }
    } else {
        for y in 0..height {
            px[y * width..(y + 1) * width].copy_from_slice(&comps[0].px[y * comps[0].stride..y * comps[0].stride + width]);
        }
    }
    Ok(Decoded { w: width, h: height, ch, px })
}
```
