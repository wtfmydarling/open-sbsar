# open-sbsar format specification — draft 1

This draft specifies the measured compiled archive format and the default CPU evaluation profile. It covers observed program versions 2, 3, 4, 5, 6, 8 and 9. Version 7 is **Open O01**. The target renderer generation is 14.1; format version and renderer generation are independent.

The specification comprises this file, [node operations](SPEC-ops.md), [function VM](SPEC-vm.md), and [FX-Maps and embedded graphics](SPEC-fxmap.md). Algorithm listings in those files are part of the specification, not external dependencies. Source citations name implementation files and functions rather than machine paths. The only evidence used is the research log, CPU engine, reader and writer. Later measurements supersede earlier ones; executable default CPU behavior supersedes conflicting comments and notes. No experimental configuration is required or normative.

Compatibility is not a claim of complete equivalence to every proprietary renderer. Rules marked **Open** identify unmeasured behavior, incomplete support, or residual approximations. The register at the end distinguishes these from fully specified CPU behavior.

## 1. Conventions

A word is an unsigned 32-bit little-endian integer. A halfword is u16 little-endian. Bit zero is least significant; bit intervals include both endpoints. Hexadecimal opcodes below are the even low-byte values, not the ordinal obtained by dividing by two. `C` is 4 channels for colour, 1 for gray. An offset is in bytes. Unless explicitly file-relative, pointers are relative to the program block at file offset `0x34`. Node references are indices in the node table, never byte offsets.

`f32` denotes IEEE binary32; reinterpretation of a word as f32 does not perform numerical conversion. Arithmetic is f32, in the written order, with separate multiply and add roundings, unless an explicit f64 computation or `fma(a,b,c)` is given. `fma` rounds `a*b+c` once. `round` rounds to nearest with ties away from zero; `floor` rounds toward negative infinity. `clamp(x,a,b)` preserves NaN. Float `min`/`max` return the non-NaN operand when only one operand is NaN. Float-to-integer casts truncate toward zero, saturate out-of-range results, and map NaN to zero. Unsigned LCG and hash arithmetic wraps modulo 2^32 before masking. `rem_euclid(x,n)` gives a nonnegative remainder. Ordinary float remainder `%` retains the dividend's sign. Preserve signed zero and subnormals. Transcendental last-bit portability is **Open O02**. [References: `image.rs`, `vm.rs::scalar_binary`, `ops.rs::pixel_seed`.]

## 2. Archive and XML interface

An `.sbsar` is a 7z archive. The measured uncompressed variant uses Copy; the minimal writer emits Copy. The measured layout contains `assemblies/content/0000/<name>.sbsasm` and its sibling `<name>.xml`. These are archive member names, not filesystem requirements. Bitmaps, fonts and lookup data are embedded in the program. For a robust reader, pairing by member stem and checking assembly UID agreement is recommended. The reference loader instead runs recursive `7z e -so -r` extraction separately for `*.sbsasm` and `*.xml`, taking each stdout as-is: it neither pairs members nor checks UIDs, and multiple matches concatenate. A bare `.sbsasm` argument reads sibling `.xml`; an XML read failure supplies an empty string. Multiple-program interpretation, encrypted archives, other compression methods and alternative directory layouts are **Open O03**. [References: `NOTES.md`, Container; `main.rs::load`, `member`; `write.py::sbsar`.]

XML is UTF-8, with root `sbsdescription`, attributes `formatversion="2.1"`, `asmuid`, `cookerbuild`, `content="full"`, and a declared `sbsdescription.dtd`. The measured writer does not ship a DTD. A reader need not retrieve one. A `graphs` element carries `count`; each `graph` has `pkgurl` and `label`, `sbspresets`, `guigroups`, `outputs`, and `inputs`. Input/output lists carry counts. Numeric `uid` binds metadata to binary entries; `identifier` supplies the public name. Names do not replace UIDs in program instructions.

An image output has `uid`, `identifier`, `type="5"`, `format`, `width`, `height`, `mipmaps`, and an `outputgui` with label/type and `channels`. Usage, display label, groups, widgets and presets are XML metadata. The minimal description uses format 12 for gray and 0 for RGBA, width/height 256, mipmaps 0, and empty channels. These metadata dimensions do not override node size functions. Input attributes include `uid`, `identifier`, `type`, `default`, `alteroutputs`, `alternodes`. `$randomseed` is an integer input; `$outputsize` is a log2-size vector when exposed. There is no required universal UID for either. A hidden integer UID `0x80000001` occurs in interface tables. Preset schema, complete GUI schema and image override binding are **Open O04**. Preserve unrecognized XML when rewriting. [References: `write.py::description`; `main.rs::input_uids`, `output_names`, `apply_sets`; `NOTES.md`, Output table and Sizes.]

Binary numeric defaults initialize execution. String type 6 defaults come from XML. Decode `amp`, `lt`, `gt`, `quot`, `apos`, decimal numeric and hexadecimal numeric character references; preserve unknown entities verbatim. The reference command-line tool resolves `--set identifier=value` through XML identifiers to UIDs using literal `<input ` and double-quoted attribute splits. An identifier absent from that XML lookup is ignored; this includes `$outputsize` on a fixed-size program without that input. For a UID already in the string-default map, store the raw override text without entity decoding. Otherwise split on commas and parse i32 when the current numeric value is integer or absent, f32 otherwise; an unparsable component becomes zero. Replace the entire value, including its component count, rather than preserving the declared width. This is reference-tool override behavior, not additional binary syntax. [References: `main.rs::input_uids`, `apply_sets`, `xml_unescape`; `program.rs::Program::parse`.]

## 3. Program envelope and interface

| File offset | Contents |
|---|---|
| 0x00 | ASCII `SBAM` |
| 0x04 | flags word, version = `(word >> 16) & 255` |
| 0x08 | cooker build, matching XML |
| 0x0c | assembly UID, matching XML |
| 0x10 | total file byte length |
| 0x14 | 0x1c |
| 0x18 | 0 |
| 0x1c | total length minus 0x1c |
| 0x20 | 0x00010002 |
| 0x24 | 0 |
| 0x28 | 1 |
| 0x2c | block byte length P |
| 0x30 | 2 |
| 0x34 | block, P bytes |
| 0x34+P | interface and trailer |

The constants are measured values and writer requirements for this profile; meanings of reserved words and other header flag bits are **Open O05**. [References: `program.rs::Program::parse`; `write.py::assemble`; `NOTES.md`, .sbsasm layout.]

The interface is `u16 output_count; u16 input_count; output_count*u32 output_uid; input_count*(u32 type,u32 uid)`. Output entry k belongs positionally to output UID k in this list; the parser neither sorts nor checks ordering. The writer preserves the caller's output order in both binary lists and XML. The writer sorts inputs by unsigned UID. Immediately afterward are four u32 trailer words: node count, node-table offset, output-table offset, defaults offset. The observed envelope ends there.

The block starts with a zero word. Data may precede the node table or occur among records. The node table is `node_count` offsets; its entries need not increase in physical order. Execute in table order. Output entries are `(u32 flags,u32 node_index)` in interface-list order, not sorted UID order. Known output flags: `0xc0` gray, `0x100` 16-bit integer, `0x400` f32, `0x500` half. `0x10000` marks image outputs in newer interfaces; the minimal older writer does not set it. Absence of that bit alone does not make an old output a value output. The reference CPU retains but ignores output flags; export uses the referenced node's image (see §6). Complete output-flag interpretation and non-image output export are **Open O06**. [References: `program.rs::Program::parse`; `reader.py::records_of`; `write.py::assemble`; `main.rs::main`; `NOTES.md`, Output table.]

Defaults are contiguous in input-table order, with no per-value tags:

| Interface type | Meaning | Words consumed |
|---|---|---:|
| 0,1,2,3 | float, float2, float3, float4 | 1,2,3,4 |
| 4 | signed int | 1 |
| 8,9,10 | int2,int3,int4 | 2,3,4 |
| 5, version ≥8 | image default RGBA f32 | 4 |
| 5, version <8 | image | 0 |
| 6 | string, XML default | 0 |
| others | **Open O07** | CPU consumes 0 |

Defaults offset zero means no binary default block. In that case records can extend to block end. Record bounds can be found by sorting distinct record starts together with table offsets, defaults (if nonzero), and P. Shared/external bitmap data must be accounted for independently of record boundaries. Function and blob pointers are absolute within the block, even when their data is physically within a record. [References: `program.rs::words_of_type`, `Program::parse`; `reader.py::records_of`, `bitmap_span`.]

A robust reader checks bounds, integer overflow, counts, alignment and reference validity before allocation. The CPU parser assumes well-formed input and is not a malformed-file validation standard. Unknown opcodes or unrecognized layouts must be reported as unsupported, not guessed. For an unknown node opcode, `parse_node` returns an error and `Program::parse` rejects the whole program; this is not a per-node runtime diagnostic or placeholder. Malformed-file acceptance behavior is **Open O08**. [References: `program.rs::parse_node`, `Program::parse`.]

## 4. Node words and slots

`opcode=opword&0xfe`, `colour=opword&1`, `log2w=(opword>>8)&15`, `log2h=(opword>>12)&15`, `B=(opword>>16)&255`, `D=opword>>24`.

B bit 0 adds a size-function pointer; bit 3 is normally set; bits 4–5 select storage 0 UNORM8, 1 UNORM16, 2 half, 3 f32; bits 6–7 select the seed slot. Other B bits are **Open O05**. For filters, D bits 0 and 1 mean wrap X and wrap Y; other nodes reuse them for other purposes. [Reference: `program.rs::parse_node`, `Node::format`, `Node::size`.]

`S(field,bit,n)` consumes according to `(field>>bit)&3`: 0 no bytes/default; 1 n literal words; 2 one function offset; 3 one image-node index. Kind 3 is meaningful only where an image is accepted; its meaning in arbitrary scalar slots is **Open O07**. Numeric constants are f32 except seeds, booleans and expressly packed data. `K` below means seed `S(B,6,1)` followed by a size-function pointer if B bit 0 is set. `P` means pixelsize `S(D,2,2)`. Slots consume bytes only when present. Parameter evaluation order can differ from byte order.

The following is the complete parsed record body after opword. `F` denotes a leading flags word, `i` a node index, `ptr` a block pointer. `C=colour?4:1`.

| Opcode | Sequential body |
|---|---|
| 00 | i, key count, [v≥8 and D&2: smoothness f32], key ptr, K |
| 02 | F, background i, foreground i, K, opacity S(F,4,1), [opacity kind 3: multiplier S(F,9,1)], [F&0x40: two crop words] |
| 04 | F, i, K, [F&31=30: mip function ptr], matrix S(F,6,4), offset S(F,25,2), matte S(F,28,4) |
| 06 gray | i, K, weights S(D,0,4), matte S(D,2,1) |
| 06 colour | F, first i, second i, K |
| 08 | F, tree ptr, ((F>>10)&63) pattern indices, K, background S(F,0,C), hurst S(F,2,1), opacitymult S(F,4,1), renderroi S(F,6,4), tilingroi S(F,8,4) |
| 0a | stream ptr, [D&4: background i], K, [D&2: C literal colour words] |
| 0c | K, colour S(D,0,C) |
| 0e | [v≥9: F], source i, gradient i, K, P, intensity S(D,5,1) |
| 10 | F, source i, height i, K, P, intensity S(F,1,1), angle S(F,3,1), highlight S(F,5,4), shadow S(F,7,4) |
| 12 | F, source i, gradient i, K, P, intensity S(F,0,1), shiftangle S(F,2,1) |
| 14 | i, K, P, intensity S(D,4,1) |
| 16 | F, i, K, P, intensity S(F,0,1), angle S(F,2,1) |
| 18 | F, source i, amount i, K, P, intensity S(F,1,1), angle S(F,3,1), amount-offset S(F,7,1) |
| 1a | i, K, P, intensity S(D,4,1) |
| 1c | i, K, hue S(D,0,1), saturation S(D,2,1), luminosity S(D,4,1) |
| 1e | F, i, K, inlow S(F,0,C), inhigh S(F,2,C), inmid S(F,4,C), outlow S(F,6,C), outhigh S(F,8,C) |
| 20 | data ptr or UID, [D&8 and D&1: alpha JPEG ptr], K |
| 22 | F, discarded word, string ptr or UID, font ptr, K, [background kind ((F>>2)&3)=3: i], colour S(F,4,C), matrix S(F,10,4), position S(F,6,2), fontsize S(F,8,1), [alignment ((F>>12)&3)=3: function ptr] |
| 24 | F, i, K, P, intensity S(F,0,1), inversedy boolean(F,2), input2alpha boolean(F,4) |
| 26 | value i, gradient i, K, coordinate S(D,1,1) |
| 28 | F, (F&65535) input indices, K, function ptr |
| 2a | F, mask i, [F&1: source i], K, P, distance S(F,2,1), [F&0x40: mode function ptr] |
| 2c | i, maximum point count, curve ptr, K |

Normal booleans are exceptions: kind 1 means true with **no literal word**, kind 2 consumes a function pointer, kinds 0/3 consume nothing and act absent. Text background precedes parameter slots and its matrix precedes position; older reader ordering is superseded. Text colour-slot placement remains **Open O09**, although the CPU layout above is exact. Warp's v9 extra flags word is consumed but unused (**Open O10**). [Reference for every row: `program.rs::parse_node`; older diagnostic checks: `reader.py::parse_record`.]

## 5. Execution and state

For default rendering, required nodes are the transitive dependency closure of output nodes plus all value processors, following explicit image inputs and kind-3 entries in the node's named parameter list. A value processor has opcode `0x28` and `(F>>16)!=0` (see [SPEC-ops §0x28](SPEC-ops.md#0x28--pixel-and-value-processor)). Evaluate this closure in ascending table index; skip all other nodes, including their seed and size functions. `run_for` substitutes selected targets for the output roots; diagnostic `keep_all` mode includes every node. [Reference: `ops.rs::Runtime::run_inner`.]

A normal image input must precede its consumer. `0xffffffff` is allowed as a disconnected `0x28` processor input and supplies transparent black. The runtime checks only explicit `node.inputs` for indices at or after the consumer. Except for that processor sentinel, such an input produces a diagnostic and a 1×1 `(0,0,0,0)` placeholder, including alpha zero for gray. This path skips seed evaluation, size evaluation and storage conversion. Forward kind-3 parameter references are not covered by that check: accessing an uncomputed image panics (`input computed`), and an out-of-bounds index can panic on indexing. Their format validity remains **Open O08**; this fallback is not a cyclic execution model. [References: `ops.rs::Runtime::run_inner`, `img`, `input_at`.]

Clear node variables at each executed node. Cross-node variable persistence observed in older measurements is not modeled by this CPU profile (**Open O24**). If a size function exists, first evaluate the seed, then run the size function with RNG state `seed&0xffffff`, `$pos=(.5,.5)`, `$size=(256,256)`, and a fresh variable store. Take components `iv()[0]` and `iv()[1]`, without the node-parameter scalar broadcast; float components truncate with saturating i32 conversion and NaN→0. A numeric scalar has second component zero; Void/string gives `(0,0)`. Bool is an exception: `iv()` replicates 0 or 1, so false gives 1×1 and true gives 2×2. Clamp each exponent to 0…13, then use `width=1<<x`, `height=1<<y`. Retain the size function's variable writes. Without a function use the static nibbles unchanged. Set current size, evaluate the seed again, and execute the operation. Thus a seed function may run twice when a size function is present. A seed function sees the node's current dimensions through `$size` and `$sizelog2`: static nibble dimensions before size evaluation, size-function dimensions afterward. It starts at RNG state zero and shares node variables; its first component is converted to i32 then u32. A literal seed is a raw word; absent seed is zero. [References: `ops.rs::run_inner`, `seed`, `exec`; `vm.rs::Value::iv`.]

Each ordinary parameter function starts its own RNG stream at `seed&0xffffff`; parameter functions share variables, not an advancing RNG state. They see the current node dimensions. Scalar function results broadcast to four components when consumed as node parameters; vector results retain their components and trailing zeroes. Literal parameters do not gain this broadcast. An absent pixelsize is `(1,1)`. Evaluate parameters in the orders listed in the operation chapters, since variable mutation makes order observable. [References: `ops.rs::slot`, `param`, `pixelsize`.]

Node random stream: `s=(16294893*s+10493233)&0xffffff`; draw `s/16777216` as f32. Pixel stream: `s=(3626811*s+51137)&0xffffff`; draw `(s>>1)/8388608`. Pixel initial state hashes the **full** node seed and f32 position bits:

```
h=0
for word in [seed, bits(pos.x), bits(pos.y)]:
    h=h+word; h=h+(h<<10); h=h^(h>>6)   # wrapping u32
h=h+(h<<3); h=h^(h>>11); h=h+(h<<15)
s=h&0xffffff
```

Processor position is the pixel centre. Value processors use this hash with `(0,0)` for RNG initialization, although their exposed `$pos` is `(.5,.5)`. FX-Map streams are specified separately. [References: `ops.rs::NodeEnv::rand`, `PixEnv::rand`, `pixel_seed`, `pixel_processor`.]

## 6. Images, storage and sampling

Images are row-major RGBA f32 internally. Gray storage is represented as `(g,g,g,1)`, except the unconverted forward-input placeholder in §5. Apply the node's storage conversion after each executed image node, not just at final export. Gray conversion stores only R then replicates it and forces alpha one. Additional intermediate conversions are operation-specific.

The reference `render` command exports each output node's image as 16-bit PNG, applying `unorm(v,16)` regardless of output flags, including half/f32 flags. A value-processor output exports its 1×1 zero image after ordinary node storage conversion (gray alpha becomes one), not its written values. `save` has an explicit `.f32` destination extension path that writes little-endian u32 width/height followed by raw RGBA f32 pixels; this is independent of output flags and is not the default `render` export. [References: `main.rs::main`, `save`; `ops.rs::pixel_processor`, `run_inner`; `image.rs::Image::quantize`.]

For UNORM n=8 or 16, this is the exact storage rule:

```
k=n+4
v32=clamp(v,0,1)
i=u64(floor(f64(v32)*2^k))
code=(i*(2^n-1)+2^(k-1)-1)>>k
stored=f32(code)/f32(2^n-1)
```

NaN casts to zero; infinities clamp to endpoints. This is truncation to n+4 fixed-point bits followed by ties-down rounding, not `round(v*(2^n-1))`. [Reference: `image.rs::unorm`, `Image::quantize`.]

Half conversion is defined by the CPU bit algorithm, including edge behavior: nonfinite input is preserved; let `e=((bits>>23)&255)-127`. If e>15 return signed 65504; if e<−24 return signed zero. Otherwise keep `10-max(0,-14-e)` mantissa bits, clear the dropped bits and increment the retained field when the dropped part exceeds half an ulp, or equals it and the retained LSB is odd. Precisely, with `drop=23-keep` and cleared bit pattern r, the tie test is `(r>>drop)&1`. At e=−24 no mantissa bits remain, so this tests the f32 exponent-field LSB, which is 1: exact ties round upward in magnitude. This algorithm can round an e=15 input upward at the boundary; do not silently substitute a different saturating-half library. Half truncation uses the same bounds and simply clears dropped bits. FX-Map and gradient-map half targets truncate before the ordinary final store. FX image sampling half rounding instead uses ties away from zero: e<−25 → signed zero; e=−25 → signed 2^-24; otherwise add half a dropped ulp then clear dropped bits. Its mixed-exponent accuracy is **Open O11**. [References: `image.rs::half`, `half_trunc`, `half_sample`.]

### 6.1 Sampling primitives

At UV `(u,v)`, nearest fetches `floor(u*W),floor(v*H)`. Wrapped coordinates use Euclidean modulo per axis; nonwrapped coordinates clamp to edge unless a border is specified. A border replaces a texel when either nonwrapped axis is out of range. FX image borders are transparent black, including alpha zero for gray images outside their extent. A 1×1 image obeys the same rule. [References: `image.rs::fetch`, `fetch_or`, `fetch_border`, `nearest`, `nearest_border`.]

All bilinear paths start with `x=u*W-.5,y=v*H-.5`, floor to `(x0,y0)`, fetch a=(0,0), b=(1,0), c=(0,1), d=(1,1) relative to it. They differ as follows:

1. **Exact float:** fractions fx=x−x0, fy=y−y0. `top=a+(b-a)*fx; bottom=c+(d-c)*fx; result=top+(bottom-top)*fy`.
2. **Separable 1/256:** replace each fraction by `round(f*256)/256`, then use the same expression.
3. **Coupled texture:** round fractions as in 2, then `cross=round(fx*fy*256)/256`; weights `(1-fx-fy+cross,fx-cross,fy-cross,cross)`. UNORM sources convert each fetched value to `round(value*65535)` unsigned codes and use `((sum(code*integer_weight)+128)>>8)/65535`, with `integer_weight=weight*256`. Float sources use the ordered four-term weighted sum. Only FX border sampling additionally applies half-sampler rounding for half sources.

Generic image resizing uses path 2 at destination pixel centres with both edges clamped, without mips. Pixel-processor bilinear samples and most image filters use path 3. Value-processor and FX function bilinear samples use path 1. Seed, size and ordinary node-parameter functions have no image sources; VM samplers 51/52 therefore return transparent black regardless of filter. FX image stamps use path 3 with a transparent border. These are separate rules. Nonfinite coordinate behavior follows the arithmetic/cast rules; complete external-renderer agreement at NaN/Inf coordinates is **Open O12**. [References: `image.rs::bilinear_exact`, `bilinear_256`, `bilinear_or`, `bilinear_border`, `resized`; `ops.rs::NodeEnv::sample`, `PixEnv::sample`, `seed`, `slot`, `run_inner`; `fxmap.rs::QuadEnv::sample`.]

### 6.2 Mip generation

Include level zero unchanged; each next level halves each dimension greater than one, stopping at 1×1. Average the corresponding 1, 2 or 4 texels in row/column order. For UNORM16 reconstruct integer codes and use `(sum+count/2)/count` integer division, ties up, then the ordinary format store. UNORM8 uses the f32 average followed by the fixed-point storage rule. Half and f32 use f32 averaging and their respective stores. Per-axis halving uses the same format-dependent rules. Per-axis doubling samples the coupled texture primitive in the supplied source format and does not quantize its result. [References: `image.rs::mips`, `downsample_axes`, `upsample_axes_fmt`.]

## 7. Writing a minimal archive

The measured writer profile uses version 2, cooker build 1660783984 and a generated 31-bit assembly UID. Include a type-4 `$randomseed` interface input, UID `0x5b6a13f4` in the example profile, default zero. UIDs can differ provided XML and binary agree. Inputs sort by UID; defaults must contain the number of words required by each type, not simply one word per input.

Construct block: zero word; node offsets starting at block offset 4; output entries immediately after the node table; node records; default words. Append each record's aligned functions then aligned blobs, patching all pointers to block-relative byte offsets. Functions use the encoding in SPEC-vm and final padding to a multiple of four; literal operands require their own alignment. Blob padding is zero to a multiple of four. Records may be laid out in execution order for simplicity. Append the interface and four-word trailer. Fill P and file length, then the header constants in §3. [References: `write.py::Record`, `Func::bytes`, `assemble`.]

For a single gray constant 1×1 node, a concrete program can use these block words: `0,16,0xc0,0,0x0108000c,0x3f800000,0`. Node table offset=4, output table=8, defaults=24, P=28. Choose output UID 1. Interface halfwords are 1,1; interface words are `1,4,0x5b6a13f4`; trailer words are `1,4,8,24`. Total file length is 112 bytes. The node opword encodes gray uniform, static exponents zero, B=8, D=1. Match assembly/build UIDs and the chosen output UID in XML.

Emit the XML structure of §2, with an empty `sbspresets count="0"`, empty `guigroups`, image output and its GUI, and the randomseed input. Archive both members using 7z Copy. Compression implementation itself is outside this measured format specification (**Open O03**). The writer demonstrates a valid construction; it is not evidence that every metadata element is mandatory. Minimality beyond the measured writer and general graph round-trip writing remain **Open O13**.

## 8. Version and renderer differences

| Distinction | Rule |
|---|---|
| versions 2–4 | normal alpha always equals height |
| versions ≥5 | normal alpha follows input2alpha boolean |
| versions <8 | gradient float keys D bit 1, repeat bit 2; midpoint bit changes smoothing |
| versions ≥8 | gradient smoothness word D bit 1, float keys bit 2, repeat bit 3; four image-default words |
| version 9 | warp extra flags word; directional warp amount-offset slot observed |
| newer functions | loop 11, four-input construction 15, pow 54 supported by this VM |
| version 2/build zero | pixels can be between records; zero inputs/defaults pointer permitted |
| renderer 11.3 vs 14.1 | 11.3 rejects version 9; distance query quantization and metric behavior differ; narrow text clipping differs |

Default CPU distance emulates 14.1 for **all** program versions. The 11.3 distance model quantizes both seed and query positions to UNORM16; 14.1 quantizes only seed positions and supports the metric modes described in SPEC-ops. 11.3 narrow text targets one or two pixels wide exhibit row-spacing artifacts absent in the 14.1 profile. Measured additional differences: the proprietary SSE2 pixel evaluator uses a minimum side of 16 for pixel positions below that size, while this CPU profile uses actual dimensions; 11.3 returns zero for function power opcode 54; D3D11 processor wrap edges differ from Vulkan/OpenGL. Complete per-generation behavior beyond measured cases is **Open O14**. [References: `NOTES.md`, Format versions and later distance/text sections; `ops.rs::normal`, `distance`, `text`.]

### GPU parity note

The optional GPU backend is hybrid; the CPU remains normative. Exact parity requires preserving separate multiply/add roundings except explicit fma sites. Measured GPU differences include transcendental approximation, contraction and subnormal flushing; some operations still fall back to CPU. GPU coverage is not a claim of full format compatibility. [Reference: `GPU.md`, checkpoint 2026-11-08.]

## 9. Open register

The following IDs are the complete list of unresolved topics in this draft. An ID may be cited multiple times without creating another item.

| ID | Open topic |
|---|---|
| O01 | Version 7 and unobserved versions/layout variants. |
| O02 | Transcendental last bits, exceptional integer overflow outside explicit wrapping operations, and renderer hardware precision. |
| O03 | Full 7z codec/encryption profile, multiple assemblies and alternative member layouts. |
| O04 | Complete XML/preset/GUI schema and external image-input binding. |
| O05 | Reserved header/node bits and discarded words. |
| O06 | Complete output flags, value-output export and metadata format codes. |
| O07 | Unknown interface/VM types and invalid slot-kind combinations. |
| O08 | Malformed, cyclic or oversized-program behavior; implementation resource guards are not measured format limits. |
| O09 | Text colour-slot placement in unmeasured combinations. |
| O10 | Semantics of nonzero version-9 warp flags. |
| O11 | Half texture sampler internal mixed-exponent precision. |
| O12 | Complete nonfinite-coordinate and extreme-value renderer parity. |
| O13 | Strict archive minimality, full writer coverage and round-trip validation. |
| O14 | Complete renderer-generation compatibility beyond recorded differences. |
| O15 | FX image anisotropic mip selection: fitted default model, remaining aspect/angle residuals. |
| O16 | FX rasterization and nearest-image boundary ties on all hardware. |
| O17 | FX float-result seed-slot interpretation; culling/image-premultiplication flags. |
| O18 | FX gradation-offset max/8-bit colour combinations and extreme/nonfinite opacity parity. |
| O19 | SVG gradient commands 4/5, old stream dialects, complete edge raster parity. |
| O20 | Text shaping/font variants beyond the supplied decoder, tiny-font width residuals, >4096 code points and distant tiling copies. |
| O21 | JPEG fallback/progressive decoding exactness and unmeasured colour-space metadata. |
| O22 | Complete behavior of unknown VM opcodes and unknown system variables. |
| O23 | Dynamic-gradient/curve repeat and remaining filter precision/edge residuals beyond measured cases. |
| O24 | Cross-node variable persistence observed in older measurements but not modeled by the CPU profile. |

There are **24 Open items**. These limits are deliberately visible: a compatible implementation can reproduce the specified CPU profile without treating an approximation or a fallback as a measured universal rule.
