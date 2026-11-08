"""Write .sbsar files from scratch, from what NOTES.md says. No cooker involved.

A program is a list of nodes; each node is a Record: a list of words where Fn(k) stands for the block
offset of the record's k-th function and Data(k) for the block offset of the k-th embedded blob."""
import os, random, struct, subprocess, tempfile

COOKER_BUILD = 1660783984
SEED_UID = 0x5B6A13F4          # any uid works; it only has to match the XML
VERSION = 2                    # format version in the header (2..6 render with sbsrender 11.3)


def f32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


class Fn:
    def __init__(self, k): self.k = k


class Data:
    def __init__(self, k): self.k = k


class Record:
    def __init__(self, words, funcs=(), blobs=()):
        self.words = list(words)
        self.funcs = list(funcs)      # bytes of each function, appended after the words
        self.blobs = [b + b"\0" * (-len(b) % 4) for b in blobs]   # data (keys, curves), after the functions


def ins(op, typ, operands):
    return [op | typ << 6 | len(operands) << 10] + list(operands)


def function(instrs):
    """instrs: list of u16 lists from ins(). Returns bytes padded to 4."""
    u16 = [len(instrs)] + [x for i in instrs for x in i]
    b = struct.pack("<%dH" % len(u16), *u16)
    return b + b"\0" * (-len(b) % 4)


T_BOOL, T_F, T_F2, T_F3, T_F4, T_I, T_I2 = 1, 4, 5, 6, 7, 8, 9


class Func:
    """Build a function: each method appends an instruction and returns its register.
    The last instruction is the result."""

    def __init__(self):
        self.ins = []          # (header, operands)

    def _add(self, op, typ, operands):
        self.ins.append((op, typ, operands))
        return len(self.ins) - 1

    def const(self, *values):
        typ = {1: T_F, 2: T_F2, 3: T_F3, 4: T_F4}[len(values)]
        words = [f32(v) for v in values]
        return self._add(0, typ, ("const", words))

    def iconst(self, v):
        return self._add(0, T_I, ("const", [v & 0xFFFFFFFF]))

    def sysvar(self, index, typ=T_F):      # 0 $time, 8 $pos (float2), 10 $number, 12 $depth
        return self._add(1, typ, [index])

    def op(self, code, typ, *regs):        # 18 add 19 sub 20 mul 21 div 22 mod 36 floor 50 rand 13 vector ...
        return self._add(code, typ, list(regs))

    def bytes(self):
        u16 = [len(self.ins)]
        for op, typ, operands in self.ins:
            if isinstance(operands, tuple):          # a constant: its words 4-byte aligned
                words = operands[1]
                ops16 = []
                pos = 2 * (len(u16) + 1)              # byte position of the first operand
                if pos % 4:
                    ops16.append(0)
                for w in words:
                    ops16 += [w & 0xFFFF, w >> 16]
            else:
                ops16 = list(operands)
            u16.append(op | typ << 6 | len(ops16) << 10)
            u16 += ops16
        b = struct.pack("<%dH" % len(u16), *u16)
        return b + b"\0" * (-len(b) % 4)


def fn_load(uid, typ):
    """A function that loads a graph input: op 2, operands are the uid (4-byte aligned)."""
    # the first instruction sits at byte 2: its operands start at byte 4, already aligned
    return function([ins(2, typ, [uid & 0xFFFF, uid >> 16])])


def opword(op, color, size=(8, 8), byte2=0x08, byte3=0):
    return (op << 1 | color) | (size[1] << 4 | size[0]) << 8 | byte2 << 16 | byte3 << 24


# ---- nodes -------------------------------------------------------------------------------------

def uniform(color=None, gray=None, size=(8, 8)):
    if gray is not None:
        return Record([opword(6, 0, size, byte3=1), f32(gray)])
    return Record([opword(6, 1, size, byte3=1)] + [f32(c) for c in color])


def blend(fg, bg, mode=0, opacity=None, color=True, size=(8, 8)):
    flags = 0x100 | mode
    tail = []
    if opacity is not None:
        flags |= 0x10
        tail = [f32(opacity)]
    return Record([opword(1, int(color), size), flags, fg, bg] + tail)


def levels(inp, inlow=None, inhigh=None, inmid=None, outlow=None, outhigh=None, clamp=True, color=True, size=(8, 8)):
    flags, tail = 0, []
    for bit, v in ((0, inlow), (2, inhigh), (4, inmid), (6, outlow), (8, outhigh)):
        if v is not None:
            flags |= 1 << bit
            tail += [f32(x) for x in v]
    if not clamp:
        flags |= 0x400
    return Record([opword(15, int(color), size), flags, inp] + tail)


def hsl(inp, hue=None, saturation=None, luminosity=None, size=(8, 8)):
    b3, tail = 0, []
    for bit, v in ((0, hue), (2, saturation), (4, luminosity)):
        if v is not None:
            b3 |= 1 << bit
            tail.append(f32(v))
    return Record([opword(14, 1, size, byte3=b3), inp] + tail)


QUAD_SLOTS = [("opacity", 19), ("branchoffset", 21), ("frameoffset", 23), ("patternsize", 25),
              ("patternrotation", 27), ("patternsuppl", 29)]


def quadrant(pattern=None, children=None, same_child=None, image=None, blend_max=False, seed=None, inherit=True, **params):
    """A tree node for fxmap(): returns (word, param words, child list). image=k draws pattern input k.
    seed: a constant seed slot (first in the parameter block); inherit: bit 3."""
    w = 0x08 if inherit else 0
    if pattern is not None:
        w |= 1 << 6 | (pattern - 2) << 8
    if image is not None:
        w |= 2 << 6 | image << 8
    if blend_max:
        w |= 1 << 18
    pw = []
    if seed is not None:
        w |= 1 << 4
        pw.append(seed & 0xFFFFFFFF)
    for name, bit in QUAD_SLOTS:
        if name in params:
            v = params[name]
            if isinstance(v, Func):
                w |= 2 << bit
                pw.append(("func", v.bytes()))
                continue
            v = v if isinstance(v, (list, tuple)) else [v]
            w |= 1 << bit
            pw += [f32(x) for x in v]
    if children is not None:
        w |= 1 << 16
    elif same_child is not None:
        w |= 1 << 17
    return ("quad", w, pw, children if children is not None else ([same_child] if same_child else []))


def switch(value, child1, child0=None):
    """Switch with a constant selector: value 1 picks output1, 0 output0."""
    w = 1 | 0x08 | (0x40 if value else 0)
    return ("switch", w, [], [child0, child1])


def pixproc(func, inputs=(), color=False, seed=0, size=(8, 8), fmt=0):
    """Pixel processor: func (a Func) is evaluated per pixel; constant random seed."""
    return Record([opword(0x14, int(color), size, byte2=0x48 | fmt << 4), len(inputs)] + list(inputs) + [seed, Fn(0)],
                  [func.bytes()])


def gradient_map(inp, keys, color=True, midpoints=False, size=(8, 8), fmt=0, repeat=False):
    """keys: [(position, (r, g, b, a) or gray, [midpoint])] as 8/16-bit keys."""
    blob = b""
    for k in keys:
        blob += struct.pack("<H", round(k[0] * 65535))
        blob += bytes(round(c * 255) for c in k[1]) if color else struct.pack("<H", round(k[1] * 65535))
        if midpoints:
            blob += struct.pack("<H", round(k[2] * 65535))
    b3 = (1 if midpoints else 0) | (4 if repeat else 0)
    return Record([opword(0, int(color), size, byte2=0x08 | fmt << 4, byte3=b3), inp, len(keys), Data(0)], blobs=[blob])


def curve(inp, curves, color=False, size=(8, 8), fmt=0):
    """curves: one (gray) or five (luminance, R, G, B, A) lists of points ((x, y), (lx, ly), (rx, ry))."""
    blob = b""
    n = max(len(c) for c in curves)
    for pts in curves:                      # each curve takes the stride of the longest
        blob += struct.pack("<I", len(pts))
        for p, l, r in pts:
            blob += struct.pack("<6f", *p, *l, *r)
        blob += b"\0" * (24 * (n - len(pts)))
    return Record([opword(0x16, int(color), size, byte2=0x08 | fmt << 4), inp, n, Data(0)], blobs=[blob])


def iterate(n, child, child0=None, count=None, seed=None, inherit=True):
    """Iterate: child on output1 repeated n times (output0, if given, is the second slot); count,
    a Func, replaces n (its pointer sits right after the node word, after a constant seed slot)."""
    w = 3 | (0x08 if inherit else 0) | n << 8 | (0x40 if child0 is not None else 0) | (0x80 if count is not None else 0)
    pre = []
    if seed is not None:
        w |= 1 << 4; pre.append(("word", seed & 0xFFFFFFFF))
    if count is not None:
        pre.append(("func", count.bytes()))
    return ("iter", w, [], [child] + ([child0] if child0 is not None else []), pre)


def fxmap(tree, color=False, seed=0, size=(8, 8), fmt=0, hurst=None, background=None, inputs=(), renderroi=None, tilingroi=None, opacitymult=None):
    """FX-Map with a constant random seed; the tree is laid out right after the record head.
    hurst (roughness), background (gray value) and the render / tiling ROIs (4 floats each) are
    written as constants when given."""
    flags, extra = len(inputs) << 10, []
    if background is not None:
        flags |= 1; extra += [f32(b) for b in (background if isinstance(background, (list, tuple)) else [background])]
    if hurst is not None:
        flags |= 1 << 2; extra.append(f32(hurst))
    funcs = []
    if isinstance(opacitymult, Func):             # a function slot
        flags |= 2 << 4; funcs.append(opacitymult.bytes()); extra.append(Fn(len(funcs) - 1))
    elif opacitymult is not None:
        flags |= 1 << 4; extra.append(f32(opacitymult))
    for bit, roi in ((6, renderroi), (8, tilingroi)):
        if isinstance(roi, Func):                 # a function slot
            flags |= 2 << bit; funcs.append(roi.bytes()); extra.append(Fn(len(funcs) - 1))
        elif roi is not None:
            flags |= 1 << bit; extra += [f32(v) for v in roi]
    head = [opword(4, int(color), size, byte2=0x48 | fmt << 4, byte3=3), flags]
    # head: opword, flags, tree pointer, seed value; tree pointer is fixed in layout()
    words = []

    def emit(node):
        kind, w, pw, kids = node[:4]
        start = len(words)
        words.append(w)
        for x in (node[4] if len(node) > 4 else []):      # slots before the children
            if x[0] == "word":
                words.append(x[1])
            else:
                funcs.append(x[1]); words.append(Fn(len(funcs) - 1))
        ptr_at = None
        if pw:
            ptr_at = len(words); words.append(None)
        kid_at = len(words)
        words.extend([0] * len(kids))
        if pw:
            words[ptr_at] = ("rel", 4 + len(words))
            for x in pw:
                if isinstance(x, tuple) and x[0] == "func":
                    funcs.append(x[1]); words.append(Fn(len(funcs) - 1))
                else:
                    words.append(x)
        for i, k in enumerate(kids):
            if k is not None:
                words[kid_at + i] = ("rel", 4 + emit(k))
        return start

    emit(tree)
    n_head = len(head) + 2 + len(inputs) + len(extra)
    words = [("rel", w[1] - 4 + n_head) if isinstance(w, tuple) and w[0] == "rel" else w for w in words]
    rec = Record(head + [("rel", n_head)] + list(inputs) + [seed] + extra + words, funcs)
    return rec


# ---- program -----------------------------------------------------------------------------------

def assemble(nodes, outputs, inputs=(), defaults=None):
    """nodes: Records; outputs: [(uid, node index, flags)]; inputs: [(type, uid)] beyond $randomseed."""
    inputs = sorted([(4, SEED_UID)] + list(inputs), key=lambda t: t[1])
    n = len(nodes)
    node_table = 4
    out_table = node_table + 4 * n
    o = out_table + 8 * len(outputs)
    offs, recs = [], []
    for r in nodes:
        offs.append(o)
        words = []
        head_len = len(r.words)
        fstart = o + 4 * head_len
        fo, foffs = fstart, []
        for f in r.funcs:
            foffs.append(fo); fo += len(f)
        boffs = []
        for bl in getattr(r, "blobs", []):
            boffs.append(fo); fo += len(bl)
        for i, w in enumerate(r.words):
            if isinstance(w, Fn): w = foffs[w.k]
            elif isinstance(w, Data): w = boffs[w.k]
            elif isinstance(w, tuple) and w[0] == "rel": w = o + 4 * w[1]   # a word index inside this record
            words.append(w)
        b = struct.pack("<%dI" % len(words), *words) + b"".join(r.funcs) + b"".join(getattr(r, "blobs", []))
        recs.append(b)
        o += len(b)
    defaults_off = o
    dvals = defaults if defaults is not None else [0] * len(inputs)
    block = struct.pack("<I", 0) + struct.pack("<%dI" % n, *offs)
    for uid, idx, flags in outputs:
        block += struct.pack("<II", flags, idx)
    block += b"".join(recs) + struct.pack("<%dI" % len(dvals), *dvals)
    iface = struct.pack("<HH", len(outputs), len(inputs))
    iface += b"".join(struct.pack("<I", uid) for uid, _, _ in outputs)
    iface += b"".join(struct.pack("<II", t, u) for t, u in inputs)
    iface += struct.pack("<4I", n, node_table, out_table, defaults_off)
    asmuid = random.getrandbits(31)
    body = struct.pack("<5I", 0x10002, 0, 1, len(block), 2) + block + iface
    total = 0x1C + 4 + len(body)
    head = b"SBAM" + struct.pack("<6I", VERSION << 16, COOKER_BUILD, asmuid, total, 0x1C, 0) + struct.pack("<I", total - 0x1C)
    return head + body, asmuid


def description(asmuid, outputs, label="main"):
    outs = "".join('''
				<output uid="%d" identifier="%s"
					type="5" format="%d" width="256" height="256" mipmaps="0">
					<outputgui label="%s" typegui="image">
						<channels/>
					</outputgui>
				</output>''' % (uid, ident, fmt, ident) for uid, ident, fmt in outputs)
    return '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE sbsdescription SYSTEM "sbsdescription.dtd">
<sbsdescription formatversion="2.1" asmuid="%d" cookerbuild="%d" content="full">
	<graphs count="1">
		<graph pkgurl="pkg://%s" label="%s">
			<sbspresets count="0">
			</sbspresets>
			<guigroups/>
			<outputs count="%d">%s
			</outputs>
			<inputs count="1">
				<input uid="%d" identifier="$randomseed"
					type="4" default="0" alteroutputs="" alternodes="0"/>
			</inputs>
		</graph>
	</graphs>
</sbsdescription>
''' % (asmuid, COOKER_BUILD, label, label, len(outputs), outs, SEED_UID)


def sbsar(path, nodes, outputs, bits16=False):
    """outputs: [(uid, identifier, node index, gray)]. Writes path (a 7z, method Copy)."""
    name = os.path.splitext(os.path.basename(path))[0]
    asm, asmuid = assemble(nodes, [(u, i, (0xC0 if g else 0) | (0x100 if bits16 else 0)) for u, _, i, g in outputs])
    xml = description(asmuid, [(u, ident, 12 if g else 0) for u, ident, _, g in outputs])
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "assemblies", "content", "0000")
        os.makedirs(c)
        open(os.path.join(c, name + ".sbsasm"), "wb").write(asm)
        open(os.path.join(c, name + ".xml"), "w").write(xml)
        if os.path.exists(path): os.remove(path)
        subprocess.run(["7z", "a", "-mx=0", os.path.abspath(path), "assemblies"], cwd=d, check=True, stdout=subprocess.DEVNULL)
    return asm
