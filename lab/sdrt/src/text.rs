//! Text node support: a minimal TrueType reader (glyf outlines, cmap 4/12, hmtx) and an exact-area
//! anti-aliased rasterizer (signed-area accumulation, nonzero fill). Fonts are embedded in the
//! .sbsar program block (u32 id, u32 length, the font file) by the text record.

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
    /// CFF outlines ('OTTO' fonts): charstrings, global and local subroutines (byte ranges)
    cff: Option<Cff>,
}

struct Cff {
    charstrings: Vec<(usize, usize)>,
    gsubrs: Vec<(usize, usize)>,
    lsubrs: Vec<(usize, usize)>,
}

/// A CFF INDEX at o: its items as absolute byte ranges, and the offset after it.
fn cff_index(d: &[u8], o: usize) -> (Vec<(usize, usize)>, usize) {
    let n = u16be(d, o) as usize;
    if n == 0 { return (vec![], o + 2); }
    let osz = d.get(o + 2).copied().unwrap_or(1) as usize;
    let off = |i: usize| (0..osz).fold(0usize, |a, k| a << 8 | d.get(o + 3 + i * osz + k).copied().unwrap_or(0) as usize);
    let data = o + 3 + (n + 1) * osz - 1;
    ((0..n).map(|i| (data + off(i), data + off(i + 1))).collect(), data + off(n))
}

/// A CFF DICT: operator → operands.
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
                // real number: nibbles (not needed for offsets; skipped)
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

/// stb_truetype's outline vertices: move, line, cubic (control 1, control 2, end), integer
/// coordinates (its `(int)` casts of the float pen).
#[derive(Clone, Copy, Debug)]
enum Vx { Move(i32, i32), Line(i32, i32), Cubic(i32, i32, i32, i32, i32, i32) }

fn u16be(d: &[u8], o: usize) -> u16 { d.get(o..o + 2).map_or(0, |b| u16::from_be_bytes([b[0], b[1]])) }
fn i16be(d: &[u8], o: usize) -> i16 { u16be(d, o) as i16 }
fn u32be(d: &[u8], o: usize) -> u32 { d.get(o..o + 4).map_or(0, |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]])) }

impl<'a> Font<'a> {
    pub fn parse(d: &'a [u8]) -> Option<Font<'a>> {
        // a TrueType collection ('ttcf'): its last font (emergencyhatch 4379, Cambria + Cambria Math:
        // the size follows font 1's hhea 1595/−455, not font 0's 1946/−455 — index 1 or the last,
        // undecided); table offsets count from the file start
        let base = if u32be(d, 0) == 0x7474_6366 {
            let nf = u32be(d, 8) as usize;
            u32be(d, 12 + 4 * nf.saturating_sub(1)) as usize
        } else { 0 };
        let ver = u32be(d, base);
        // TrueType outlines, or CFF ('OTTO')
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
            // header, Name INDEX, Top DICT INDEX, String INDEX, Global Subr INDEX; CharStrings (17)
            // and Private (18: size, offset) from the Top DICT; Subrs (19) relative to Private
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
        // format 4
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

    /// Pair kerning from the 'kern' table (Microsoft version 0, horizontal format-0 subtables,
    /// values summed), in font units; GPOS is not read (FreeType's FT_Get_Kerning reads only 'kern').
    pub fn kern(&self, left: usize, right: usize) -> f64 {
        let d = self.d;
        let Some(&(k, _)) = self.tables.get(b"kern") else { return 0.0 };
        if u16be(d, k) != 0 { return 0.0; }
        let nt = u16be(d, k + 2) as usize;
        let (mut o, mut v) = (k + 4, 0.0);
        let key = (left as u32) << 16 | right as u32;
        for _ in 0..nt {
            let (len, cov) = (u16be(d, o + 2) as usize, u16be(d, o + 4));
            // coverage: bit 0 horizontal, bit 1 minimum, bit 2 cross-stream, high byte the format
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

    /// A CFF font's glyph g as stb_truetype's vertices (Type 2 charstring); None for TrueType.
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

    /// A CFF glyph flattened as stb_truetype does (cubics: halving while the control polygon is
    /// longer than the chord by the flatness), in font units; None for TrueType fonts.
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

    /// A CFF glyph's box (xMin, xMax) over its vertices, control points included (stb's
    /// GetGlyphBox for CFF); None for an empty glyph or a TrueType font.
    fn cff_xbounds(&self, g: usize) -> Option<(f64, f64)> {
        let vs = self.cff_vertices(g)?;
        let (mut lo, mut hi) = (i32::MAX, i32::MIN);
        for v in vs {
            let xs: &[i32] = &match v { Vx::Move(x, _) | Vx::Line(x, _) => [x, x, x], Vx::Cubic(a, _, b, _, x, _) => [x, a, b] };
            for &x in xs { lo = lo.min(x); hi = hi.max(x); }
        }
        (lo <= hi).then_some((lo as f64, hi as f64))
    }

    /// The glyf header's bounding box (xMin, xMax) in font units; None for an empty glyph.
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

    /// Contours of glyph g in font units (y up) as quadratic points (x, y, on_curve).
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
        // composite
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

/// Flatten quadratic contours (already in pixel space) into closed polylines.
/// One quadratic from `a` (already in the polyline) over control `q` to `b`, as `steps` chords;
/// steps 0: bisect, halving while the Euclidean deviation |a + b − 2q| (output pixels) is ≥ √2
/// (SDRT_TEXTFT; SDRT_TEXTFTNORM=max for the larger component), each halving quartering it
/// (scratch/fb/textprobe.py, one quadratic in a made font: deviation along x splits in 2 from
/// 1.38–1.45 px, in 4 from 5.5–5.8; along the diagonal 2 chords for 1.0–3.5 px of the larger
/// component, 4 for 4–14, 8 for 16–40; FreeType's own ¼ px of the larger component would give more).
fn quad(poly: &mut Vec<(f64, f64)>, a: (f64, f64), q: (f64, f64), b: (f64, f64), steps: usize) {
    let steps = if steps > 0 { steps } else {
        let thr: f64 = crate::envc::var("SDRT_TEXTFT").ok().and_then(|v| v.parse().ok()).unwrap_or(std::f64::consts::SQRT_2);
        let (ex, ey) = (a.0 + b.0 - 2.0 * q.0, a.1 + b.1 - 2.0 * q.1);
        let mut dev = if crate::envc::var("SDRT_TEXTFTNORM").as_deref() == Ok("max") { ex.abs().max(ey.abs()) } else { ex.hypot(ey) };
        let mut n = 1;
        while dev >= thr && n < 1 << 16 { dev /= 4.0; n *= 2; }
        n
    };
    for s in 1..=steps {
        let t = s as f64 / steps as f64;
        poly.push(((1.0 - t) * (1.0 - t) * a.0 + 2.0 * (1.0 - t) * t * q.0 + t * t * b.0,
                   (1.0 - t) * (1.0 - t) * a.1 + 2.0 * (1.0 - t) * t * q.1 + t * t * b.1));
    }
}

/// stb_truetype-style flattening in font units: implied on-curve points between two off-curve
/// points are the integer midpoint (a + b) >> 1, and each quadratic is bisected recursively while
/// the squared distance between its midpoint and its chord's, (|a − 2q + b| / 4)², exceeds the
/// flatness² (`flat`, font units: 0.35 px over the scale). Points come back in font units.
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

pub fn flatten(contours: &[Vec<(f64, f64, bool)>], steps: usize) -> Vec<Vec<(f64, f64)>> {
    let mut out = vec![];
    for c in contours {
        if c.len() < 2 { continue; }
        // start at an on-curve point (or the midpoint of the first two off-curve points)
        let n = c.len();
        let start = (0..n).find(|&i| c[i].2);
        let pts: Vec<(f64, f64, bool)> = match start {
            Some(s) => (0..n).map(|k| c[(s + k) % n]).collect(),
            None => {
                let m = ((c[0].0 + c[1].0) / 2.0, (c[0].1 + c[1].1) / 2.0, true);
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
                    Some(q) => quad(&mut poly, cur, q, (p.0, p.1), steps),
                    None => poly.push((p.0, p.1)),
                }
                cur = (p.0, p.1);
            } else {
                if let Some(q) = ctrl {
                    let mid = ((q.0 + p.0) / 2.0, (q.1 + p.1) / 2.0);
                    quad(&mut poly, cur, q, mid, steps);
                    cur = mid;
                }
                ctrl = Some((p.0, p.1));
            }
        }
        out.push(poly);
    }
    out
}

/// Exact-area coverage of closed polylines (pixel units, y down) into a w×h buffer (nonzero,
/// |winding| clamped to 1): signed-area accumulation per scanline.
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
        // stb_truetype's active-edge init for edges starting above scan_y_top = 0 evaluates x at
        // -1 instead of 0 (creamstone 120..239 serifs/arches above hhea ascent, and glyphs above
        // y = 0). SDRT_TEXTTOP=exact restores exact geometric clipping.
        let mut x = a.0 + (if a.1 < 0.0 && crate::envc::var("SDRT_TEXTTOP").as_deref() != Ok("exact") { -1.0 - a.1 } else { y0 - a.1 }) * dxdy;
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
                // within one pixel column
                let xm = 0.5 * (x + xnext) - x0f;
                if x0i < 0 { acc[row] += d; }
                else if (x0i as usize) < w {
                    acc[row + x0i as usize] += d - d * xm;
                    if (x0i as usize) + 1 < w { acc[row + x0i as usize + 1] += d * xm; }
                }
            } else {
                // the crossing's area spread over the columns it spans (sums to d)
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
