//! Baseline JPEG decoding as sbsrender does it (NOTES "Embedded JPEGs"): the Chen–Wang integer
//! IDCT of the MPEG reference decoder, as in NanoJPEG — bit-exact with sbsrender on lizard_scales
//! node 0 (1024² gray, every pixel). Color: NanoJPEG's YCbCr → RGB with chroma replicated (see
//! `decode`). Progressive / arithmetic / 12-bit files are refused.

/// Width, height, channels (1 or 3) and the 8-bit pixels, rows top to bottom.
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

/// Column c of the row-transformed block into out[0], out[stride], … (8 rows).
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

/// A Huffman table: (code length, value) for every 16-bit prefix.
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
                        // a restart marker: let it be read as bits (checked by the scan loop)
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
        // chroma: replicated, except that at odd luma columns of a 2× horizontally subsampled
        // plane a value below 128 is one less (measured: brick_bonus_castle_wall and burning
        // coals, 4:2:0, every pixel of three 64² regions)
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
