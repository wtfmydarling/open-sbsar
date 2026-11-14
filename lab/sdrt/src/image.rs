//! Images: RGBA f32 pixels (grayscale keeps its value in all of R, G, B and alpha 1), sampling with
//! pixel centres at (x + 0.5) / w, and the quantization of each storage format.

#[derive(Clone)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub color: bool,
    pub px: Vec<[f32; 4]>,
}

impl Image {
    pub fn new(w: usize, h: usize, color: bool, fill: [f32; 4]) -> Image {
        Image { w, h, color, px: vec![fill; w * h] }
    }

    pub fn gray(v: f32) -> [f32; 4] {
        [v, v, v, 1.0]
    }

    pub fn at(&self, x: usize, y: usize) -> [f32; 4] {
        self.px[y * self.w + x]
    }

    /// Pixel (x, y) with wrapping (tile) or clamping per axis.
    pub fn fetch(&self, x: i64, y: i64, wrap: (bool, bool)) -> [f32; 4] {
        let fx = if wrap.0 { x.rem_euclid(self.w as i64) } else { x.clamp(0, self.w as i64 - 1) };
        let fy = if wrap.1 { y.rem_euclid(self.h as i64) } else { y.clamp(0, self.h as i64 - 1) };
        self.px[fy as usize * self.w + fx as usize]
    }

    /// Bilinear sample at uv in [0, 1]² (pixel centres at (x + 0.5) / w).
    pub fn bilinear(&self, u: f32, v: f32, wrap: (bool, bool)) -> [f32; 4] {
        // the GPU's texture unit rounds bilinear weights to 1/256 (directional motion blur, FX-Map
        // image patterns, normals, warps: NOTES "sdrt semantics"); SDRT_BIL256=0 turns it off
        static BIL256: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *BIL256.get_or_init(|| crate::envc::var("SDRT_BIL256").map_or(true, |v| v != "0")) {
            return self.bilinear_256(u, v, wrap);
        }
        self.bilinear_exact(u, v, wrap)
    }

    /// Bilinear sample with exact (float) weights.
    pub fn bilinear_exact(&self, u: f32, v: f32, wrap: (bool, bool)) -> [f32; 4] {
        let x = u * self.w as f32 - 0.5;
        let y = v * self.h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let a = self.fetch(x0, y0, wrap);
        let b = self.fetch(x0 + 1, y0, wrap);
        let c = self.fetch(x0, y0 + 1, wrap);
        let d = self.fetch(x0 + 1, y0 + 1, wrap);
        let mut o = [0.0; 4];
        for k in 0..4 {
            let top = a[k] + (b[k] - a[k]) * fx;
            let bot = c[k] + (d[k] - c[k]) * fx;
            o[k] = top + (bot - top) * fy;
        }
        o
    }

    /// Bilinear sample whose interpolation fractions use the GPU texture
    /// unit's measured 8-bit fixed-point precision.  Coordinates themselves
    /// remain floating point; only fx/fy are rounded to nearest 1/256.
    pub fn bilinear_256(&self, u: f32, v: f32, wrap: (bool, bool)) -> [f32; 4] {
        let x = u * self.w as f32 - 0.5;
        let y = v * self.h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let fx = ((x - x0) * 256.0).round() / 256.0;
        let fy = ((y - y0) * 256.0).round() / 256.0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let a = self.fetch(x0, y0, wrap);
        let b = self.fetch(x0 + 1, y0, wrap);
        let c = self.fetch(x0, y0 + 1, wrap);
        let d = self.fetch(x0 + 1, y0 + 1, wrap);
        let mut o = [0.0; 4];
        for k in 0..4 {
            let top = a[k] + (b[k] - a[k]) * fx;
            let bot = c[k] + (d[k] - c[k]) * fx;
            o[k] = top + (bot - top) * fy;
        }
        o
    }

    /// Texel (x, y): wrapped on tiled axes, `border` past the edge of the others.
    /// `border` None clamps to the edge texel.
    fn fetch_or(&self, x: i64, y: i64, wrap: (bool, bool), border: Option<[f32; 4]>) -> [f32; 4] {
        let (w, h) = (self.w as i64, self.h as i64);
        if (!wrap.0 && (x < 0 || x >= w)) || (!wrap.1 && (y < 0 || y >= h)) {
            if let Some(b) = border { return b; }
            return self.fetch(x, y, wrap);
        }
        self.px[y.rem_euclid(h) as usize * self.w + x.rem_euclid(w) as usize]
    }

    /// Bilinear (1/256 weights, as bilinear_256) whose texels past a non-tiled edge are `border`
    /// (a GPU border colour): the transformation node at mip level ≥ 1 blends its edge texels with
    /// the matte (redbrickbuilding node 5: weights 0.25 / 0.125 exactly as the border texel's).
    pub fn bilinear_or(&self, u: f32, v: f32, wrap: (bool, bool), border: Option<[f32; 4]>, format: u32) -> [f32; 4] {
        let x = u * self.w as f32 - 0.5;
        let y = v * self.h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let fx = ((x - x0) * 256.0).round() / 256.0;
        let fy = ((y - y0) * 256.0).round() / 256.0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let a = self.fetch_or(x0, y0, wrap, border);
        let b = self.fetch_or(x0 + 1, y0, wrap, border);
        let c = self.fetch_or(x0, y0 + 1, wrap, border);
        let d = self.fetch_or(x0 + 1, y0 + 1, wrap, border);
        // the cross weight rounds to 1/256 too, the others follow from it (as bilinear_border;
        // xfmag.py: the corner alpha 0.926² → 0.855)
        let cross = (fx * fy * 256.0).round() / 256.0;
        let wt = [1.0 - fx - fy + cross, fx - cross, fy - cross, cross];
        let mut o = [0.0; 4];
        for k in 0..4 {
            if format <= 1 {
                // UNORM sources interpolate in 16-bit integers, ties up (as bilinear_border;
                // xfcross.py: 8- and 16-bit sources were one code low with float weights)
                let sum: u32 = [a[k], b[k], c[k], d[k]].iter().zip(wt)
                    .map(|(v, w)| (v * 65535.0).round() as u32 * (w * 256.0) as u32).sum();
                o[k] = ((sum + 128) >> 8) as f32 / 65535.0;
            } else {
                o[k] = a[k] * wt[0] + b[k] * wt[1] + c[k] * wt[2] + d[k] * wt[3];
            }
        }
        o
    }

    /// Texel (x, y), transparent black outside the image (a texture with a zero border colour).
    fn fetch_border(&self, x: i64, y: i64) -> [f32; 4] {
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 {
            return [0.0; 4];
        }
        let p = self.px[y as usize * self.w + x as usize];
        if self.color { p } else { [p[0], p[0], p[0], 1.0] }
    }

    /// FX-Map image sampling against a transparent black border. Fractional coordinates and the
    /// cross term use 8 fractional bits; derive the other three weights from the cross term so
    /// their sum and both marginals stay exact. Independently rounding all four weights is wrong
    /// at ties. Measured with parity-basis images in scratch/fx_polish/weight_probe.py.
    pub fn bilinear_border(&self, u: f32, v: f32, format: u32) -> [f32; 4] {
        let x = u * self.w as f32 - 0.5;
        let y = v * self.h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let fx = ((x - x0) * 256.0).round() / 256.0;
        let fy = ((y - y0) * 256.0).round() / 256.0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let (a, b) = (self.fetch_border(x0, y0), self.fetch_border(x0 + 1, y0));
        let (c, d) = (self.fetch_border(x0, y0 + 1), self.fetch_border(x0 + 1, y0 + 1));
        let mut o = [0.0; 4];
        static COUPLED_WEIGHTS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *COUPLED_WEIGHTS.get_or_init(|| crate::envc::var("SDRT_FX_FILTER").as_deref() != Ok("legacy")) {
            let cross = (fx * fy * 256.0).round() / 256.0;
            let weights = [1.0 - fx - fy + cross, fx - cross, fy - cross, cross];
            for k in 0..4 {
                if format <= 1 {
                    // UNORM inputs interpolate in 16-bit integer space, with ties up. Integer
                    // arithmetic also prevents f32 accumulation from changing those ties.
                    let sum: u32 = [a[k], b[k], c[k], d[k]].iter().zip(weights)
                        .map(|(v, w)| (v * 65535.0).round() as u32 * (w * 256.0) as u32).sum();
                    let q = (sum + 128) >> 8;
                    o[k] = q as f32 / 65535.0;
                } else {
                    o[k] = a[k] * weights[0] + b[k] * weights[1] + c[k] * weights[2] + d[k] * weights[3];
                    if format == 2 { o[k] = half_sample(o[k]); }
                }
            }
            return o;
        }
        for k in 0..4 {
            let top = a[k] + (b[k] - a[k]) * fx;
            let bot = c[k] + (d[k] - c[k]) * fx;
            o[k] = top + (bot - top) * fy;
        }
        o
    }

    pub fn nearest_border(&self, u: f32, v: f32) -> [f32; 4] {
        self.fetch_border((u * self.w as f32).floor() as i64, (v * self.h as f32).floor() as i64)
    }

    pub fn nearest(&self, u: f32, v: f32, wrap: (bool, bool)) -> [f32; 4] {
        self.fetch((u * self.w as f32).floor() as i64, (v * self.h as f32).floor() as i64, wrap)
    }

    /// Twice the size, bilinear at the new pixel centres.
    pub fn upsample2(&self, wrap: (bool, bool)) -> Image {
        self.upsample_axes(true, true, wrap)
    }

    /// Half the size along the chosen axes: the average of each texel pair, stored in `format`.
    pub fn downsample_axes(&self, along_x: bool, along_y: bool, format: u32) -> Image {
        use rayon::prelude::*;
        let (nx, ny) = (1 + along_x as usize, 1 + along_y as usize);
        let (w, h) = (self.w / nx, self.h / ny);
        let mut o = Image::new(w, h, self.color, [0.0; 4]);
        // test switches read once (per-pixel env lookups made a 4096² halving take seconds)
        let ds_round = if format <= 1 { crate::envc::var("SDRT_DSROUND").ok() } else { None };
        // 16-bit halvings: the integer average of the stored codes, ties up, as the GPU's mip
        // generator (rock-basket-wave-pavement 701: 1,0,1,0 → 32768, the fixed-point rule gave 32767
        // and a `> 0.5` in PP 702 flipped 576 pixels). 8-bit keeps the fixed-point rule (abs034).
        // SDRT_DSINT=0: fixed-point for both, =all: integer for both
        let ds_int = match crate::envc::var("SDRT_DSINT").as_deref() {
            Ok("0") => false,
            Ok("all") | Ok("1") => format <= 1,
            _ => format == 1,
        };
        o.px.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let mut acc = [0.0f32; 4];
                for dy in 0..ny {
                    for dx in 0..nx {
                        let p = self.at(x * nx + dx, y * ny + dy);
                        for c in 0..4 { acc[c] += p[c]; }
                    }
                }
                row[x] = acc.map(|a| a / (nx * ny) as f32);
                if let Some(mode) = &ds_round {
                    let m = if format == 0 { 255.0f32 } else { 65535.0 };
                    let f = |v: f32| -> f32 { let t = v.clamp(0.0, 1.0) * m; (match mode.as_str() {
                        "trunc" => t.floor(), "rne" => { let r = t.round(); if (t - t.floor() - 0.5).abs() < 1e-6 && (r as i64) % 2 != 0 { r - 1.0 } else { r } },
                        "up" => (t + 0.5).floor(), _ => t.round() }) / m };
                    row[x] = row[x].map(f);
                }
                if ds_int {
                    // integer average of the stored codes, ties up (a GPU mip generator)
                    let (m, n) = (if format == 0 { 255.0 } else { 65535.0 }, (nx * ny) as u32);
                    let mut sum = [0u32; 4];
                    for dy in 0..ny { for dx in 0..nx {
                        let p = self.at(x * nx + dx, y * ny + dy);
                        for c in 0..4 { sum[c] += (p[c].clamp(0.0, 1.0) * m).round() as u32; }
                    } }
                    row[x] = sum.map(|t| ((t + n / 2) / n) as f32 / m);
                }
            }
        });
        o.quantize(format);
        o
    }

    /// Twice the size along the chosen axes, the texture unit's bilinear (edges clamped, UNORM
    /// formats ≤ 1 interpolated in 16-bit integers) at the new pixel centres.
    pub fn upsample_axes_fmt(&self, along_x: bool, along_y: bool, wrap: (bool, bool), format: u32) -> Image {
        use rayon::prelude::*;
        let (w, h) = (self.w << along_x as usize, self.h << along_y as usize);
        let mut o = Image::new(w, h, self.color, [0.0; 4]);
        o.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, q) in row.iter_mut().enumerate() {
                *q = self.bilinear_or((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32, wrap, None, format);
            }
        });
        o
    }

    /// Twice the size along the chosen axes, bilinear at the new pixel centres.
    pub fn upsample_axes(&self, along_x: bool, along_y: bool, wrap: (bool, bool)) -> Image {
        use rayon::prelude::*;
        let (w, h) = (self.w << along_x as usize, self.h << along_y as usize);
        let mut o = Image::new(w, h, self.color, [0.0; 4]);
        o.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, q) in row.iter_mut().enumerate() {
                *q = self.bilinear((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32, wrap);
            }
        });
        o
    }

    /// Resample to another size by bilinear sampling at the new pixel centres.
    pub fn resized(&self, w: usize, h: usize) -> Image {
        self.resized_fmt(w, h, 3)
    }

    /// Resample to another size in the source's texture format (UNORM formats ≤ 1 interpolate in
    /// 16-bit integers with coupled 1/256 weights, as the GPU texture unit).
    pub fn resized_fmt(&self, w: usize, h: usize, format: u32) -> Image {
        if (w, h) == (self.w, self.h) {
            return self.clone();
        }
        // SDRT_RESIZEMIP=1 (test): a larger input halved (pair averages) while at least twice the
        // size — brick-wall (1) 2, a 128² sharpen of a 4096² bitmap (an artifact of small copies):
        // mean 0.30 → 0.18, not the rule; off by default
        if (self.w >= 2 * w || self.h >= 2 * h) && crate::env_once!("SDRT_RESIZEMIP").as_deref() == Ok("1") {
            let mut cur = self.downsample_axes(self.w >= 2 * w, self.h >= 2 * h, 3);
            while cur.w >= 2 * w || cur.h >= 2 * h {
                cur = cur.downsample_axes(cur.w >= 2 * w, cur.h >= 2 * h, 3);
            }
            return cur.resized_fmt(w, h, format);
        }
        let mut o = Image::new(w, h, self.color, [0.0; 4]);
        // an input of another size is resampled with its edges clamped (waterdamagedconcrete blends
        // 101/166/231/294 and feathergen 395: edge pixels 0.013–0.03 off with wrap, exact with clamp).
        // SDRT_RESIZE=wrap: the old wrap; tex[clamp|wrap]: the texture unit's UNORM interpolation (test)
        let mode = crate::env_once!("SDRT_RESIZE").unwrap_or("clamp");
        let wrap = mode.contains("wrap");
        let use_tex = mode.starts_with("tex") || (format <= 1 && mode != "float");
        for y in 0..h {
            for x in 0..w {
                let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                o.px[y * w + x] = if use_tex { self.bilinear_or(u, v, (wrap, wrap), None, 1) } else { self.bilinear(u, v, (wrap, wrap)) };
            }
        }
        o
    }

    /// Truncate to an integer format (mip levels: the average of 2×2 pixels, rounded down).
    pub fn quantize_floor(&mut self, format: u32) {
        let s = match format {
            0 => 255.0,
            1 => 65535.0,
            _ => return,
        };
        for p in &mut self.px {
            *p = p.map(|v| ((v * s + 1e-3).floor() / s).clamp(0.0, 1.0));
        }
    }

    /// Store as the node's format would: 8-bit, 16-bit, half or float. Grayscale keeps one value.
    pub fn quantize(&mut self, format: u32) {
        let q: fn(f32) -> f32 = match format {
            0 => |v| unorm(v, 8) as f32 / 255.0,
            1 => |v| unorm(v, 16) as f32 / 65535.0,
            2 => half,
            _ => |v| v,
        };
        let color = self.color;
        for p in &mut self.px {
            if color {
                *p = p.map(q);
            } else {
                let g = q(p[0]);
                *p = [g, g, g, 1.0];
            }
        }
    }
}

/// Float to an n-bit integer as SD stores it (measured with sbsrender for 8 and 16 bits): through a
/// fixed point of n + 4 bits, truncated, then rounded with ties down.
pub fn unorm(v: f32, bits: u32) -> u32 {
    let k = bits + 4;
    let i = (v.clamp(0.0, 1.0) as f64 * (1u64 << k) as f64).floor() as u64;
    ((i * ((1u64 << bits) - 1) + (1u64 << (k - 1)) - 1) >> k) as u32
}

/// A mip chain: level 0 is the image; each next level averages 2×2 pixels of the previous one and
/// is stored in `format` before the next is built (8-bit: 97.5% exact against sbsrender, the rest
/// ±1/255 where both levels round exact ties).
pub fn mips(img: &Image, format: u32) -> Vec<Image> {
    // UNORM mip levels: the integer average of the four texels with ties up, as the GPU's mip
    // generator (quickpavement 269: a 2× minified nearest transformation was one code off; the chain
    // 271 → 274 → steep gradient maps 278/758/1245: 0.122 → exact). SDRT_MIPROUND=old: float average
    let mr = crate::envc::var("SDRT_MIPROUND").ok();
    // 8-bit levels: the float average stored with the fixed-point rule — rock_walkway 656…703,
    // red_brick_v1 826, beef-jerky 1113/1119 (8-bit chains down to 1×1, 2/255 off with integer
    // averages) exact. SDRT_MIPROUND=all: integer for 8-bit too, =old: float for both
    let int_mips = match mr.as_deref() { Some("old") => false, Some("all") => true, _ => format == 1 };
    let mut v = vec![img.clone()];
    while v.last().unwrap().w > 1 || v.last().unwrap().h > 1 {
        let s = v.last().unwrap();
        let (w, h) = ((s.w / 2).max(1), (s.h / 2).max(1));
        let (nx, ny) = (s.w / w, s.h / h);
        let mut o = Image::new(w, h, s.color, [0.0; 4]);
        use rayon::prelude::*;
        o.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, q) in row.iter_mut().enumerate() {
                let mut acc = [0.0f32; 4];
                for dy in 0..ny {
                    for dx in 0..nx {
                        let p = s.at(x * nx + dx, y * ny + dy);
                        for c in 0..4 {
                            acc[c] += p[c];
                        }
                    }
                }
                *q = acc.map(|a| a / (nx * ny) as f32);
                if format <= 1 && int_mips {
                    // UNORM levels: the texture unit's average of the integer texels, ties up
                    let (m, n) = (if format == 0 { 255.0 } else { 65535.0 }, (nx * ny) as u32);
                    let mut sum = [0u32; 4];
                    for dy in 0..ny {
                        for dx in 0..nx {
                            let p = s.at(x * nx + dx, y * ny + dy);
                            for c in 0..4 { sum[c] += (p[c].clamp(0.0, 1.0) * m).round() as u32; }
                        }
                    }
                    *q = sum.map(|t| ((t + n / 2) / n) as f32 / m);
                }
            }
        });
        o.quantize(format);
        v.push(o);
    }
    v
}

/// Round to the nearest IEEE half float.
pub fn half(v: f32) -> f32 {
    if !v.is_finite() {
        return v;
    }
    let b = v.to_bits();
    let sign = b & 0x8000_0000;
    let exp = ((b >> 23) & 0xFF) as i32 - 127;
    if exp > 15 {
        return f32::from_bits(sign | 0x477F_E000); // 65504
    }
    if exp < -24 {
        return f32::from_bits(sign);
    }
    // keep 10 mantissa bits (fewer for subnormal halves), round to nearest even
    let keep = if exp < -14 { 10 - (-14 - exp) } else { 10 };
    let drop = 23 - keep;
    let m = b & ((1 << drop) - 1);
    let mut r = b - m;
    let half_ulp = 1u32 << (drop - 1);
    if m > half_ulp || (m == half_ulp && (r >> drop) & 1 == 1) {
        r += 1 << drop;
    }
    f32::from_bits(r)
}

/// The half-float texture sampler rounds ties away from zero, unlike storage. This removes most
/// half-input errors; its internal mixed-exponent precision remains open (half_ties.py).
fn half_sample(v: f32) -> f32 {
    if !v.is_finite() { return v; }
    let bits = v.to_bits();
    let sign = bits & 0x8000_0000;
    let exponent = ((bits >> 23) & 255) as i32 - 127;
    if exponent > 15 { return f32::from_bits(sign | 0x477f_e000); }
    if exponent < -25 { return f32::from_bits(sign); }
    if exponent == -25 { return f32::from_bits(sign | (103 << 23)); }
    let keep = if exponent < -14 { 10 - (-14 - exponent) } else { 10 };
    let drop = 23 - keep;
    let rounded = (bits + (1 << (drop - 1))) & !((1 << drop) - 1);
    f32::from_bits(rounded)
}

/// Round toward zero when storing a gray half-float FX-Map target.
pub fn half_trunc(v: f32) -> f32 {
    if !v.is_finite() { return v; }
    let bits = v.to_bits();
    let sign = bits & 0x8000_0000;
    let exponent = ((bits >> 23) & 255) as i32 - 127;
    if exponent > 15 { return f32::from_bits(sign | 0x477f_e000); }
    if exponent < -24 { return f32::from_bits(sign); }
    let keep = if exponent < -14 { 10 - (-14 - exponent) } else { 10 };
    f32::from_bits(bits & !((1 << (23 - keep)) - 1))
}
