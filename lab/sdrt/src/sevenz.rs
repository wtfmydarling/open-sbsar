//! A self-contained, in-process 7z reader for .sbsar archives.
//!
//! WebAssembly-friendly: no threads, no file system (bytes in, entries out).
//! Supports Copy, LZMA, and LZMA2; validates CRCs; handles plain and encoded headers.

use std::io::Read;

/// Maximum total unpacked byte size across an entire archive (1 GiB).
pub const MAX_UNPACKED: u64 = 1024 * 1024 * 1024;

/// Maximum LZMA/LZMA2 dictionary size allowed (256 MiB).
pub const MAX_DICT_SIZE: u32 = 256 * 1024 * 1024;

/// Maximum encoded-header nesting depth (1 = one encoded header wrapping kHeader).
const MAX_ENCODED_HEADER_DEPTH: usize = 1;

/// One entry extracted from a 7z archive.
#[derive(Debug, Clone)]
pub struct Entry {
    pub name: String,
    pub is_directory: bool,
    pub data: Vec<u8>,
}

// 7z format NIDs
const K_END: u8 = 0x00;
const K_HEADER: u8 = 0x01;
const K_ARCHIVE_PROPERTIES: u8 = 0x02;
const K_ADDITIONAL_STREAMS_INFO: u8 = 0x03;
const K_MAIN_STREAMS_INFO: u8 = 0x04;
const K_FILES_INFO: u8 = 0x05;
const K_PACK_INFO: u8 = 0x06;
const K_UNPACK_INFO: u8 = 0x07;
const K_SUB_STREAMS_INFO: u8 = 0x08;
const K_SIZE: u8 = 0x09;
const K_CRC: u8 = 0x0A;
const K_FOLDER: u8 = 0x0B;
const K_CODERS_UNPACK_SIZE: u8 = 0x0C;
const K_NUM_UNPACK_STREAM: u8 = 0x0D;
const K_EMPTY_STREAM: u8 = 0x0E;
const K_EMPTY_FILE: u8 = 0x0F;
const K_NAME: u8 = 0x11;
const K_ENCODED_HEADER: u8 = 0x17;

const SIGNATURE: &[u8; 6] = &[0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C];

/// Read all entries from a 7z archive in memory.
pub fn read(bytes: &[u8]) -> Result<Vec<Entry>, String> {
    if bytes.len() < 32 {
        return Err("archive smaller than 7z signature header (32 bytes)".into());
    }
    if &bytes[0..6] != SIGNATURE {
        return Err("not a 7z archive: invalid signature".into());
    }
    let version_major = bytes[6];
    if version_major != 0 {
        return Err(format!("unsupported 7z version {version_major}.{}", bytes[7]));
    }

    let start_header_crc = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let next_header_offset = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    let next_header_size = u64::from_le_bytes(bytes[20..28].try_into().unwrap());
    let next_header_crc = u32::from_le_bytes(bytes[28..32].try_into().unwrap());

    if start_header_crc != 0 {
        let calc_crc = crc32fast::hash(&bytes[12..32]);
        if calc_crc != start_header_crc {
            return Err("start header CRC mismatch".into());
        }
    }

    let header_start = 32usize
        .checked_add(u64_to_usize(next_header_offset, "next header offset")?)
        .ok_or("next header offset overflow")?;
    let header_size = u64_to_usize(next_header_size, "next header size")?;
    let header_end = header_start.checked_add(header_size).ok_or("header range overflow")?;

    if header_end > bytes.len() {
        return Err("next header extends past end of archive".into());
    }

    let header_slice = &bytes[header_start..header_end];
    let calc_next_crc = crc32fast::hash(header_slice);
    if calc_next_crc != next_header_crc {
        return Err("next header CRC mismatch".into());
    }

    let mut unpacked_budget = MAX_UNPACKED;
    let mut header_buf: Vec<u8>;
    let mut active_header: &[u8] = header_slice;
    let mut depth = 0usize;

    let (streams_info, files) = loop {
        let mut r = Reader::new(active_header);
        let nid = r.read_u8()?;
        if nid == K_HEADER {
            break read_header(&mut r)?;
        }
        if nid != K_ENCODED_HEADER {
            return Err(format!("expected kHeader or kEncodedHeader, got 0x{nid:02X}"));
        }
        if depth >= MAX_ENCODED_HEADER_DEPTH {
            return Err("encoded header nesting depth exceeded".into());
        }
        depth += 1;

        let enc_streams = read_streams_info(&mut r)?;
        if enc_streams.folders.len() != 1 {
            return Err(format!(
                "encoded header must have exactly 1 folder, got {}",
                enc_streams.folders.len()
            ));
        }
        validate_streams_info(&enc_streams, bytes.len(), &mut unpacked_budget)?;
        let mut sub_stream_idx = 0;
        let unpacked_header_parts =
            decompress_folder(&enc_streams, 0, &mut sub_stream_idx, bytes)?;
        if unpacked_header_parts.len() != 1 {
            return Err("encoded header must unpack to 1 stream".into());
        }
        header_buf = unpacked_header_parts.into_iter().next().unwrap();
        active_header = &header_buf;
    };

    validate_streams_info(&streams_info, bytes.len(), &mut unpacked_budget)?;
    assemble_entries(&streams_info, files, bytes)
}

fn read_header(r: &mut Reader) -> Result<(StreamsInfo, Vec<FileMetadata>), String> {
    let mut streams_info = StreamsInfo {
        pack_info: None,
        folders: Vec::new(),
        sub_streams: None,
    };
    let mut files = Vec::new();
    let mut seen_streams = false;
    let mut seen_files = false;

    loop {
        let nid = r.read_u8()?;
        match nid {
            K_END => break,
            K_ARCHIVE_PROPERTIES => {
                loop {
                    let pid = r.read_u8()?;
                    if pid == K_END {
                        break;
                    }
                    let sz = r.read_varint_usize("archive property size")?;
                    r.skip(sz)?;
                }
            }
            K_ADDITIONAL_STREAMS_INFO => {
                return Err("additional streams unsupported".into());
            }
            K_MAIN_STREAMS_INFO => {
                if seen_streams {
                    return Err("duplicate kMainStreamsInfo in header".into());
                }
                seen_streams = true;
                streams_info = read_streams_info(r)?;
            }
            K_FILES_INFO => {
                if seen_files {
                    return Err("duplicate kFilesInfo in header".into());
                }
                seen_files = true;
                files = read_files_info(r)?;
            }
            other => return Err(format!("unexpected nid 0x{other:02X} in header")),
        }
    }

    Ok((streams_info, files))
}

fn validate_streams_info(
    streams_info: &StreamsInfo,
    archive_len: usize,
    unpacked_budget: &mut u64,
) -> Result<(), String> {
    let total_pack_streams = streams_info
        .folders
        .iter()
        .try_fold(0usize, |acc, f| acc.checked_add(f.packed_streams.len()))
        .ok_or("total packed stream count overflow")?;

    if !streams_info.folders.is_empty() {
        let pack_info = streams_info
            .pack_info
            .as_ref()
            .ok_or("archive has folders but no pack info")?;
        if total_pack_streams > pack_info.pack_sizes.len() {
            return Err("folders require more pack streams than pack info provides".into());
        }
        let mut cursor = 32usize
            .checked_add(u64_to_usize(pack_info.pack_pos, "pack_pos")?)
            .ok_or("pack start offset overflow")?;
        for &sz in &pack_info.pack_sizes[..total_pack_streams] {
            let sz_usize = u64_to_usize(sz, "pack stream size")?;
            cursor = cursor
                .checked_add(sz_usize)
                .ok_or("pack stream end offset overflow")?;
        }
        if cursor > archive_len {
            return Err("pack stream extends past end of archive".into());
        }
    }

    let mut total_sub_streams = 0usize;
    for folder in &streams_info.folders {
        if folder.unpack_sizes.len() != folder.total_out_streams {
            return Err("folder unpack_sizes length mismatch".into());
        }
        for &sz in &folder.unpack_sizes {
            if sz > MAX_UNPACKED {
                return Err(format!(
                    "coder unpack size {sz} exceeds MAX_UNPACKED ({MAX_UNPACKED})"
                ));
            }
            u64_to_usize(sz, "coder unpack size")?;
        }
        let folder_unpack = folder.total_unpack_size();
        if folder_unpack > *unpacked_budget {
            return Err(format!(
                "archive total unpacked size exceeds MAX_UNPACKED ({MAX_UNPACKED})"
            ));
        }
        *unpacked_budget -= folder_unpack;

        if streams_info.sub_streams.is_none() && folder.num_sub_streams != 1 {
            return Err("multiple sub-streams without kSubStreamsInfo".into());
        }
        total_sub_streams = total_sub_streams
            .checked_add(folder.num_sub_streams)
            .ok_or("total sub-stream count overflow")?;
    }

    if let Some(sub) = &streams_info.sub_streams {
        if sub.unpack_sizes.len() != total_sub_streams {
            return Err("sub-stream unpack_sizes count does not match folders".into());
        }
    }

    Ok(())
}

fn assemble_entries(
    streams_info: &StreamsInfo,
    files: Vec<FileMetadata>,
    archive_bytes: &[u8],
) -> Result<Vec<Entry>, String> {
    let streamed_files = files.iter().filter(|f| f.has_stream).count();
    let total_sub_streams: usize = streams_info
        .folders
        .iter()
        .try_fold(0usize, |acc, f| acc.checked_add(f.num_sub_streams))
        .ok_or("total sub-stream count overflow")?;
    if streamed_files > total_sub_streams {
        return Err("more streamed files than folder streams available".into());
    }

    let mut entries = Vec::with_capacity(cap(files.len()));
    let mut folder_idx = 0;
    let mut sub_stream_idx = 0;
    let mut current_folder_sub_streams: Vec<Vec<u8>> = Vec::new();
    let mut current_folder_pos = 0;

    for file in files {
        if !file.has_stream {
            entries.push(Entry {
                name: file.name,
                is_directory: file.is_directory,
                data: Vec::new(),
            });
            continue;
        }

        while current_folder_pos >= current_folder_sub_streams.len() {
            if folder_idx >= streams_info.folders.len() {
                return Err("more streamed files than folder streams available".into());
            }
            current_folder_sub_streams = decompress_folder(
                streams_info,
                folder_idx,
                &mut sub_stream_idx,
                archive_bytes,
            )?;
            current_folder_pos = 0;
            folder_idx += 1;
        }

        let slot = current_folder_sub_streams
            .get_mut(current_folder_pos)
            .ok_or("missing decompressed sub-stream")?;
        let data = std::mem::take(slot);
        current_folder_pos += 1;

        entries.push(Entry {
            name: file.name,
            is_directory: false,
            data,
        });
    }

    Ok(entries)
}

fn u64_to_usize(v: u64, what: &str) -> Result<usize, String> {
    usize::try_from(v).map_err(|_| format!("{what} ({v}) overflows usize"))
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or("unexpected end of 7z header stream")?;
        self.pos += 1;
        Ok(b)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], String> {
        if self.remaining() < len {
            return Err("unexpected end of 7z header stream while reading slice".into());
        }
        let end = self.pos + len;
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn skip(&mut self, len: usize) -> Result<(), String> {
        if self.remaining() < len {
            return Err("unexpected end of 7z header stream while skipping".into());
        }
        self.pos += len;
        Ok(())
    }

    fn read_u32_le(&mut self) -> Result<u32, String> {
        let slice = self.read_exact(4)?;
        Ok(u32::from_le_bytes(slice.try_into().unwrap()))
    }

    fn read_varint(&mut self) -> Result<u64, String> {
        let first = self.read_u8()? as u64;
        let mut mask = 0x80u64;
        let mut value = 0u64;
        for i in 0..8 {
            if (first & mask) == 0 {
                return Ok(value | ((first & (mask - 1)) << (8 * i)));
            }
            let b = self.read_u8()? as u64;
            value |= b << (8 * i);
            mask >>= 1;
        }
        Ok(value)
    }

    fn read_varint_usize(&mut self, what: &str) -> Result<usize, String> {
        let v = self.read_varint()?;
        u64_to_usize(v, what)
    }

    fn read_bits(&mut self, count: usize) -> Result<Vec<bool>, String> {
        let needed_bytes = count.div_ceil(8);
        if needed_bytes > self.remaining() {
            return Err("bitmask count exceeds remaining header length".into());
        }
        let mut bits = Vec::with_capacity(cap(count));
        let mut mask = 0u8;
        let mut cache = 0u8;
        for _ in 0..count {
            if mask == 0 {
                mask = 0x80;
                cache = self.read_u8()?;
            }
            bits.push((cache & mask) != 0);
            mask >>= 1;
        }
        Ok(bits)
    }

    fn read_all_or_bits(&mut self, count: usize) -> Result<Vec<bool>, String> {
        if count > self.data.len() {
            return Err("bitmask count exceeds header length".into());
        }
        let all = self.read_u8()? != 0;
        if all {
            Ok(vec![true; count])
        } else {
            self.read_bits(count)
        }
    }
}

struct Coder {
    method_id: Vec<u8>,
    num_in_streams: usize,
    num_out_streams: usize,
    properties: Vec<u8>,
}

struct BindPair {
    in_index: usize,
    out_index: usize,
}

struct Folder {
    coders: Vec<Coder>,
    total_out_streams: usize,
    bind_pairs: Vec<BindPair>,
    packed_streams: Vec<usize>,
    unpack_sizes: Vec<u64>,
    crc: Option<u32>,
    num_sub_streams: usize,
}

impl Folder {
    fn total_unpack_size(&self) -> u64 {
        if self.total_out_streams == 0 {
            return 0;
        }
        for i in (0..self.total_out_streams).rev() {
            if !self.bind_pairs.iter().any(|bp| bp.out_index == i) {
                return self.unpack_sizes.get(i).copied().unwrap_or(0);
            }
        }
        0
    }
}

struct PackInfo {
    pack_pos: u64,
    pack_sizes: Vec<u64>,
    pack_crcs: Vec<Option<u32>>,
}

struct SubStreamsInfo {
    unpack_sizes: Vec<u64>,
    crcs: Vec<Option<u32>>,
}

struct StreamsInfo {
    pack_info: Option<PackInfo>,
    folders: Vec<Folder>,
    sub_streams: Option<SubStreamsInfo>,
}

struct FileMetadata {
    name: String,
    has_stream: bool,
    is_directory: bool,
}

fn read_streams_info(r: &mut Reader) -> Result<StreamsInfo, String> {
    let mut pack_info = None;
    let mut folders = Vec::new();
    let mut sub_streams = None;

    let mut nid = r.read_u8()?;
    if nid == K_PACK_INFO {
        pack_info = Some(read_pack_info(r)?);
        nid = r.read_u8()?;
    }

    if nid == K_UNPACK_INFO {
        folders = read_unpack_info(r)?;
        nid = r.read_u8()?;
    }

    if nid == K_SUB_STREAMS_INFO {
        sub_streams = Some(read_sub_streams_info(r, &mut folders)?);
        nid = r.read_u8()?;
    }

    if nid != K_END {
        return Err(format!("expected kEnd in streams info, got 0x{nid:02X}"));
    }

    Ok(StreamsInfo {
        pack_info,
        folders,
        sub_streams,
    })
}

fn read_pack_info(r: &mut Reader) -> Result<PackInfo, String> {
    let pack_pos = r.read_varint()?;
    let num_pack_streams = r.read_varint_usize("num_pack_streams")?;
    if num_pack_streams > r.remaining() {
        return Err("pack stream count exceeds remaining header length".into());
    }
    let mut pack_sizes = Vec::with_capacity(cap(num_pack_streams));
    let mut pack_crcs = vec![None; num_pack_streams];
    let mut seen_sizes = false;
    let mut seen_crcs = false;

    loop {
        let nid = r.read_u8()?;
        match nid {
            K_END => break,
            K_SIZE => {
                if seen_sizes {
                    return Err("duplicate kSize in pack info".into());
                }
                seen_sizes = true;
                for _ in 0..num_pack_streams {
                    pack_sizes.push(r.read_varint()?);
                }
            }
            K_CRC => {
                if seen_crcs {
                    return Err("duplicate kCRC in pack info".into());
                }
                seen_crcs = true;
                let defined = r.read_all_or_bits(num_pack_streams)?;
                for i in 0..num_pack_streams {
                    if defined[i] {
                        pack_crcs[i] = Some(r.read_u32_le()?);
                    }
                }
            }
            other => return Err(format!("unexpected nid 0x{other:02X} in pack info")),
        }
    }

    if num_pack_streams > 0 && !seen_sizes {
        return Err("missing kSize in pack info".into());
    }

    Ok(PackInfo {
        pack_pos,
        pack_sizes,
        pack_crcs,
    })
}

fn validate_folder_graph(
    coders: &[Coder],
    total_in: usize,
    total_out: usize,
    bind_pairs: &[BindPair],
    packed_streams: &[usize],
) -> Result<(), String> {
    let mut used_in = vec![false; total_in];
    let mut used_out = vec![false; total_out];

    for bp in bind_pairs {
        if bp.in_index >= total_in {
            return Err(format!(
                "bind pair in_index {} out of range (total_in={total_in})",
                bp.in_index
            ));
        }
        if bp.out_index >= total_out {
            return Err(format!(
                "bind pair out_index {} out of range (total_out={total_out})",
                bp.out_index
            ));
        }
        if used_in[bp.in_index] {
            return Err(format!("duplicate bind pair in_index {}", bp.in_index));
        }
        if used_out[bp.out_index] {
            return Err(format!("duplicate bind pair out_index {}", bp.out_index));
        }
        used_in[bp.in_index] = true;
        used_out[bp.out_index] = true;
    }

    let mut seen_packed = vec![false; total_in];
    for &ps in packed_streams {
        if ps >= total_in {
            return Err(format!(
                "packed stream index {ps} out of range (total_in={total_in})"
            ));
        }
        if used_in[ps] {
            return Err(format!("packed stream index {ps} is already bound"));
        }
        if seen_packed[ps] {
            return Err(format!("duplicate packed stream index {ps}"));
        }
        seen_packed[ps] = true;
    }

    // Build stream -> coder maps and check for cycles in the coder DAG via Kahn's algorithm.
    let mut in_to_coder = Vec::with_capacity(total_in);
    let mut out_to_coder = Vec::with_capacity(total_out);
    for (ci, c) in coders.iter().enumerate() {
        in_to_coder.extend(std::iter::repeat_n(ci, c.num_in_streams));
        out_to_coder.extend(std::iter::repeat_n(ci, c.num_out_streams));
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); coders.len()];
    let mut in_deg = vec![0usize; coders.len()];
    for bp in bind_pairs {
        let src = out_to_coder[bp.out_index];
        let dst = in_to_coder[bp.in_index];
        if src == dst {
            return Err("cycle detected in folder bind pairs (self-loop)".into());
        }
        adj[src].push(dst);
        in_deg[dst] += 1;
    }

    let mut stack: Vec<usize> = (0..coders.len()).filter(|&i| in_deg[i] == 0).collect();
    let mut visited = 0usize;
    while let Some(u) = stack.pop() {
        visited += 1;
        for &v in &adj[u] {
            in_deg[v] -= 1;
            if in_deg[v] == 0 {
                stack.push(v);
            }
        }
    }
    if visited != coders.len() {
        return Err("cycle detected in folder bind pairs".into());
    }

    Ok(())
}

fn read_unpack_info(r: &mut Reader) -> Result<Vec<Folder>, String> {
    if r.read_u8()? != K_FOLDER {
        return Err("expected kFolder".into());
    }
    let num_folders = r.read_varint_usize("num_folders")?;
    if num_folders > r.remaining() {
        return Err("folder count exceeds remaining header length".into());
    }
    let external = r.read_u8()?;
    if external != 0 {
        return Err("external unpack info unsupported".into());
    }

    let mut folders = Vec::with_capacity(cap(num_folders));
    for _ in 0..num_folders {
        let num_coders = r.read_varint_usize("num_coders")?;
        if num_coders == 0 || num_coders > r.remaining() {
            return Err("invalid coder count in folder".into());
        }
        let mut coders = Vec::with_capacity(cap(num_coders));
        let mut total_in = 0usize;
        let mut total_out = 0usize;

        for _ in 0..num_coders {
            let flags = r.read_u8()?;
            let id_size = (flags & 0x0F) as usize;
            let is_simple = (flags & 0x10) == 0;
            let has_props = (flags & 0x20) != 0;
            if (flags & 0x80) != 0 {
                return Err("alternative coder methods unsupported".into());
            }
            let method_id = r.read_exact(id_size)?.to_vec();
            let (num_in, num_out) = if is_simple {
                (1usize, 1usize)
            } else {
                (
                    r.read_varint_usize("coder num_in_streams")?,
                    r.read_varint_usize("coder num_out_streams")?,
                )
            };
            total_in = total_in
                .checked_add(num_in)
                .ok_or("folder total_in overflow")?;
            total_out = total_out
                .checked_add(num_out)
                .ok_or("folder total_out overflow")?;
            if total_out > r.remaining() {
                return Err("folder output stream count exceeds remaining header length".into());
            }
            if total_in > r.remaining().saturating_mul(2).saturating_add(1) {
                return Err("folder input stream count exceeds remaining header length".into());
            }
            let properties = if has_props {
                let prop_size = r.read_varint_usize("coder property size")?;
                r.read_exact(prop_size)?.to_vec()
            } else {
                Vec::new()
            };
            coders.push(Coder {
                method_id,
                num_in_streams: num_in,
                num_out_streams: num_out,
                properties,
            });
        }

        if total_out == 0 {
            return Err("folder has 0 output streams".into());
        }
        let num_bind_pairs = total_out - 1;
        if num_bind_pairs.saturating_mul(2) > r.remaining() {
            return Err("bind pair count exceeds remaining header length".into());
        }
        let mut bind_pairs = Vec::with_capacity(cap(num_bind_pairs));
        for _ in 0..num_bind_pairs {
            bind_pairs.push(BindPair {
                in_index: r.read_varint_usize("bind pair in_index")?,
                out_index: r.read_varint_usize("bind pair out_index")?,
            });
        }

        let num_packed_streams = total_in
            .checked_sub(num_bind_pairs)
            .ok_or_else(|| "total_in < num_bind_pairs".to_string())?;
        if num_packed_streams == 0 {
            return Err("folder has 0 packed streams".into());
        }
        let mut packed_streams = Vec::with_capacity(cap(num_packed_streams));
        if num_packed_streams == 1 {
            let mut found = None;
            for i in 0..total_in {
                if !bind_pairs.iter().any(|bp| bp.in_index == i) {
                    found = Some(i);
                    break;
                }
            }
            packed_streams.push(found.ok_or_else(|| "cannot find packed stream index".to_string())?);
        } else {
            if num_packed_streams > r.remaining() {
                return Err("packed stream count exceeds remaining header length".into());
            }
            for _ in 0..num_packed_streams {
                packed_streams.push(r.read_varint_usize("packed stream index")?);
            }
        }

        validate_folder_graph(&coders, total_in, total_out, &bind_pairs, &packed_streams)?;

        folders.push(Folder {
            coders,
            total_out_streams: total_out,
            bind_pairs,
            packed_streams,
            unpack_sizes: Vec::new(),
            crc: None,
            num_sub_streams: 1,
        });
    }

    if r.read_u8()? != K_CODERS_UNPACK_SIZE {
        return Err("expected kCodersUnpackSize".into());
    }

    for folder in &mut folders {
        for _ in 0..folder.total_out_streams {
            folder.unpack_sizes.push(r.read_varint()?);
        }
    }

    let mut seen_crc = false;
    loop {
        let nid = r.read_u8()?;
        match nid {
            K_END => break,
            K_CRC => {
                if seen_crc {
                    return Err("duplicate kCRC in unpack info".into());
                }
                seen_crc = true;
                let defined = r.read_all_or_bits(num_folders)?;
                for i in 0..num_folders {
                    if defined[i] {
                        folders[i].crc = Some(r.read_u32_le()?);
                    }
                }
            }
            other => return Err(format!("unexpected nid 0x{other:02X} in unpack info")),
        }
    }

    Ok(folders)
}

fn read_sub_streams_info(r: &mut Reader, folders: &mut [Folder]) -> Result<SubStreamsInfo, String> {
    for f in folders.iter_mut() {
        f.num_sub_streams = 1;
    }

    let mut nid = r.read_u8()?;
    if nid == K_NUM_UNPACK_STREAM {
        let mut total_sub = 0usize;
        for folder in folders.iter_mut() {
            let n = r.read_varint_usize("num_sub_streams")?;
            total_sub = total_sub
                .checked_add(n)
                .ok_or("total sub-stream count overflow")?;
            if n > r.remaining().saturating_add(1) {
                return Err("sub-stream count exceeds remaining header length".into());
            }
            folder.num_sub_streams = n;
        }
        nid = r.read_u8()?;
    }

    let has_k_size = nid == K_SIZE;
    let mut unpack_sizes = Vec::new();
    for folder in folders.iter() {
        let total_unpack = folder.total_unpack_size();
        if folder.num_sub_streams == 0 {
            if total_unpack != 0 {
                return Err("folder with 0 sub-streams has non-zero unpack size".into());
            }
            continue;
        }
        if folder.num_sub_streams > 1 && !has_k_size {
            return Err("missing kSize for folder with multiple sub-streams".into());
        }
        let mut sum = 0u64;
        if has_k_size {
            for _ in 0..folder.num_sub_streams - 1 {
                let sz = r.read_varint()?;
                sum = sum
                    .checked_add(sz)
                    .ok_or("sub-stream size sum overflow")?;
                unpack_sizes.push(sz);
            }
        }
        if sum > total_unpack {
            return Err("sub-stream sizes exceed folder total unpack size".into());
        }
        unpack_sizes.push(total_unpack - sum);
    }
    if has_k_size {
        nid = r.read_u8()?;
    }

    let mut num_digests = 0usize;
    for folder in folders.iter() {
        if folder.num_sub_streams != 1 || folder.crc.is_none() {
            num_digests = num_digests
                .checked_add(folder.num_sub_streams)
                .ok_or("sub-stream digest count overflow")?;
        }
    }
    if num_digests > r.data.len() {
        return Err("sub-stream digest count exceeds header length".into());
    }

    let mut missing_crcs = vec![None; num_digests];
    if nid == K_CRC {
        let defined = r.read_all_or_bits(num_digests)?;
        for i in 0..num_digests {
            if defined[i] {
                missing_crcs[i] = Some(r.read_u32_le()?);
            }
        }
        nid = r.read_u8()?;
    }

    if nid != K_END {
        return Err(format!("expected kEnd in sub streams info, got 0x{nid:02X}"));
    }

    let mut crcs = Vec::with_capacity(cap(unpack_sizes.len()));
    let mut next_missing = 0;
    for folder in folders.iter() {
        if folder.num_sub_streams == 1 && folder.crc.is_some() {
            crcs.push(folder.crc);
        } else {
            for _ in 0..folder.num_sub_streams {
                crcs.push(missing_crcs.get(next_missing).copied().flatten());
                next_missing += 1;
            }
        }
    }

    Ok(SubStreamsInfo { unpack_sizes, crcs })
}

fn read_files_info(r: &mut Reader) -> Result<Vec<FileMetadata>, String> {
    let num_files = r.read_varint_usize("num_files")?;
    if num_files > r.remaining() {
        return Err("file count exceeds remaining header length".into());
    }
    let mut is_empty_stream: Option<Vec<bool>> = None;
    let mut is_empty_file: Option<Vec<bool>> = None;
    let mut names = Vec::with_capacity(cap(num_files));
    let mut seen_names = false;

    loop {
        let prop_type = r.read_u8()?;
        if prop_type == 0 {
            break;
        }
        let size = r.read_varint_usize("file property size")?;
        if size > r.remaining() {
            return Err("file property size exceeds remaining header length".into());
        }
        match prop_type {
            K_EMPTY_STREAM => {
                if is_empty_stream.is_some() {
                    return Err("duplicate kEmptyStream property".into());
                }
                let start = r.pos;
                is_empty_stream = Some(r.read_bits(num_files)?);
                let consumed = r.pos - start;
                if consumed > size {
                    return Err("kEmptyStream exceeds declared property size".into());
                }
                r.skip(size - consumed)?;
            }
            K_EMPTY_FILE => {
                if is_empty_file.is_some() {
                    return Err("duplicate kEmptyFile property".into());
                }
                let empty_stream = is_empty_stream
                    .as_ref()
                    .ok_or("kEmptyFile without preceding kEmptyStream")?;
                let count = empty_stream.iter().filter(|&&b| b).count();
                let start = r.pos;
                is_empty_file = Some(r.read_bits(count)?);
                let consumed = r.pos - start;
                if consumed > size {
                    return Err("kEmptyFile exceeds declared property size".into());
                }
                r.skip(size - consumed)?;
            }
            K_NAME => {
                if seen_names {
                    return Err("duplicate kName property".into());
                }
                seen_names = true;
                if size == 0 {
                    return Err("invalid empty file names property".into());
                }
                let external = r.read_u8()?;
                if external != 0 {
                    return Err("external file names unsupported".into());
                }
                let bytes = r.read_exact(size - 1)?;
                let mut current_utf16 = Vec::new();
                let mut idx = 0;
                while idx + 1 < bytes.len() {
                    let code = u16::from_le_bytes([bytes[idx], bytes[idx + 1]]);
                    idx += 2;
                    if code == 0 {
                        names.push(String::from_utf16_lossy(&current_utf16));
                        current_utf16.clear();
                    } else {
                        current_utf16.push(code);
                    }
                }
            }
            _ => {
                r.skip(size)?;
            }
        }
    }

    if names.len() != num_files {
        return Err(format!("names count {} != num_files {}", names.len(), num_files));
    }

    let mut files = Vec::with_capacity(cap(num_files));
    let mut empty_idx = 0usize;
    for (i, name) in names.into_iter().enumerate() {
        let empty = match is_empty_stream.as_ref() {
            Some(s) => *s.get(i).ok_or("kEmptyStream bitmask length mismatch")?,
            None => false,
        };
        let (has_stream, is_directory) = if !empty {
            (true, false)
        } else {
            let is_dir = if let Some(ef) = &is_empty_file {
                !*ef.get(empty_idx)
                    .ok_or("kEmptyFile bitmask length mismatch")?
            } else {
                true
            };
            empty_idx += 1;
            (false, is_dir)
        };
        files.push(FileMetadata {
            name,
            has_stream,
            is_directory,
        });
    }

    Ok(files)
}

fn decompress_folder(
    streams_info: &StreamsInfo,
    folder_idx: usize,
    sub_stream_idx: &mut usize,
    archive_bytes: &[u8],
) -> Result<Vec<Vec<u8>>, String> {
    let folder = streams_info
        .folders
        .get(folder_idx)
        .ok_or("folder index out of bounds")?;
    let pack_info = streams_info
        .pack_info
        .as_ref()
        .ok_or("archive has folders but no pack info")?;

    let prior_folders = streams_info
        .folders
        .get(..folder_idx)
        .ok_or("folder slice out of bounds")?;
    let first_pack_stream_index: usize = prior_folders
        .iter()
        .try_fold(0usize, |acc, f| acc.checked_add(f.packed_streams.len()))
        .ok_or("pack stream index overflow")?;

    if folder.packed_streams.len() != 1 {
        return Err("multi-pack-stream folders unsupported".into());
    }

    let prior_pack_sizes = pack_info
        .pack_sizes
        .get(..first_pack_stream_index)
        .ok_or("pack stream index out of bounds")?;
    let pack_offset = prior_pack_sizes
        .iter()
        .try_fold(0u64, |acc, &x| acc.checked_add(x))
        .ok_or("pack offset overflow")?;
    let pack_size_u64 = *pack_info
        .pack_sizes
        .get(first_pack_stream_index)
        .ok_or("pack stream index out of bounds")?;
    let pack_size = u64_to_usize(pack_size_u64, "pack stream size")?;

    let pack_start = 32usize
        .checked_add(u64_to_usize(pack_info.pack_pos, "pack_pos")?)
        .and_then(|v| v.checked_add(u64_to_usize(pack_offset, "pack_offset").ok()?))
        .ok_or("pack start offset overflow")?;
    let pack_end = pack_start.checked_add(pack_size).ok_or("pack range overflow")?;

    if pack_end > archive_bytes.len() {
        return Err("pack stream extends past end of archive".into());
    }

    let packed_slice = &archive_bytes[pack_start..pack_end];
    if let Some(expected_crc) = pack_info.pack_crcs.get(first_pack_stream_index).copied().flatten() {
        if crc32fast::hash(packed_slice) != expected_crc {
            return Err("pack stream CRC mismatch".into());
        }
    }

    let mut curr_coder_idx = folder.packed_streams[0];
    let mut owned_data: Option<Vec<u8>> = None;
    let mut visited_count = 0usize;

    loop {
        if visited_count > folder.coders.len() {
            return Err("cycle detected in folder bind pairs".into());
        }
        visited_count += 1;

        let coder = folder
            .coders
            .get(curr_coder_idx)
            .ok_or("invalid coder index")?;
        if coder.num_in_streams != 1 || coder.num_out_streams != 1 {
            return Err("multi-stream coder unsupported (e.g. BCJ2)".into());
        }
        let unpack_size = *folder
            .unpack_sizes
            .get(curr_coder_idx)
            .ok_or("missing unpack size for coder")?;

        let input_ref: &[u8] = match owned_data.as_deref() {
            Some(buf) => buf,
            None => packed_slice,
        };
        let next_data = decompress_coder(coder, input_ref, unpack_size)?;

        if let Some(bp) = folder.bind_pairs.iter().find(|bp| bp.out_index == curr_coder_idx) {
            curr_coder_idx = bp.in_index;
            owned_data = Some(next_data);
        } else {
            owned_data = Some(next_data);
            break;
        }
    }

    let current_data = owned_data.ok_or("folder produced no output")?;

    if current_data.len() as u64 != folder.total_unpack_size() {
        return Err("folder unpacked data length does not match declared folder size".into());
    }

    if let Some(folder_crc) = folder.crc {
        if crc32fast::hash(&current_data) != folder_crc {
            return Err("folder CRC mismatch".into());
        }
    }

    let mut sub_streams = Vec::with_capacity(cap(folder.num_sub_streams));
    let mut offset = 0usize;

    if folder.num_sub_streams == 1 {
        let (sz, expected_crc) = if let Some(sub_info) = &streams_info.sub_streams {
            let sz_u64 = *sub_info
                .unpack_sizes
                .get(*sub_stream_idx)
                .ok_or("missing sub-stream unpack size")?;
            let sz = u64_to_usize(sz_u64, "sub-stream unpack size")?;
            let crc = sub_info.crcs.get(*sub_stream_idx).copied().flatten();
            *sub_stream_idx += 1;
            (sz, crc)
        } else {
            (current_data.len(), folder.crc)
        };
        if sz != current_data.len() {
            return Err("sum of sub-stream sizes does not match folder unpack size".into());
        }
        if let Some(crc) = expected_crc {
            if crc32fast::hash(&current_data) != crc {
                return Err("sub-stream CRC mismatch".into());
            }
        }
        sub_streams.push(current_data);
        return Ok(sub_streams);
    }

    for _ in 0..folder.num_sub_streams {
        let (sz, expected_crc) = if let Some(sub_info) = &streams_info.sub_streams {
            let sz_u64 = *sub_info
                .unpack_sizes
                .get(*sub_stream_idx)
                .ok_or("missing sub-stream unpack size")?;
            let sz = u64_to_usize(sz_u64, "sub-stream unpack size")?;
            let crc = sub_info.crcs.get(*sub_stream_idx).copied().flatten();
            *sub_stream_idx += 1;
            (sz, crc)
        } else {
            return Err("multiple sub-streams without kSubStreamsInfo".into());
        };

        let end = offset.checked_add(sz).ok_or("sub-stream offset overflow")?;
        if end > current_data.len() {
            return Err("sub-stream size exceeds folder unpack data".into());
        }
        let piece = current_data[offset..end].to_vec();
        offset = end;

        if let Some(crc) = expected_crc {
            if crc32fast::hash(&piece) != crc {
                return Err("sub-stream CRC mismatch".into());
            }
        }
        sub_streams.push(piece);
    }

    if offset != current_data.len() {
        return Err("sum of sub-stream sizes does not match folder unpack size".into());
    }

    Ok(sub_streams)
}

fn decompress_coder(coder: &Coder, input: &[u8], uncompressed_size: u64) -> Result<Vec<u8>, String> {
    if uncompressed_size > MAX_UNPACKED {
        return Err(format!(
            "declared unpack size {uncompressed_size} exceeds MAX_UNPACKED ({MAX_UNPACKED})"
        ));
    }
    let unpack_usize = u64_to_usize(uncompressed_size, "coder unpack size")?;

    // Copy coder: empty method ID or 0x00
    if coder.method_id.is_empty() || coder.method_id == [0x00] {
        if input.len() as u64 != uncompressed_size {
            return Err(format!(
                "Copy coder size mismatch: input {} bytes != declared {} bytes",
                input.len(),
                uncompressed_size
            ));
        }
        return Ok(input.to_vec());
    }

    // LZMA: [0x03, 0x01, 0x01]
    if coder.method_id == [0x03, 0x01, 0x01] {
        if coder.properties.len() != 5 {
            return Err(format!(
                "LZMA properties must be 5 bytes, got {}",
                coder.properties.len()
            ));
        }
        let props = coder.properties[0];
        if props > 224 {
            return Err(format!("invalid LZMA properties byte: {props}"));
        }
        let pb = props / 45;
        let rem = props % 45;
        let lp = rem / 9;
        let lc = rem % 9;
        if lc > 8 || lp > 4 || pb > 4 {
            return Err(format!("invalid LZMA properties lc={lc} lp={lp} pb={pb}"));
        }
        let dict_size = u32::from_le_bytes(coder.properties[1..5].try_into().unwrap());
        if dict_size > MAX_DICT_SIZE {
            return Err(format!(
                "LZMA dictionary size {dict_size} exceeds MAX_DICT_SIZE ({MAX_DICT_SIZE})"
            ));
        }

        let reader = lzma_rust2::LzmaReader::new_with_props(
            input,
            uncompressed_size,
            props,
            dict_size,
            None,
        )
        .map_err(|e| format!("LZMA initialization failed: {e}"))?;

        let mut out = Vec::with_capacity(cap(unpack_usize));
        reader
            .take(uncompressed_size)
            .read_to_end(&mut out)
            .map_err(|e| format!("LZMA decompression failed: {e}"))?;
        if out.len() as u64 != uncompressed_size {
            return Err(format!(
                "LZMA output size mismatch: got {}, expected {}",
                out.len(),
                uncompressed_size
            ));
        }
        return Ok(out);
    }

    // LZMA2: [0x21]
    if coder.method_id == [0x21] {
        if coder.properties.len() != 1 {
            return Err(format!(
                "LZMA2 properties must be 1 byte, got {}",
                coder.properties.len()
            ));
        }
        let prop = coder.properties[0] as u32;
        if (prop & !0x3F) != 0 {
            return Err(format!("unsupported LZMA2 property bits: 0x{prop:02X}"));
        }
        let dict_size = if prop == 40 {
            0xFFFF_FFFF
        } else if prop > 40 {
            return Err("LZMA2 dictionary size exceeds 4GiB".into());
        } else {
            (2 | (prop & 1)) << (prop / 2 + 11)
        };
        if dict_size > MAX_DICT_SIZE {
            return Err(format!(
                "LZMA2 dictionary size {dict_size} exceeds MAX_DICT_SIZE ({MAX_DICT_SIZE})"
            ));
        }

        let reader = lzma_rust2::Lzma2Reader::new(input, dict_size, None);
        let mut out = Vec::with_capacity(cap(unpack_usize));
        reader
            .take(uncompressed_size)
            .read_to_end(&mut out)
            .map_err(|e| format!("LZMA2 decompression failed: {e}"))?;
        if out.len() as u64 != uncompressed_size {
            return Err(format!(
                "LZMA2 output size mismatch: got {}, expected {}",
                out.len(),
                uncompressed_size
            ));
        }
        return Ok(out);
    }

    // Any other coder (BCJ, PPMd, AES encryption, etc.)
    Err(format!(
        "unsupported compression method ID: {:?} (encryption, BCJ, etc.)",
        coder.method_id
    ))
}

/// Bounds a preallocation taken from an untrusted header field, so a
/// malformed archive cannot request an enormous allocation up front.
/// Vectors still grow past this as real data arrives.
fn cap(n: usize) -> usize {
    n.min(1 << 20)
}
