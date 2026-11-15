//! A self-contained, in-process 7z reader for .sbsar archives.
//!
//! WebAssembly-friendly: no threads, no file system (bytes in, entries out).
//! Supports Copy, LZMA, and LZMA2; validates CRCs; handles plain and encoded headers.

use std::io::Read;

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
        .checked_add(usize::try_from(next_header_offset).map_err(|_| "next header offset overflow")?)
        .ok_or("next header offset overflow")?;
    let header_size = usize::try_from(next_header_size).map_err(|_| "next header size overflow")?;
    let header_end = header_start.checked_add(header_size).ok_or("header range overflow")?;

    if header_end > bytes.len() {
        return Err("next header extends past end of archive".into());
    }

    let header_slice = &bytes[header_start..header_end];
    let calc_next_crc = crc32fast::hash(header_slice);
    if calc_next_crc != next_header_crc {
        return Err("next header CRC mismatch".into());
    }

    let mut r = Reader::new(header_slice);
    let nid = r.read_u8()?;

    let (streams_info, files) = if nid == K_ENCODED_HEADER {
        let enc_streams = read_streams_info(&mut r)?;
        if enc_streams.folders.is_empty() {
            return Err("encoded header has no folders".into());
        }
        let mut sub_stream_idx = 0;
        let unpacked_header_parts = decompress_folder(&enc_streams, 0, &mut sub_stream_idx, bytes)?;
        let unpacked_header = unpacked_header_parts.into_iter().next().ok_or("empty unpacked header")?;
        let mut hdr_r = Reader::new(&unpacked_header);
        let main_nid = hdr_r.read_u8()?;
        if main_nid != K_HEADER {
            return Err(format!("expected kHeader inside encoded header, got 0x{main_nid:02X}"));
        }
        read_header(&mut hdr_r)?
    } else if nid == K_HEADER {
        read_header(&mut r)?
    } else {
        return Err(format!("expected kHeader or kEncodedHeader, got 0x{nid:02X}"));
    };

    assemble_entries(&streams_info, files, bytes)
}

fn read_header(r: &mut Reader) -> Result<(StreamsInfo, Vec<FileMetadata>), String> {
    let mut streams_info = StreamsInfo {
        pack_info: None,
        folders: Vec::new(),
        sub_streams: None,
    };
    let mut files = Vec::new();

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
                    let sz = r.read_varint()? as usize;
                    r.skip(sz)?;
                }
            }
            K_ADDITIONAL_STREAMS_INFO => {
                return Err("additional streams unsupported".into());
            }
            K_MAIN_STREAMS_INFO => {
                streams_info = read_streams_info(r)?;
            }
            K_FILES_INFO => {
                files = read_files_info(r)?;
            }
            other => return Err(format!("unexpected nid 0x{other:02X} in header")),
        }
    }

    Ok((streams_info, files))
}

fn assemble_entries(
    streams_info: &StreamsInfo,
    files: Vec<FileMetadata>,
    archive_bytes: &[u8],
) -> Result<Vec<Entry>, String> {
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

        if current_folder_pos >= current_folder_sub_streams.len() {
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

        let data = std::mem::take(&mut current_folder_sub_streams[current_folder_pos]);
        current_folder_pos += 1;

        entries.push(Entry {
            name: file.name,
            is_directory: false,
            data,
        });
    }

    Ok(entries)
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
        if self.pos >= self.data.len() {
            return Err("unexpected end of 7z header stream".into());
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], String> {
        if self.remaining() < len {
            return Err("unexpected end of 7z header stream while reading slice".into());
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
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

    fn read_bits(&mut self, count: usize) -> Result<Vec<bool>, String> {
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
    let num_pack_streams = r.read_varint()? as usize;
    let mut pack_sizes = Vec::with_capacity(cap(num_pack_streams));
    let mut pack_crcs = vec![None; num_pack_streams];

    loop {
        let nid = r.read_u8()?;
        match nid {
            K_END => break,
            K_SIZE => {
                for _ in 0..num_pack_streams {
                    pack_sizes.push(r.read_varint()?);
                }
            }
            K_CRC => {
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

    Ok(PackInfo {
        pack_pos,
        pack_sizes,
        pack_crcs,
    })
}

fn read_unpack_info(r: &mut Reader) -> Result<Vec<Folder>, String> {
    if r.read_u8()? != K_FOLDER {
        return Err("expected kFolder".into());
    }
    let num_folders = r.read_varint()? as usize;
    let external = r.read_u8()?;
    if external != 0 {
        return Err("external unpack info unsupported".into());
    }

    let mut folders = Vec::with_capacity(cap(num_folders));
    for _ in 0..num_folders {
        let num_coders = r.read_varint()? as usize;
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
                (1, 1)
            } else {
                (r.read_varint()? as usize, r.read_varint()? as usize)
            };
            total_in += num_in;
            total_out += num_out;
            let properties = if has_props {
                let prop_size = r.read_varint()? as usize;
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
        let mut bind_pairs = Vec::with_capacity(cap(num_bind_pairs));
        for _ in 0..num_bind_pairs {
            bind_pairs.push(BindPair {
                in_index: r.read_varint()? as usize,
                out_index: r.read_varint()? as usize,
            });
        }

        let num_packed_streams = total_in
            .checked_sub(num_bind_pairs)
            .ok_or_else(|| "total_in < num_bind_pairs".to_string())?;
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
            for _ in 0..num_packed_streams {
                packed_streams.push(r.read_varint()? as usize);
            }
        }

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

    loop {
        let nid = r.read_u8()?;
        match nid {
            K_END => break,
            K_CRC => {
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
        for folder in folders.iter_mut() {
            folder.num_sub_streams = r.read_varint()? as usize;
        }
        nid = r.read_u8()?;
    }

    let mut unpack_sizes = Vec::new();
    for folder in folders.iter() {
        if folder.num_sub_streams == 0 {
            continue;
        }
        let total_unpack = folder.total_unpack_size();
        let mut sum = 0u64;
        if nid == K_SIZE {
            for _ in 0..folder.num_sub_streams - 1 {
                let sz = r.read_varint()?;
                unpack_sizes.push(sz);
                sum += sz;
            }
        }
        if sum > total_unpack {
            return Err("sub-stream sizes exceed folder total unpack size".into());
        }
        unpack_sizes.push(total_unpack - sum);
    }
    if nid == K_SIZE {
        nid = r.read_u8()?;
    }

    let mut num_digests = 0;
    for folder in folders.iter() {
        if folder.num_sub_streams != 1 || folder.crc.is_none() {
            num_digests += folder.num_sub_streams;
        }
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
    let num_files = r.read_varint()? as usize;
    let mut is_empty_stream = None;
    let mut is_empty_file = None;
    let mut names = Vec::with_capacity(cap(num_files));

    loop {
        let prop_type = r.read_u8()?;
        if prop_type == 0 {
            break;
        }
        let size = r.read_varint()? as usize;
        match prop_type {
            K_EMPTY_STREAM => {
                is_empty_stream = Some(r.read_bits(num_files)?);
            }
            K_EMPTY_FILE => {
                let count = is_empty_stream
                    .as_ref()
                    .map_or(0, |s| s.iter().filter(|&&b| b).count());
                is_empty_file = Some(r.read_bits(count)?);
            }
            K_NAME => {
                let external = r.read_u8()?;
                if external != 0 {
                    return Err("external file names unsupported".into());
                }
                if size == 0 {
                    return Err("invalid empty file names property".into());
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
    let mut empty_idx = 0;
    for (i, name) in names.into_iter().enumerate() {
        let empty = is_empty_stream.as_ref().map_or(false, |s| s[i]);
        let (has_stream, is_directory) = if !empty {
            (true, false)
        } else {
            let is_dir = if let Some(ef) = &is_empty_file {
                !ef[empty_idx]
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
    let folder = &streams_info.folders[folder_idx];
    let pack_info = streams_info
        .pack_info
        .as_ref()
        .ok_or("archive has folders but no pack info")?;

    let first_pack_stream_index: usize = streams_info.folders[..folder_idx]
        .iter()
        .map(|f| f.packed_streams.len())
        .sum();

    if folder.packed_streams.len() != 1 {
        return Err("multi-pack-stream folders unsupported".into());
    }

    let pack_offset: u64 = pack_info.pack_sizes[..first_pack_stream_index]
        .iter()
        .sum();
    let pack_size = *pack_info
        .pack_sizes
        .get(first_pack_stream_index)
        .ok_or("pack stream index out of bounds")? as usize;

    let pack_start = 32usize
        .checked_add(usize::try_from(pack_info.pack_pos).map_err(|_| "pack_pos overflow")?)
        .and_then(|v| v.checked_add(usize::try_from(pack_offset).map_err(|_| "pack_offset overflow").ok()?))
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
    let mut current_data = packed_slice.to_vec();
    let mut visited_count = 0;

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

        let next_data = decompress_coder(coder, &current_data, unpack_size)?;

        if let Some(bp) = folder.bind_pairs.iter().find(|bp| bp.out_index == curr_coder_idx) {
            curr_coder_idx = bp.in_index;
            current_data = next_data;
        } else {
            current_data = next_data;
            break;
        }
    }

    if let Some(folder_crc) = folder.crc {
        if crc32fast::hash(&current_data) != folder_crc {
            return Err("folder CRC mismatch".into());
        }
    }

    let mut sub_streams = Vec::with_capacity(cap(folder.num_sub_streams));
    let mut offset = 0usize;

    for _ in 0..folder.num_sub_streams {
        let (sz, expected_crc) = if let Some(sub_info) = &streams_info.sub_streams {
            let sz = *sub_info
                .unpack_sizes
                .get(*sub_stream_idx)
                .ok_or("missing sub-stream unpack size")? as usize;
            let crc = sub_info.crcs.get(*sub_stream_idx).copied().flatten();
            *sub_stream_idx += 1;
            (sz, crc)
        } else {
            (current_data.len(), folder.crc)
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
        if coder.properties.len() < 5 {
            return Err("LZMA properties too short (< 5 bytes)".into());
        }
        let props = coder.properties[0];
        let dict_size = u32::from_le_bytes(coder.properties[1..5].try_into().unwrap());
        let mut reader = lzma_rust2::LzmaReader::new_with_props(
            input,
            uncompressed_size,
            props,
            dict_size,
            None,
        )
        .map_err(|e| format!("LZMA initialization failed: {e}"))?;

        let mut out = Vec::with_capacity(cap(uncompressed_size as usize));
        reader
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
        if coder.properties.is_empty() {
            return Err("LZMA2 properties too short (empty)".into());
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

        let reader = lzma_rust2::Lzma2Reader::new(input, dict_size, None);
        let mut out = Vec::with_capacity(cap(uncompressed_size as usize));
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
