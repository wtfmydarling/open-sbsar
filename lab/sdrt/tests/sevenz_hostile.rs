use open_sbsar::sevenz;
use std::panic;
use std::path::PathBuf;
use std::time::Instant;

// 7z format NIDs for building test archives
const K_END: u8 = 0x00;
const K_HEADER: u8 = 0x01;
const K_ARCHIVE_PROPERTIES: u8 = 0x02;
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

const ONE_GIB: u64 = 1024 * 1024 * 1024;
const MIB_256: u32 = 256 * 1024 * 1024;

fn push_varint(out: &mut Vec<u8>, v: u64) {
    if v < 0x80 {
        out.push(v as u8);
        return;
    }
    for n in 1..8u32 {
        let limit = 1u64 << (7 * (n + 1));
        if v < limit {
            let prefix = (!0u8) << (8 - n);
            let high = (v >> (8 * n)) as u8;
            out.push(prefix | high);
            for i in 0..n {
                out.push((v >> (8 * i)) as u8);
            }
            return;
        }
    }
    out.push(0xFF);
    out.extend_from_slice(&v.to_le_bytes());
}

fn wrap_7z(packed_data: &[u8], header: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + packed_data.len() + header.len());
    out.extend_from_slice(&[0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C]);
    out.extend_from_slice(&[0x00, 0x04]); // version 0.4
    let next_offset = (packed_data.len() as u64).to_le_bytes();
    let next_size = (header.len() as u64).to_le_bytes();
    let next_crc = crc32fast::hash(header).to_le_bytes();

    let mut start_info = [0u8; 20];
    start_info[0..8].copy_from_slice(&next_offset);
    start_info[8..16].copy_from_slice(&next_size);
    start_info[16..20].copy_from_slice(&next_crc);
    let start_crc = crc32fast::hash(&start_info).to_le_bytes();

    out.extend_from_slice(&start_crc);
    out.extend_from_slice(&start_info);
    out.extend_from_slice(packed_data);
    out.extend_from_slice(header);
    out
}

fn push_utf16_names(out: &mut Vec<u8>, names: &[&str]) {
    let mut payload = vec![0u8]; // external = 0
    for name in names {
        for unit in name.encode_utf16() {
            payload.extend_from_slice(&unit.to_le_bytes());
        }
        payload.extend_from_slice(&0u16.to_le_bytes());
    }
    out.push(K_NAME);
    push_varint(out, payload.len() as u64);
    out.extend_from_slice(&payload);
}

fn push_files_info_simple(hdr: &mut Vec<u8>, names: &[&str]) {
    hdr.push(K_FILES_INFO);
    push_varint(hdr, names.len() as u64);
    push_utf16_names(hdr, names);
    hdr.push(K_END);
}

fn push_simple_copy_streams_info(hdr: &mut Vec<u8>, sizes: &[u64]) {
    hdr.push(K_MAIN_STREAMS_INFO);
    // PackInfo
    hdr.push(K_PACK_INFO);
    push_varint(hdr, 0); // pack_pos
    push_varint(hdr, sizes.len() as u64);
    hdr.push(K_SIZE);
    for &sz in sizes {
        push_varint(hdr, sz);
    }
    hdr.push(K_END);
    // UnpackInfo
    hdr.push(K_UNPACK_INFO);
    hdr.push(K_FOLDER);
    push_varint(hdr, sizes.len() as u64);
    hdr.push(0); // external = 0
    for _ in sizes {
        push_varint(hdr, 1); // 1 coder
        hdr.push(0x01); // simple, method_id len = 1
        hdr.push(0x00); // Copy
    }
    hdr.push(K_CODERS_UNPACK_SIZE);
    for &sz in sizes {
        push_varint(hdr, sz);
    }
    hdr.push(K_END);
    hdr.push(K_END);
}

fn push_single_coder_header(method_id: &[u8], props: &[u8], pack_size: u64, unpack_size: u64) -> Vec<u8> {
    let mut hdr = Vec::new();
    hdr.push(K_HEADER);
    hdr.push(K_MAIN_STREAMS_INFO);

    hdr.push(K_PACK_INFO);
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 1);
    hdr.push(K_SIZE);
    push_varint(&mut hdr, pack_size);
    hdr.push(K_END);

    hdr.push(K_UNPACK_INFO);
    hdr.push(K_FOLDER);
    push_varint(&mut hdr, 1);
    hdr.push(0);
    push_varint(&mut hdr, 1); // 1 coder
    let mut flags = (method_id.len() as u8) & 0x0F;
    if !props.is_empty() {
        flags |= 0x20;
    }
    hdr.push(flags);
    hdr.extend_from_slice(method_id);
    if !props.is_empty() {
        push_varint(&mut hdr, props.len() as u64);
        hdr.extend_from_slice(props);
    }
    hdr.push(K_CODERS_UNPACK_SIZE);
    push_varint(&mut hdr, unpack_size);
    hdr.push(K_END);
    hdr.push(K_END);

    push_files_info_simple(&mut hdr, &["a.bin"]);
    hdr.push(K_END);
    hdr
}

fn assert_err_no_panic(label: &str, bytes: &[u8]) {
    let res = panic::catch_unwind(panic::AssertUnwindSafe(|| sevenz::read(bytes)));
    match res {
        Err(_) => panic!("{label}: panicked instead of returning Err"),
        Ok(Ok(_)) => panic!("{label}: unexpectedly succeeded on malformed input"),
        Ok(Err(_)) => {}
    }
}

#[test]
fn test_valid_baseline_roundtrip() {
    let payload = b"hello world 7z!";
    let mut hdr = vec![K_HEADER];
    push_simple_copy_streams_info(&mut hdr, &[payload.len() as u64]);
    push_files_info_simple(&mut hdr, &["hello.txt"]);
    hdr.push(K_END);
    let archive = wrap_7z(payload, &hdr);
    let entries = sevenz::read(&archive).expect("valid baseline archive should decode");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "hello.txt");
    assert_eq!(entries[0].data, payload);
}

// --- Class 1: Decompression bombs & size consistency ---

#[test]
fn bomb_single_file_exceeds_max_unpacked() {
    let hdr = push_single_coder_header(&[0x00], &[], 16, ONE_GIB + 1);
    let archive = wrap_7z(&[0u8; 16], &hdr);
    assert_err_no_panic("single file > MAX_UNPACKED", &archive);
}

#[test]
fn bomb_multi_folder_sum_exceeds_max_unpacked() {
    let mut hdr = vec![K_HEADER];
    let half_plus = 600 * 1024 * 1024;
    hdr.push(K_MAIN_STREAMS_INFO);
    hdr.push(K_PACK_INFO);
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 2);
    hdr.push(K_SIZE);
    push_varint(&mut hdr, 4);
    push_varint(&mut hdr, 4);
    hdr.push(K_END);

    hdr.push(K_UNPACK_INFO);
    hdr.push(K_FOLDER);
    push_varint(&mut hdr, 2);
    hdr.push(0);
    for _ in 0..2 {
        push_varint(&mut hdr, 1);
        hdr.push(0x23); // simple + has_props, len=3
        hdr.extend_from_slice(&[0x03, 0x01, 0x01]); // LZMA
        push_varint(&mut hdr, 5);
        hdr.push(93); // lc=3, lp=0, pb=2
        hdr.extend_from_slice(&4096u32.to_le_bytes());
    }
    hdr.push(K_CODERS_UNPACK_SIZE);
    push_varint(&mut hdr, half_plus);
    push_varint(&mut hdr, half_plus);
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a.bin", "b.bin"]);
    hdr.push(K_END);

    let archive = wrap_7z(&[0u8; 8], &hdr);
    assert_err_no_panic("multi-folder sum > MAX_UNPACKED", &archive);
}

#[test]
fn bomb_encoded_header_exceeds_max_unpacked() {
    let mut hdr = vec![K_ENCODED_HEADER];
    hdr.push(K_PACK_INFO);
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 1);
    hdr.push(K_SIZE);
    push_varint(&mut hdr, 8);
    hdr.push(K_END);

    hdr.push(K_UNPACK_INFO);
    hdr.push(K_FOLDER);
    push_varint(&mut hdr, 1);
    hdr.push(0);
    push_varint(&mut hdr, 1);
    hdr.push(0x23);
    hdr.extend_from_slice(&[0x03, 0x01, 0x01]);
    push_varint(&mut hdr, 5);
    hdr.push(93);
    hdr.extend_from_slice(&4096u32.to_le_bytes());
    hdr.push(K_CODERS_UNPACK_SIZE);
    push_varint(&mut hdr, ONE_GIB + 1);
    hdr.push(K_END);
    hdr.push(K_END);

    let archive = wrap_7z(&[0u8; 8], &hdr);
    assert_err_no_panic("encoded header > MAX_UNPACKED", &archive);
}

#[test]
fn bomb_lzma_u64_max_unpack_size() {
    // Raw LZMA1 stream (lc=3, lp=0, pb=2, dict=4096) encoding 1024 zero bytes.
    let lzma_stream: &[u8] = &[0, 0, 111, 253, 255, 255, 163, 183, 128, 12, 130, 219, 255, 255, 159, 240, 0, 0];

    let mut props = vec![93u8];
    props.extend_from_slice(&4096u32.to_le_bytes());
    let hdr = push_single_coder_header(&[0x03, 0x01, 0x01], &props, lzma_stream.len() as u64, u64::MAX);
    let archive = wrap_7z(lzma_stream, &hdr);
    assert_err_no_panic("LZMA u64::MAX unpack size", &archive);
}

#[test]
fn bomb_lzma2_stream_larger_than_declared_unpack_size() {
    // Valid LZMA2 stream (uncompressed chunk of 1024 bytes + 0x00 end) while header declares 10 bytes.
    let mut lzma2_stream = Vec::with_capacity(1028);
    lzma2_stream.push(0x01); // uncompressed chunk, reset dict
    lzma2_stream.push(0x03); // (1024 - 1) >> 8
    lzma2_stream.push(0xFF); // (1024 - 1) & 0xFF
    lzma2_stream.extend(std::iter::repeat_n(0x41u8, 1024));
    lzma2_stream.push(0x00); // end of stream

    let hdr = push_single_coder_header(&[0x21], &[0x00], lzma2_stream.len() as u64, 10);
    let archive = wrap_7z(&lzma2_stream, &hdr);
    // Must never produce > 10 bytes; bounded reader either stops at 10 or errors on excess.
    let res = panic::catch_unwind(panic::AssertUnwindSafe(|| sevenz::read(&archive)));
    match res {
        Err(_) => panic!("panicked on LZMA2 larger than declared"),
        Ok(Ok(entries)) => assert!(entries[0].data.len() <= 10),
        Ok(Err(_)) => {}
    }
}

#[test]
fn lzma2_invalid_chunk_lc_lp() {
    // LZMA2 chunk with control=0xE0 (reset everything), uncomp=1, comp=6, props=5 (lc=5, lp=0 -> lc+lp > 4)
    let lzma2_stream = [0xE0, 0x00, 0x00, 0x00, 0x05, 5, 0, 0, 0, 0, 0, 0];
    let hdr = push_single_coder_header(&[0x21], &[0x00], lzma2_stream.len() as u64, 1);
    let archive = wrap_7z(&lzma2_stream, &hdr);
    assert_err_no_panic("LZMA2 chunk lc+lp > 4", &archive);
}

#[test]
fn substream_sizes_exceed_folder_size() {
    let mut hdr = vec![K_HEADER];
    push_simple_copy_streams_info(&mut hdr, &[10]);
    // Replace trailing K_END of streams_info with K_SUB_STREAMS_INFO + K_END
    hdr.pop();
    hdr.push(K_SUB_STREAMS_INFO);
    hdr.push(K_NUM_UNPACK_STREAM);
    push_varint(&mut hdr, 2);
    hdr.push(K_SIZE);
    push_varint(&mut hdr, 11); // > folder total 10
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a", "b"]);
    hdr.push(K_END);

    let archive = wrap_7z(&[0u8; 10], &hdr);
    assert_err_no_panic("substream sizes > folder size", &archive);
}

#[test]
fn substream_sizes_u64_overflow() {
    let mut hdr = vec![K_HEADER];
    push_simple_copy_streams_info(&mut hdr, &[10]);
    hdr.pop();
    hdr.push(K_SUB_STREAMS_INFO);
    hdr.push(K_NUM_UNPACK_STREAM);
    push_varint(&mut hdr, 3);
    hdr.push(K_SIZE);
    push_varint(&mut hdr, u64::MAX - 5);
    push_varint(&mut hdr, 10); // wraps sum to 4 <= 10!
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a", "b", "c"]);
    hdr.push(K_END);

    let archive = wrap_7z(&[0u8; 10], &hdr);
    assert_err_no_panic("substream sizes u64 overflow", &archive);
}

#[test]
fn substream_missing_k_size_when_multiple_substreams() {
    let mut hdr = vec![K_HEADER];
    push_simple_copy_streams_info(&mut hdr, &[10]);
    hdr.pop();
    hdr.push(K_SUB_STREAMS_INFO);
    hdr.push(K_NUM_UNPACK_STREAM);
    push_varint(&mut hdr, 2);
    // Missing K_SIZE!
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a", "b"]);
    hdr.push(K_END);

    let archive = wrap_7z(&[0u8; 10], &hdr);
    assert_err_no_panic("missing K_SIZE for 2 substreams", &archive);
}

#[test]
fn substream_zero_count_with_nonzero_folder_size() {
    let mut hdr = vec![K_HEADER];
    push_simple_copy_streams_info(&mut hdr, &[10]);
    hdr.pop();
    hdr.push(K_SUB_STREAMS_INFO);
    hdr.push(K_NUM_UNPACK_STREAM);
    push_varint(&mut hdr, 0);
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &[]);
    hdr.push(K_END);

    let archive = wrap_7z(&[0u8; 10], &hdr);
    assert_err_no_panic("0 substreams with nonzero folder size", &archive);
}

#[test]
fn substream_zero_count_with_streamed_file() {
    let mut hdr = vec![K_HEADER];
    push_simple_copy_streams_info(&mut hdr, &[0]);
    hdr.pop();
    hdr.push(K_SUB_STREAMS_INFO);
    hdr.push(K_NUM_UNPACK_STREAM);
    push_varint(&mut hdr, 0);
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a"]);
    hdr.push(K_END);

    let archive = wrap_7z(&[], &hdr);
    assert_err_no_panic("0 substreams with 1 streamed file", &archive);
}

// --- Class 2: LZMA/LZMA2 dictionary size limit & property validation ---

#[test]
fn lzma_dict_size_exceeds_256mib() {
    let lzma_stream: &[u8] = &[0, 0, 111, 253, 255, 255, 163, 183, 128, 12, 130, 219, 255, 255, 159, 240, 0, 0];
    let mut props = vec![93u8];
    props.extend_from_slice(&(MIB_256 + 1).to_le_bytes());
    let hdr = push_single_coder_header(&[0x03, 0x01, 0x01], &props, lzma_stream.len() as u64, 1024);
    let archive = wrap_7z(lzma_stream, &hdr);
    assert_err_no_panic("LZMA dict > 256 MiB", &archive);
}

#[test]
fn lzma_invalid_props_byte() {
    let mut props = vec![225u8]; // pb = 5 (out of range 0..=4)
    props.extend_from_slice(&4096u32.to_le_bytes());
    let hdr = push_single_coder_header(&[0x03, 0x01, 0x01], &props, 5, 4);
    let archive = wrap_7z(&[0u8; 5], &hdr);
    assert_err_no_panic("LZMA invalid props 225", &archive);
}

#[test]
fn lzma_short_props() {
    let hdr = push_single_coder_header(&[0x03, 0x01, 0x01], &[93, 0, 0, 1], 5, 4);
    let archive = wrap_7z(&[0u8; 5], &hdr);
    assert_err_no_panic("LZMA short props", &archive);
}

#[test]
fn lzma2_dict_size_exceeds_256mib() {
    let lzma2_valid = [0x01, 0x00, 0x03, 0x41, 0x41, 0x41, 0x41, 0x00];
    for prop in [33u8, 36, 40, 41, 0x40] {
        let hdr = push_single_coder_header(&[0x21], &[prop], lzma2_valid.len() as u64, 4);
        let archive = wrap_7z(&lzma2_valid, &hdr);
        assert_err_no_panic(&format!("LZMA2 invalid/oversized prop {prop}"), &archive);
    }
}

#[test]
fn lzma2_empty_props() {
    let hdr = push_single_coder_header(&[0x21], &[], 4, 4);
    let archive = wrap_7z(&[0u8; 4], &hdr);
    assert_err_no_panic("LZMA2 empty props", &archive);
}

// --- Class 3: Varint overflow, OOM counts, bounds checks ---

#[test]
fn count_huge_num_pack_streams_no_oom() {
    let mut hdr = vec![K_HEADER, K_MAIN_STREAMS_INFO, K_PACK_INFO];
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 1_000_000_000);
    hdr.push(K_CRC);
    hdr.push(1); // all defined = true!
    hdr.push(K_END);
    hdr.push(K_END);
    hdr.push(K_END);

    let archive = wrap_7z(&[], &hdr);
    assert_err_no_panic("huge num_pack_streams with K_CRC all=1", &archive);
}

#[test]
fn count_huge_num_folders_no_oom() {
    let mut hdr = vec![K_HEADER, K_MAIN_STREAMS_INFO, K_UNPACK_INFO, K_FOLDER];
    push_varint(&mut hdr, 1_000_000_000);
    hdr.push(0);
    hdr.push(K_END);
    hdr.push(K_END);

    let archive = wrap_7z(&[], &hdr);
    assert_err_no_panic("huge num_folders", &archive);
}

#[test]
fn count_huge_num_files_no_oom() {
    let mut hdr = vec![K_HEADER, K_FILES_INFO];
    push_varint(&mut hdr, 1_000_000_000);
    hdr.push(K_END);
    hdr.push(K_END);

    let archive = wrap_7z(&[], &hdr);
    assert_err_no_panic("huge num_files", &archive);
}

#[test]
fn overflow_total_in_out_in_non_simple_coder() {
    let mut hdr = vec![K_HEADER, K_MAIN_STREAMS_INFO, K_UNPACK_INFO, K_FOLDER];
    push_varint(&mut hdr, 1); // 1 folder
    hdr.push(0); // external = 0
    push_varint(&mut hdr, 2); // 2 coders
    // coder 0: non-simple, num_in = usize::MAX/2 + 10, num_out = usize::MAX/2 + 10
    hdr.push(0x11); // non-simple, method_id len = 1
    hdr.push(0x00);
    push_varint(&mut hdr, (usize::MAX as u64) / 2 + 10);
    push_varint(&mut hdr, (usize::MAX as u64) / 2 + 10);
    // coder 1: non-simple
    hdr.push(0x11);
    hdr.push(0x00);
    push_varint(&mut hdr, (usize::MAX as u64) / 2 + 10);
    push_varint(&mut hdr, (usize::MAX as u64) / 2 + 10);
    hdr.push(K_END);
    hdr.push(K_END);
    hdr.push(K_END);

    let archive = wrap_7z(&[], &hdr);
    assert_err_no_panic("total_in / total_out overflow", &archive);
}

#[test]
fn overflow_pack_offset_sum() {
    let mut hdr = vec![K_HEADER, K_MAIN_STREAMS_INFO, K_PACK_INFO];
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 2);
    hdr.push(K_SIZE);
    push_varint(&mut hdr, u64::MAX - 5);
    push_varint(&mut hdr, 10);
    hdr.push(K_END);
    // 2 folders
    hdr.push(K_UNPACK_INFO);
    hdr.push(K_FOLDER);
    push_varint(&mut hdr, 2);
    hdr.push(0);
    for _ in 0..2 {
        push_varint(&mut hdr, 1);
        hdr.push(0x01);
        hdr.push(0x00);
    }
    hdr.push(K_CODERS_UNPACK_SIZE);
    push_varint(&mut hdr, 1);
    push_varint(&mut hdr, 1);
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a", "b"]);
    hdr.push(K_END);

    let archive = wrap_7z(&[0u8; 32], &hdr);
    assert_err_no_panic("pack_offset u64 overflow", &archive);
}

#[test]
fn panic_empty_file_before_empty_stream() {
    let mut hdr = vec![K_HEADER, K_FILES_INFO];
    push_varint(&mut hdr, 1); // 1 file
    // K_EMPTY_FILE appears BEFORE K_EMPTY_STREAM
    hdr.push(K_EMPTY_FILE);
    push_varint(&mut hdr, 0);
    // K_EMPTY_STREAM: 1 bit = true (0x80)
    hdr.push(K_EMPTY_STREAM);
    push_varint(&mut hdr, 1);
    hdr.push(0x80);
    push_utf16_names(&mut hdr, &["dir"]);
    hdr.push(K_END);
    hdr.push(K_END);

    let archive = wrap_7z(&[], &hdr);
    assert_err_no_panic("K_EMPTY_FILE before K_EMPTY_STREAM", &archive);
}

#[test]
fn panic_pack_info_missing_k_size() {
    let mut hdr = vec![K_HEADER, K_MAIN_STREAMS_INFO];
    hdr.push(K_PACK_INFO);
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 1); // 1 pack stream, NO K_SIZE!
    hdr.push(K_END);

    hdr.push(K_UNPACK_INFO);
    hdr.push(K_FOLDER);
    push_varint(&mut hdr, 2); // 2 folders -> folder 1 sets first_pack_stream_index = 1 > pack_sizes.len() = 0
    hdr.push(0);
    for _ in 0..2 {
        push_varint(&mut hdr, 1);
        hdr.push(0x01);
        hdr.push(0x00);
    }
    hdr.push(K_CODERS_UNPACK_SIZE);
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 0);
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a", "b"]);
    hdr.push(K_END);

    let archive = wrap_7z(&[], &hdr);
    assert_err_no_panic("pack_info missing K_SIZE", &archive);
}

#[test]
fn archive_properties_huge_skip() {
    let mut hdr = vec![K_HEADER, K_ARCHIVE_PROPERTIES, 0x01];
    push_varint(&mut hdr, u64::MAX);
    hdr.push(K_END);
    let archive = wrap_7z(&[], &hdr);
    assert_err_no_panic("archive properties huge skip", &archive);
}

// --- Class 4: Invalid references ---

fn build_two_coder_archive(bind_in: u64, bind_out: u64) -> Vec<u8> {
    let mut hdr = vec![K_HEADER, K_MAIN_STREAMS_INFO];
    hdr.push(K_PACK_INFO);
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 1);
    hdr.push(K_SIZE);
    push_varint(&mut hdr, 4);
    hdr.push(K_END);

    hdr.push(K_UNPACK_INFO);
    hdr.push(K_FOLDER);
    push_varint(&mut hdr, 1);
    hdr.push(0);
    push_varint(&mut hdr, 2); // 2 coders
    for _ in 0..2 {
        hdr.push(0x01);
        hdr.push(0x00);
    }
    push_varint(&mut hdr, bind_in);
    push_varint(&mut hdr, bind_out);
    hdr.push(K_CODERS_UNPACK_SIZE);
    push_varint(&mut hdr, 4);
    push_varint(&mut hdr, 4);
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a"]);
    hdr.push(K_END);
    wrap_7z(&[0u8; 4], &hdr)
}

#[test]
fn invalid_ref_bind_pair_in_out_of_range() {
    assert_err_no_panic("bind_pair in_index out of range", &build_two_coder_archive(99, 0));
}

#[test]
fn invalid_ref_bind_pair_out_out_of_range() {
    assert_err_no_panic("bind_pair out_index out of range", &build_two_coder_archive(0, 99));
}

#[test]
fn invalid_ref_bind_pair_self_loop_cycle() {
    assert_err_no_panic("bind_pair self-loop cycle", &build_two_coder_archive(0, 0));
}

#[test]
fn invalid_ref_bind_pair_disconnected_cycle() {
    // 3 coders: bind pairs (in=0, out=1) and (in=1, out=0).
    // Packed stream is 2 -> coder 2 runs cleanly, never visiting the cycle {0, 1}!
    let mut hdr = vec![K_HEADER, K_MAIN_STREAMS_INFO];
    hdr.push(K_PACK_INFO);
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 1);
    hdr.push(K_SIZE);
    push_varint(&mut hdr, 4);
    hdr.push(K_END);

    hdr.push(K_UNPACK_INFO);
    hdr.push(K_FOLDER);
    push_varint(&mut hdr, 1);
    hdr.push(0);
    push_varint(&mut hdr, 3); // 3 coders
    for _ in 0..3 {
        hdr.push(0x01);
        hdr.push(0x00);
    }
    // 2 bind pairs forming a cycle on coders 0 and 1
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 1);
    push_varint(&mut hdr, 1);
    push_varint(&mut hdr, 0);

    hdr.push(K_CODERS_UNPACK_SIZE);
    push_varint(&mut hdr, 4);
    push_varint(&mut hdr, 4);
    push_varint(&mut hdr, 4);
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a"]);
    hdr.push(K_END);

    let archive = wrap_7z(&[0u8; 4], &hdr);
    assert_err_no_panic("disconnected bind_pair cycle", &archive);
}

#[test]
fn invalid_ref_bind_pair_duplicate_indices() {
    // 3 coders, 2 bind pairs both using out=0 (and in=0, in=1)
    let mut hdr = vec![K_HEADER, K_MAIN_STREAMS_INFO];
    hdr.push(K_PACK_INFO);
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 1);
    hdr.push(K_SIZE);
    push_varint(&mut hdr, 4);
    hdr.push(K_END);

    hdr.push(K_UNPACK_INFO);
    hdr.push(K_FOLDER);
    push_varint(&mut hdr, 1);
    hdr.push(0);
    push_varint(&mut hdr, 3);
    for _ in 0..3 {
        hdr.push(0x01);
        hdr.push(0x00);
    }
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 1);
    push_varint(&mut hdr, 0);

    hdr.push(K_CODERS_UNPACK_SIZE);
    push_varint(&mut hdr, 4);
    push_varint(&mut hdr, 4);
    push_varint(&mut hdr, 4);
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a"]);
    hdr.push(K_END);

    let archive = wrap_7z(&[0u8; 4], &hdr);
    assert_err_no_panic("duplicate bind_pair out_index", &archive);
}

#[test]
fn invalid_ref_packed_stream_out_of_range() {
    // 1 non-simple coder with num_in = 2, num_out = 1 -> 2 packed streams, index 99 out of range
    let mut hdr = vec![K_HEADER, K_MAIN_STREAMS_INFO];
    hdr.push(K_PACK_INFO);
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 2);
    hdr.push(K_SIZE);
    push_varint(&mut hdr, 4);
    push_varint(&mut hdr, 4);
    hdr.push(K_END);

    hdr.push(K_UNPACK_INFO);
    hdr.push(K_FOLDER);
    push_varint(&mut hdr, 1);
    hdr.push(0);
    push_varint(&mut hdr, 1);
    hdr.push(0x11); // non-simple
    hdr.push(0x00);
    push_varint(&mut hdr, 2); // num_in
    push_varint(&mut hdr, 1); // num_out
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 99); // out of range!
    hdr.push(K_CODERS_UNPACK_SIZE);
    push_varint(&mut hdr, 4);
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a"]);
    hdr.push(K_END);

    let archive = wrap_7z(&[0u8; 8], &hdr);
    assert_err_no_panic("packed_stream index out of range", &archive);
}

#[test]
fn invalid_ref_missing_pack_info_and_missing_streams() {
    // Folder exists, no K_PACK_INFO
    let mut hdr = vec![K_HEADER, K_MAIN_STREAMS_INFO, K_UNPACK_INFO, K_FOLDER];
    push_varint(&mut hdr, 1);
    hdr.push(0);
    push_varint(&mut hdr, 1);
    hdr.push(0x01);
    hdr.push(0x00);
    hdr.push(K_CODERS_UNPACK_SIZE);
    push_varint(&mut hdr, 4);
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a"]);
    hdr.push(K_END);
    assert_err_no_panic("folder with missing pack_info", &wrap_7z(&[0u8; 4], &hdr));

    // More streamed files than substreams
    let mut hdr2 = vec![K_HEADER];
    push_simple_copy_streams_info(&mut hdr2, &[4]);
    push_files_info_simple(&mut hdr2, &["a", "b"]);
    hdr2.push(K_END);
    assert_err_no_panic("more streamed files than folder streams", &wrap_7z(&[0u8; 4], &hdr2));
}

#[test]
fn invalid_ref_folder_zero_packed_streams() {
    let mut hdr = vec![K_HEADER, K_MAIN_STREAMS_INFO];
    hdr.push(K_PACK_INFO);
    push_varint(&mut hdr, 0);
    push_varint(&mut hdr, 0);
    hdr.push(K_END);

    hdr.push(K_UNPACK_INFO);
    hdr.push(K_FOLDER);
    push_varint(&mut hdr, 1);
    hdr.push(0);
    push_varint(&mut hdr, 1);
    hdr.push(0x11); // non-simple
    hdr.push(0x00);
    push_varint(&mut hdr, 0); // num_in = 0 -> num_packed_streams = 0
    push_varint(&mut hdr, 1); // num_out = 1
    hdr.push(K_CODERS_UNPACK_SIZE);
    push_varint(&mut hdr, 0);
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a"]);
    hdr.push(K_END);
    assert_err_no_panic("folder with 0 packed streams", &wrap_7z(&[], &hdr));
}

#[test]
fn substream_missing_k_size_two_folders_mismatch() {
    // Folder 0 has 2 substreams, folder 1 has 1 substream, K_SIZE omitted.
    let mut hdr = vec![K_HEADER];
    push_simple_copy_streams_info(&mut hdr, &[4, 4]);
    hdr.pop();
    hdr.push(K_SUB_STREAMS_INFO);
    hdr.push(K_NUM_UNPACK_STREAM);
    push_varint(&mut hdr, 2);
    push_varint(&mut hdr, 1);
    hdr.push(K_END);
    hdr.push(K_END);
    push_files_info_simple(&mut hdr, &["a", "b"]);
    hdr.push(K_END);
    assert_err_no_panic("missing K_SIZE with 2 folders", &wrap_7z(&[0u8; 8], &hdr));
}

#[test]
fn lzma_and_lzma2_extra_props_rejected() {
    let lzma_stream: &[u8] = &[0, 0, 111, 253, 255, 255, 163, 183, 128, 12, 130, 219, 255, 255, 159, 240, 0, 0];
    let mut props6 = vec![93u8];
    props6.extend_from_slice(&4096u32.to_le_bytes());
    props6.push(0); // extra 6th byte
    let hdr1 = push_single_coder_header(&[0x03, 0x01, 0x01], &props6, lzma_stream.len() as u64, 1024);
    assert_err_no_panic("LZMA 6-byte props", &wrap_7z(lzma_stream, &hdr1));

    let lzma2_valid = [0x01, 0x00, 0x03, 0x41, 0x41, 0x41, 0x41, 0x00];
    let hdr2 = push_single_coder_header(&[0x21], &[0x00, 0x00], lzma2_valid.len() as u64, 4);
    assert_err_no_panic("LZMA2 2-byte props", &wrap_7z(&lzma2_valid, &hdr2));
}

// --- Class 5: Nested encoded headers ---

#[test]
fn nested_encoded_header_rejected() {
    // Inner valid plain header
    let payload = b"ok";
    let mut inner_plain_hdr = vec![K_HEADER];
    push_simple_copy_streams_info(&mut inner_plain_hdr, &[payload.len() as u64]);
    push_files_info_simple(&mut inner_plain_hdr, &["ok.txt"]);
    inner_plain_hdr.push(K_END);

    // Middle encoded header (Copy-encoded inner_plain_hdr)
    let mut mid_enc_hdr = vec![K_ENCODED_HEADER];
    mid_enc_hdr.push(K_PACK_INFO);
    push_varint(&mut mid_enc_hdr, payload.len() as u64);
    push_varint(&mut mid_enc_hdr, 1);
    hdr_append_copy_folder(&mut mid_enc_hdr, inner_plain_hdr.len() as u64);

    // Outer encoded header (Copy-encoded mid_enc_hdr)
    let mut packed = Vec::new();
    packed.extend_from_slice(payload);
    let inner_offset = packed.len() as u64;
    packed.extend_from_slice(&inner_plain_hdr);
    let mid_offset = packed.len() as u64;
    packed.extend_from_slice(&mid_enc_hdr);

    // Fix mid_enc_hdr pack_pos to inner_offset
    let mut fixed_mid_enc_hdr = vec![K_ENCODED_HEADER];
    fixed_mid_enc_hdr.push(K_PACK_INFO);
    push_varint(&mut fixed_mid_enc_hdr, inner_offset);
    push_varint(&mut fixed_mid_enc_hdr, 1);
    hdr_append_copy_folder(&mut fixed_mid_enc_hdr, inner_plain_hdr.len() as u64);

    let mut packed2 = Vec::new();
    packed2.extend_from_slice(payload);
    packed2.extend_from_slice(&inner_plain_hdr);
    let mid2_offset = packed2.len() as u64;
    packed2.extend_from_slice(&fixed_mid_enc_hdr);

    let mut outer_enc_hdr = vec![K_ENCODED_HEADER];
    outer_enc_hdr.push(K_PACK_INFO);
    push_varint(&mut outer_enc_hdr, mid2_offset);
    push_varint(&mut outer_enc_hdr, 1);
    hdr_append_copy_folder(&mut outer_enc_hdr, fixed_mid_enc_hdr.len() as u64);

    let archive = wrap_7z(&packed2, &outer_enc_hdr);
    assert_err_no_panic("nested encoded header", &archive);
    let _ = mid_offset;
}

fn hdr_append_copy_folder(hdr: &mut Vec<u8>, size: u64) {
    hdr.push(K_SIZE);
    push_varint(hdr, size);
    hdr.push(K_END);
    hdr.push(K_UNPACK_INFO);
    hdr.push(K_FOLDER);
    push_varint(hdr, 1);
    hdr.push(0);
    push_varint(hdr, 1);
    hdr.push(0x01);
    hdr.push(0x00);
    hdr.push(K_CODERS_UNPACK_SIZE);
    push_varint(hdr, size);
    hdr.push(K_END);
    hdr.push(K_END);
}

// --- Class 6: Deterministic seeded mutation test on 20 real library files ---

struct Xorshift64(u64);
impl Xorshift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn read_vm_hwm_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb);
        }
    }
    None
}

#[test]
fn mutation_fuzz_20_real_files_x_2000() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let list_path = manifest_dir.join("../../scratch/t/all_sbsar.txt");
    // the private library list exists only in the research checkout
    let Ok(list_text) = std::fs::read_to_string(&list_path) else {
        eprintln!("skipped: {} not found", list_path.display());
        return;
    };
    let mut files_with_sizes: Vec<(u64, PathBuf)> = list_text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .filter_map(|l| {
            let p = PathBuf::from(l);
            let sz = std::fs::metadata(&p).ok()?.len();
            Some((sz, p))
        })
        .collect();
    files_with_sizes.sort_by_key(|(sz, _)| *sz);
    assert!(files_with_sizes.len() >= 20, "expected >= 20 sbsar files");

    // Pick 20 evenly-spaced files across the library (from smallest up to ~10 MiB so 40,000 runs finish quickly)
    let eligible: Vec<PathBuf> = files_with_sizes
        .into_iter()
        .filter(|(sz, _)| *sz <= 10 * 1024 * 1024)
        .map(|(_, p)| p)
        .collect();
    let step = eligible.len() / 20;
    let selected: Vec<PathBuf> = (0..20).map(|i| eligible[i * step].clone()).collect();

    let mut rng = Xorshift64(0x5E7E_0727_2026_1115);
    let mut panics = 0usize;
    let mut total_cases = 0usize;
    let mut max_time = std::time::Duration::ZERO;

    let suite_start = Instant::now();
    for path in &selected {
        let orig = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert!(orig.len() >= 32);

        for iter in 0..2000 {
            total_cases += 1;
            let mut mutated = orig.clone();
            let mode = rng.next() % 10;
            if mode < 2 {
                // Truncation
                let new_len = (rng.next() as usize) % mutated.len();
                mutated.truncate(new_len);
            } else if mode < 6 {
                // Raw random byte flips (1..=4 bytes)
                let flips = 1 + (rng.next() as usize % 4);
                for _ in 0..flips {
                    let idx = (rng.next() as usize) % mutated.len();
                    mutated[idx] = (rng.next() & 0xFF) as u8;
                }
            } else {
                // Mutate inside header (or start header) and repair CRC so parser/decoder is deeply exercised
                let next_off = u64::from_le_bytes(orig[12..20].try_into().unwrap()) as usize;
                let next_sz = u64::from_le_bytes(orig[20..28].try_into().unwrap()) as usize;
                let hdr_start = 32usize.saturating_add(next_off);
                let hdr_end = hdr_start.saturating_add(next_sz);
                if hdr_end <= mutated.len() && hdr_start < hdr_end {
                    let flips = 1 + (rng.next() as usize % 3);
                    for _ in 0..flips {
                        let idx = hdr_start + ((rng.next() as usize) % (hdr_end - hdr_start));
                        mutated[idx] = (rng.next() & 0xFF) as u8;
                    }
                    let new_next_crc = crc32fast::hash(&mutated[hdr_start..hdr_end]).to_le_bytes();
                    mutated[28..32].copy_from_slice(&new_next_crc);
                    let new_start_crc = crc32fast::hash(&mutated[12..32]).to_le_bytes();
                    mutated[8..12].copy_from_slice(&new_start_crc);
                } else {
                    let idx = (rng.next() as usize) % mutated.len();
                    mutated[idx] ^= 0xFF;
                }
            }
            let _ = iter;

            let t0 = Instant::now();
            let res = panic::catch_unwind(panic::AssertUnwindSafe(|| sevenz::read(&mutated)));
            let dt = t0.elapsed();
            if dt > max_time {
                max_time = dt;
            }
            if res.is_err() {
                panics += 1;
            }
        }
    }

    let total_elapsed = suite_start.elapsed();
    let vms_kb = read_vm_hwm_kb().unwrap_or(0);
    eprintln!(
        "mutation_fuzz: cases={} panics={} max_case_ms={:.2} total_s={:.2} VmHWM_MB={:.1}",
        total_cases,
        panics,
        max_time.as_secs_f64() * 1000.0,
        total_elapsed.as_secs_f64(),
        vms_kb as f64 / 1024.0
    );
    assert_eq!(panics, 0, "mutation fuzzing triggered {panics} panics");
}
