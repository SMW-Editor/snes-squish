//! # Optimal SNES compression library
//!
//! This library implements some compression formats commonly found in SNES
//! games. It is optimized towards minimum output size, and achieves guaranteed
//! optimal output for all inputs. The performance is probably worse than other
//! compressors, but should still be fast enough for most uses. The absolute
//! worst-case 32KB input the author is aware of takes 700ms to compress, but
//! most real-world inputs should take under 100ms. (This worst case only occurs
//! with LZ3: with LZ2, the same input takes 270ms).
//!
//! See the [`Algorithm`] enum for implemented algorithms. For basic usage, you
//! can just pass an Algorithm variant along with your data to [`compress`] or
//! [`decompress`].
//!
//! You can also do a kind of "shared-dictionary" compression if you know there
//! is some data that's likely to be repeated in multiple compressed files. This
//! is implemented by putting some data (the dictionary) at the beginning of the
//! output file before decompressing, allowing the compressed data to make
//! backreferences into the dictionary. To compress files like this, prepend the
//! dictionary to your input data and use [`compress_into`], setting `offset` to
//! the size of the dictionary. When decompressing, use [`decompress_into`],
//! and make sure the output buffer already contains the dictionary.

/// Which compression algorithm to use
#[non_exhaustive]
#[derive(Debug, Copy, Clone)]
pub enum Algorithm {
    /// LC_LZ2 compression, as used in SMW, YI, ALttP, and probably others. Of
    /// the implemented algorithms, this is the fastest to decompress, but has
    /// the worst compression ratio.
    LZ2,
    /// LC_LZ3 compression, used in Pokemon G&S, and commonly in SMW romhacks.
    /// Slow to decompress, but has the best ratio.
    LZ3,
}

// re-export the variants for convenience
pub use Algorithm::*;

// Internal constants that correspond to Algorithm variants.
// We can't use Algorithm directly because const generics are more limited in
// stable rust, so we use these values internally and have the public compress
// function dispatch to the right implementation manually.
const ALG_LZ2: u8 = 0;
const ALG_LZ3: u8 = 1;

// these functions made more sense when I tried to implement HAL compression
// too. but keeping them around doesn't really make things any messier.

/// whether the algorithm supports short relative backreference commands
const fn has_relative_backref(alg: u8) -> bool {
    return alg == ALG_LZ3;
}
/// whether the algorithm has zero-fill instead of inc-fill as command 3
const fn has_zerofill(alg: u8) -> bool {
    return alg == ALG_LZ3;
}
/// whether the algorithm has bit-reverse and reverse-order backreference
/// commands
const fn has_alt_backref(alg: u8) -> bool {
    return alg != ALG_LZ2;
}

fn next_byte(data: &mut &[u8]) -> Option<u8> {
    let (res, rest) = data.split_first()?;
    *data = rest;
    Some(*res)
}
fn next_chunk<'a>(data: &mut &'a [u8], size: usize) -> Option<&'a [u8]> {
    if data.len() > size {
        let (res, rest) = data.split_at(size);
        *data = rest;
        Some(res)
    } else {
        None
    }
}
fn next_word_be(data: &mut &[u8]) -> Option<u16> {
    let d = next_chunk(data, 2)?;
    Some(u16::from_be_bytes(d.try_into().ok()?))
}

#[derive(Debug, Copy, Clone)]
struct Packet<'a> {
    len: usize,
    kind: PacketKind<'a>,
}
#[derive(Debug, Copy, Clone)]
enum BackrefKind {
    // 0..=127, corresponds to actual offset 1..=128
    Rel(u8),
    // 0..=32767 (or ..=65535 for non-lz3), straight offset
    Abs(u16),
}

impl BackrefKind {
    fn encode<const ALG: u8>(&self, buf: &mut Vec<u8>) {
        match &self {
            BackrefKind::Rel(x) => {
                assert!(has_relative_backref(ALG));
                assert!(*x < 128);
                buf.push(x | 0x80);
            }
            BackrefKind::Abs(x) => {
                if has_relative_backref(ALG) {
                    assert!(*x < 32678);
                }
                buf.extend_from_slice(&x.to_be_bytes());
            }
        }
    }
    fn offset(&self, pos: usize) -> usize {
        match &self {
            BackrefKind::Rel(x) => pos - 1 - *x as usize,
            BackrefKind::Abs(x) => *x as usize,
        }
    }
    fn len(&self) -> usize {
        match &self {
            BackrefKind::Rel(_) => 1,
            BackrefKind::Abs(_) => 2,
        }
    }
}
fn parse_backref_kind<const ALG: u8>(data: &mut &[u8]) -> Option<BackrefKind> {
    if !has_relative_backref(ALG) {
        return Some(BackrefKind::Abs(next_word_be(data)?));
    }
    let b = next_byte(data)?;
    if b & 0x80 == 0x80 {
        return Some(BackrefKind::Rel(b & 0x7f));
    } else {
        let b2 = next_byte(data)?;
        return Some(BackrefKind::Abs((b as u16) << 8 | (b2 as u16)));
    }
}

#[derive(Debug, Copy, Clone)]
enum PacketKind<'a> {
    Direct(&'a [u8]),
    ByteFill(u8),
    WordFill(&'a [u8]),
    ZeroFill,
    IncreasingFill(u8),
    Backref(BackrefKind),
    BackwardsBackref(BackrefKind),
    BitRevBackref(BackrefKind),
    Eof,
}

impl<'a> Packet<'a> {
    fn read<const ALG: u8>(data: &mut &'a [u8]) -> Option<Self> {
        let mut header = next_byte(data)? as usize;
        if header == 0xFF {
            return Some(Packet {
                len: 0,
                kind: PacketKind::Eof,
            });
        }
        let (cmd, len) = if header >> 5 == 0b111 {
            header <<= 8;
            header |= next_byte(data)? as usize;
            ((header >> 10) & 0x7, (header & 0x3FF) + 1)
        } else {
            (header >> 5, (header & 0x1F) + 1)
        };
        let kind = match cmd {
            0 => PacketKind::Direct(next_chunk(data, len)?),
            1 => PacketKind::ByteFill(next_byte(data)?),
            2 => PacketKind::WordFill(next_chunk(data, 2)?),
            3 if has_zerofill(ALG) => PacketKind::ZeroFill,
            3 => PacketKind::IncreasingFill(next_byte(data)?),
            4 => PacketKind::Backref(parse_backref_kind::<ALG>(data)?),
            5 if has_alt_backref(ALG) => PacketKind::BitRevBackref(parse_backref_kind::<ALG>(data)?),
            6 if has_alt_backref(ALG) => {
                PacketKind::BackwardsBackref(parse_backref_kind::<ALG>(data)?)
            }
            _ => return None,
        };
        Some(Packet { len, kind })
    }
    fn compress<const ALG: u8>(&self, buf: &mut Vec<u8>) {
        if self.len > 1024 {
            panic!("cannot encode >1024 byte packets");
        }
        if matches!(self.kind, PacketKind::Eof) {
            buf.push(0xFF);
        }
        if self.len == 0 {
            return;
        }
        let len = self.len - 1;
        if len > 0x1F {
            let packed = (0b111 << 13) | (self.cmd() << 10) | len;
            buf.extend_from_slice(&(packed as u16).to_be_bytes());
        } else {
            let packed = self.cmd() << 5 | len;
            buf.push(packed as u8);
        }
        match &self.kind {
            PacketKind::Direct(c) => buf.extend_from_slice(c),
            PacketKind::ByteFill(b) => buf.push(*b),
            PacketKind::WordFill(b) => buf.extend_from_slice(b),
            PacketKind::ZeroFill if has_zerofill(ALG) => {}
            PacketKind::IncreasingFill(b) if !has_zerofill(ALG) => buf.push(*b),
            PacketKind::Backref(b) => b.encode::<ALG>(buf),
            PacketKind::BitRevBackref(b) if has_alt_backref(ALG) => b.encode::<ALG>(buf),
            PacketKind::BackwardsBackref(b) if has_alt_backref(ALG) => b.encode::<ALG>(buf),
            _ => panic!("invalid packet for this algorithm"),
        }
    }
    fn decompress<const ALG: u8>(&self, buf: &mut Vec<u8>) {
        match self.kind {
            PacketKind::Direct(c) => buf.extend_from_slice(c),
            PacketKind::ByteFill(b) => buf.extend((0..self.len).map(|_| b)),
            PacketKind::WordFill(b) => buf.extend((0..).flat_map(|_| b).take(self.len)),
            PacketKind::ZeroFill => buf.extend((0..self.len).map(|_| 0)),
            PacketKind::IncreasingFill(b) => {
                buf.extend((0..self.len).map(|c| b.wrapping_add(c as u8)))
            }
            PacketKind::Backref(b) => {
                let off = b.offset(buf.len());
                for i in off..off + self.len {
                    buf.push(buf[i]);
                }
            }
            PacketKind::BitRevBackref(b) => {
                let off = b.offset(buf.len());
                for i in off..off + self.len {
                    buf.push(buf[i].reverse_bits());
                }
            }
            PacketKind::BackwardsBackref(b) => {
                let off = b.offset(buf.len());
                for i in 0..self.len {
                    buf.push(buf[off - i]);
                }
            }
            PacketKind::Eof => {}
        }
    }
    fn eof() -> Self {
        Packet {
            len: 0,
            kind: PacketKind::Eof,
        }
    }
    fn clen(&self) -> usize {
        let cmd_len = if self.len - 1 > 0x1f { 2 } else { 1 };
        let arg_len = match self.kind {
            PacketKind::Direct(_) => self.len,
            PacketKind::ByteFill(_) => 1,
            PacketKind::WordFill(_) => 2,
            PacketKind::ZeroFill => 0,
            PacketKind::IncreasingFill(_) => 1,
            PacketKind::Backref(x) => x.len(),
            PacketKind::BackwardsBackref(x) => x.len(),
            PacketKind::BitRevBackref(x) => x.len(),
            PacketKind::Eof => unreachable!(),
        };
        cmd_len + arg_len
    }
    fn cmd(&self) -> usize {
        match &self.kind {
            PacketKind::Direct(..) => 0,
            PacketKind::ByteFill(..) => 1,
            PacketKind::WordFill(..) => 2,
            PacketKind::ZeroFill => 3,
            PacketKind::IncreasingFill(..) => 3,
            PacketKind::Backref(..) => 4,
            PacketKind::BitRevBackref(..) => 5,
            PacketKind::BackwardsBackref(..) => 6,
            PacketKind::Eof => unreachable!(),
        }
    }
}

/// Decompress the given `data`, writing the output into `buf`.
///
/// `ALG` should be one of the `ALG_*` constants. Will panic when
/// encountering an invalid backreference in the input.
fn decompress_internal<const ALG: u8>(mut data: &[u8], buf: &mut Vec<u8>) {
    // TODO: better semantics for garbage inputs - currently the panics on bad
    // backreferences are mostly accidental
    while let Some(c) = Packet::read::<ALG>(&mut data) {
        c.decompress::<ALG>(buf);
    }
}

fn _analyze_internal<const ALG: u8>(mut data: &[u8], buf: &mut Vec<u8>) {
    while let Some(c) = Packet::read::<ALG>(&mut data) {
        println!("{:?}", c);
        c.decompress::<ALG>(buf);
    }
}

/// Finds the longest possible backreference command to use at
/// data[needle_pos..]. Returns (command, max_length) if found.
fn find_backref<'a, const ALG: u8>(
    orig_len: usize,
    data: &[u8],
    suff: &[i32],
    inv_suff: &[u32],
    lcp: &[u32],
    needle_pos: usize,
    look_for_short: bool,
) -> Option<(PacketKind<'a>, u32)> {
    // This is possibly the most involved part of the compressor. The (inverse)
    // suffix array is used to locate all occurrences of the needle. We can
    // efficiently enumerate all matches of the needle using the LCP table,
    // which gives how many bytes suff[i] and suff[i-1] have in common: once
    // this is below needle.len(), the respective suffix array entries differ
    // earlier than the end of the needle, i.e. entry does not begin with needle
    // anymore. Also, (for non-LZ2 algorithms), the suffix array is built from
    // not just the data, but also the bit-reverse and reverse-order data, all
    // concatenated together. This means matches for bit-reverse backref and
    // reverse-order backref will show up in the same search as regular
    // backrefs. We can tell which kind of backref to use by where in the
    // original data the suffix array entry starts at.

    // todo: the linear scanning here still makes the worst-case perf kinda
    // suck... also maybe we shouldn't use this at all for relative backrefs

    let start_pos = inv_suff[needle_pos] as usize;
    // how long a match needs to be in order to ever be worth using
    let min_match_len = if look_for_short { 2 } else { 3 };
    let do_i = |i: usize, offset: usize, matchlen: &mut u32| {
        let ind = suff[i] as usize;
        // Since `data` is actually 3 copies of `orig_data` concatenated with
        // different transforms, figure out where in the original data this
        // match occurred in, and what transform was applied.
        let (typ, realind) = if has_alt_backref(ALG) {
            //(ind / L, ind % L)
            let mut x = ind;
            let mut y: u32 = 0;
            if x >= orig_len { x -= orig_len; y += 1; }
            if x >= orig_len { x -= orig_len; y += 1; }
            (y, x)
        } else {
            (0, ind)
        };
        // for backwards: realind is the index into the reversed data where the match starts
        // what we need: position of the *last* byte of the match
        // realind=1 means 2nd to last byte is the first matching
        // so we'd need realrealind = len-2
        let realind = if typ == 1 {
            orig_len - 1 - realind
        } else {
            realind
        };
        let mk_packet = |repk| match typ {
            0 => PacketKind::Backref(repk),
            1 => PacketKind::BackwardsBackref(repk),
            2 => PacketKind::BitRevBackref(repk),
            3.. => unreachable!(),
        };
        *matchlen = (*matchlen).min(lcp[i+offset]);
        // backwards refs must not actually go past the beginning of data.
        // but for LCP purposes we still need to keep the lcp-derived length
        // around
        let real_matchlen = if typ != 1 { *matchlen } else { (*matchlen).min(realind as u32 + 1) };
        if realind < needle_pos && real_matchlen >= min_match_len
        {
            if !has_relative_backref(ALG) {
                if realind <= u16::MAX as usize {
                    *matchlen = real_matchlen;
                    return Some(Some(mk_packet(BackrefKind::Abs(realind as u16))));
                }
            } else {
                if !look_for_short {
                    if realind < 32768 {
                        *matchlen = real_matchlen;
                         return Some(Some(mk_packet(BackrefKind::Abs(realind as u16))));
                    }
                } else {
                    if needle_pos - realind < 129 {
                        let x = needle_pos - realind - 1;
                        *matchlen = real_matchlen;
                        return Some(Some(mk_packet(BackrefKind::Rel(x as u8))));
                    }
                }
            }
        }
        if *matchlen < min_match_len {
            return Some(None);
        }
        None
    };
    let mut matchlen = u32::MAX;
    let mut output = None;
    if let Some(Some(v)) = (start_pos + 1..data.len()).find_map(|i| do_i(i, 0, &mut matchlen)) {
        output = Some((v, matchlen));
    }
    let mut matchlen2 = u32::MAX;
    if let Some(Some(v)) = (0..start_pos).rev().find_map(|i| do_i(i, 1, &mut matchlen2)) {
        if let Some(k) = output {
            if matchlen2 > k.1 {
                output = Some((v, matchlen2));
            }
        } else {
            output = Some((v, matchlen2));
        }
    }
    output
}

// build inverse suffix array given a regular suffix array
fn inv_suffix_array(suff_arr: &[i32]) -> Vec<u32> {
    let mut out = vec![0u32; suff_arr.len()];
    for (i, v) in suff_arr.iter().enumerate() {
        out[*v as usize] = i as u32;
    }
    out
}

fn lcp_len(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .take_while(|(ca, cb)| ca == cb)
        .count() as u32
}

// build longest common prefix table.
// this function adapted from https://github.com/BurntSushi/suffix
// (lcp_lens_linear in src/table.rs), which is licensed under unlicense.
fn build_lcp(data: &[u8], suff: &[i32], inv_suff: &[u32]) -> Vec<u32> {
    let mut lcps = vec![0u32; data.len()];
    let mut len = 0u32;
    for (sufi2, &rank) in inv_suff.iter().enumerate() {
        if rank == 0 {
            continue;
        }
        let sufi1 = suff[(rank - 1) as usize];
        len += lcp_len(
            &data[(sufi1 as u32 + len) as usize..],
            &data[(sufi2 as u32 + len) as usize..],
        );
        lcps[rank as usize] = len;
        if len > 0 {
            len -= 1;
        }
    }
    lcps
}

// this derives a lexicographic Ord, which means when taking the minimum, we'll
// first get the min by cost, and then by ind.
#[derive(Eq, Ord, PartialEq, PartialOrd, Copy, Clone)]
struct SegTreeEntry {
    cost: u32,
    ind: u32,
}

struct OpConstantSize;
impl segment_tree::ops::Operation<SegTreeEntry> for OpConstantSize {
    fn combine(&self, a: &SegTreeEntry, b: &SegTreeEntry) -> SegTreeEntry {
        //*a.min(b)
        if a.cost.cmp(&b.cost).then(b.ind.cmp(&a.ind)).is_lt() { *a } else { *b }
    }
}

struct OpLinearSize;
impl segment_tree::ops::Operation<SegTreeEntry> for OpLinearSize {
    fn combine(&self, a: &SegTreeEntry, b: &SegTreeEntry) -> SegTreeEntry {
        // if the command is linear-size, larger indices are worse, since they
        // need a longer command.
        // so the real cost is cost+ind
        if (a.cost as u64 + a.ind as u64) > (b.cost as u64 + b.ind as u64) { *b } else { *a }
    }
}

struct SegTrees<const ALG: u8> {
    constant: segment_tree::SegmentPoint<SegTreeEntry, OpConstantSize>,
    linear: segment_tree::SegmentPoint<SegTreeEntry, OpLinearSize>,
}

impl<const ALG: u8> SegTrees<ALG> {
    fn new(len: usize) -> Self {
        let len = (len + 1) as u32;
        let constant = segment_tree::SegmentPoint::build((0..len).map(|x| SegTreeEntry { cost: 0, ind: x }).collect(), OpConstantSize);
        let linear = segment_tree::SegmentPoint::build((0..len).map(|x| SegTreeEntry { cost: 0, ind: x }).collect(), OpLinearSize);
        Self {
            constant,
            linear,
        }
    }
    fn update(&mut self, ind: usize, cost: u32) {
        self.constant.modify(ind, SegTreeEntry { cost, ind: ind as u32 });
        self.linear.modify(ind, SegTreeEntry { cost, ind: ind as u32 });
    }
}

/// Compression implementation. Same arguments as compress_into, but takes the
/// algorithm as a const-generic parameter to allow monomorphizing.
fn compress_internal<'a, const ALG: u8>(data: &'a [u8], offset: usize, buf: &mut Vec<u8>) {
    #[derive(Debug, Copy, Clone)]
    struct DPEntry<'b> {
        cost: u32,
        cmd: Packet<'b>,
        prev: u32,
    }
    //let data_len = data.len() as u32;
    // best_for_suffix[i] = best way to encode data[i..]
    let mut best_for_suffix: Vec<DPEntry> = vec![DPEntry { cost: 0, cmd: Packet::eof(), prev: 0 }; data.len() + 1];
    // segment trees for efficiently finding the best spot to continue from.
    // effectively they allow computing "what is the best cost in data[i..j]",
    // which allows choosing the length of the next command so as to minimize
    // total cost.
    // this segment tree uses a different comparison operator, because the "raw
    // data" command scales with the length of data to encode, unlike all other
    // commands whose size is constant.
    let mut segtrees = SegTrees::<ALG>::new(data.len());
    // end of the data is free to encode.
    segtrees.update(data.len(), 0);
    // See find_backref for how all of these data structures are used.
    let mut fuckedupdata = data.to_vec();
    if has_alt_backref(ALG) {
        fuckedupdata.extend(data.iter().rev());
        fuckedupdata.extend(data.iter().map(|c| c.reverse_bits()));
    }
    // suff like suffering amirite
    // (okay, i personally find the suffix arrays to cause less suffering than
    // segment trees. but close enough.)
    let suff = {
        let mut tmp = vec![0i32; fuckedupdata.len()];
        divsufsort::sort_in_place(&fuckedupdata, &mut tmp);
        tmp
    };
    let inv_suff = inv_suffix_array(&suff);
    let lcp = build_lcp(&fuckedupdata, &suff, &inv_suff);

    // current possible lengths of the fill commands.
    let mut bytefill_len = 1_usize;
    let mut wordfill_len = 2_usize;
    let mut incfill_len = 1_usize;

    // TODO document implementation ideas here again
    for i in (offset..data.len()).rev() {
        let mut best = DPEntry { cost: u32::MAX, cmd: Packet::eof(), prev: 0 };
        {
            // direct copy:
            // needs to be handled separately because it uses a different
            // segment tree than the other commands
            let mut best_ind = segtrees.linear.query_noiden(i+1, (i+32).min(data.len())+1);
            let bound = (i+1024).min(data.len()) + 1;
            if i+33 < bound {
                let best_ind2 = segtrees.linear.query_noiden(i+33, bound);
                best_ind = if best_ind2.cost as usize + best_ind2.ind as usize + 1 < best_ind.cost as usize + best_ind.ind as usize {
                    best_ind2
                } else { best_ind };
            }
            let l = best_ind.ind as usize - i;
            debug_assert!(l > 0);
            let cmd = Packet { len: l, kind: PacketKind::Direct(&data[i..i+l as usize]) };
            let newcost = cmd.clen() as u32 + best_ind.cost;
            if newcost < best.cost {
                best = DPEntry { cost: newcost, cmd, prev: best_ind.ind };
            }
        }
        let update = |kind: PacketKind<'a>, max_len: usize, best: &mut DPEntry<'a>| {
            if max_len == 0 { return; }
            // first check short commands
            let upper = (i+max_len).min(data.len());
            let mut best_ind = segtrees.constant.query_noiden(i+1, upper.min(i+32)+1);
            // then long ones
            if i+33 < upper+1 {
                let best_ind2 = segtrees.constant.query_noiden(i+33, upper+1);
                // long commands are 1 byte longer, so they have to be better by
                // at least 1 extra byte
                if best_ind2.cost + 1 < best_ind.cost {
                    best_ind = best_ind2;
                }
            }
            let best_cmd = Packet { len: best_ind.ind as usize - i, kind };
            if (best_cmd.clen() + best_ind.cost as usize) < best.cost as usize {
                *best = DPEntry { cost: best_cmd.clen() as u32 + best_ind.cost, cmd: best_cmd, prev: best_ind.ind };
            }
        };
        if i+1 < data.len() {
            if data[i] == data[i+1] {
                bytefill_len = (bytefill_len + 1).min(1024);
            } else {
                bytefill_len = 1;
            }
            if has_zerofill(ALG) && data[i] == 0 {
                update(PacketKind::ZeroFill, bytefill_len, &mut best);
            } else {
                update(PacketKind::ByteFill(data[i]), bytefill_len, &mut best);
            }
            if !has_zerofill(ALG) {
                if data[i].wrapping_add(1) == data[i+1] {
                    incfill_len = (incfill_len + 1).min(1024);
                } else {
                    incfill_len = 1;
                }
                update(PacketKind::IncreasingFill(data[i]), incfill_len, &mut best);
            }
        }
        if i+2 < data.len() {
            if data[i] == data[i+2] {
                wordfill_len = (wordfill_len + 1).min(1024);
            } else {
                wordfill_len = 2;
            }
            update(PacketKind::WordFill(&data[i..i+2]), wordfill_len, &mut best);
        }
        // long backref
        if let Some((pk, max_len)) = find_backref::<ALG>(data.len(), &fuckedupdata, &suff, &inv_suff, &lcp, i, false) {
            let max = (max_len as usize).min(1024);
            update(pk, max, &mut best);
        }
        // short backref
        if has_relative_backref(ALG) {
            if let Some((pk, max_len)) = find_backref::<ALG>(data.len(), &fuckedupdata, &suff, &inv_suff, &lcp, i, true) {
                let max = (max_len as usize).min(1024);
                update(pk, max, &mut best);
            }
        }
        if matches!(best.cmd.kind, PacketKind::Eof) {
            panic!();
        }
        best_for_suffix[i] = best;
        segtrees.update(i, best.cost);
    }
    //dbg!(best_for_suffix[offset].cost);
    let mut pkts = vec![];
    let mut i = offset;
    loop {
        let pk = best_for_suffix[i].cmd;
        pkts.push(pk);
        if i == data.len() { break; }
        i = best_for_suffix[i].prev as usize;
    }
    //println!("{:?}", &pkts);
    for x in pkts.iter() {
        x.compress::<ALG>(buf);
    }
    //dbg!(buf.len());
}

/// Compress the given `data` with `alg`, writing the output into `buf`.
///
/// Specifying non-zero `offset` will only start compressing from index `offset`
/// into data, but still allows making backreferences to the first `offset`
/// bytes of data, allowing a basic shared dictionary system to be implemented.
pub fn compress_into(alg: Algorithm, data: &[u8], offset: usize, buf: &mut Vec<u8>) {
    match alg {
        LZ2 => compress_internal::<ALG_LZ2>(data, offset, buf),
        LZ3 => compress_internal::<ALG_LZ3>(data, offset, buf),
    }
}

/// Compress the `data` with `alg` and return the compressed data.
pub fn compress(alg: Algorithm, data: &[u8]) -> Vec<u8> {
    let mut out = vec![];
    compress_into(alg, data, 0, &mut out);
    out
}

/// Decompress the given `data` with `alg`, writing the output into `buf`.
///
/// If `data` was compressed using a dictionary, this dictionary should be
/// present at the beginning of `buf`.
///
/// If the compressed data contains illegal backreferences, this function
/// panics.
pub fn decompress_into(alg: Algorithm, data: &[u8], buf: &mut Vec<u8>) {
    match alg {
        LZ2 => decompress_internal::<ALG_LZ2>(data, buf),
        LZ3 => decompress_internal::<ALG_LZ3>(data, buf),
    }
}

/// Decompress the given `data` with `alg` and return the decompressed data.
///
/// If the compressed data contains illegal backreferences, this function
/// panics.
pub fn decompress(alg: Algorithm, data: &[u8]) -> Vec<u8> {
    let mut out = vec![];
    decompress_into(alg, data, &mut out);
    out
}

// currently not exposing analyze because i don't feel like it's particularly
// useful...
