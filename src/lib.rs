pub const LZ2: u8 = 0;
pub const LZ3: u8 = 1;
pub const HAL: u8 = 2;

// "feature macros" for the different compression algorithms
const fn has_relative_rep(alg: u8) -> bool {
    return alg == LZ3;
}
const fn has_zerofill(alg: u8) -> bool {
    return alg == LZ3;
}
const fn has_alt_repeats(alg: u8) -> bool {
    return alg != LZ2;
}
const fn has_double_word(alg: u8) -> bool {
    return alg == HAL;
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
enum RepeatKind {
    // 0..=127, corresponds to actual offset 1..=128
    Rel(u8),
    // 0..=32767 (or ..=65535 for non-lz3), straight offset
    Abs(u16),
}

impl RepeatKind {
    fn encode<const ALG: u8>(&self, buf: &mut Vec<u8>) {
        match &self {
            RepeatKind::Rel(x) => {
                assert!(has_relative_rep(ALG));
                assert!(*x < 128);
                buf.push(x | 0x80);
            }
            RepeatKind::Abs(x) => {
                if has_relative_rep(ALG) {
                    assert!(*x < 32678);
                }
                buf.extend_from_slice(&x.to_be_bytes());
            }
        }
    }
    fn offset(&self, pos: usize) -> usize {
        match &self {
            RepeatKind::Rel(x) => pos - 1 - *x as usize,
            RepeatKind::Abs(x) => *x as usize,
        }
    }
    fn len(&self) -> usize {
        match &self {
            RepeatKind::Rel(_) => 1,
            RepeatKind::Abs(_) => 2,
        }
    }
}
fn parse_repeat_kind<const ALG: u8>(data: &mut &[u8]) -> Option<RepeatKind> {
    if !has_relative_rep(ALG) {
        return Some(RepeatKind::Abs(next_word_be(data)?));
    }
    let b = next_byte(data)?;
    if b & 0x80 == 0x80 {
        return Some(RepeatKind::Rel(b & 0x7f));
    } else {
        let b2 = next_byte(data)?;
        return Some(RepeatKind::Abs((b as u16) << 8 | (b2 as u16)));
    }
}

#[derive(Debug, Copy, Clone)]
enum PacketKind<'a> {
    Direct(&'a [u8]),
    ByteFill(u8),
    WordFill(&'a [u8]),
    ZeroFill,
    IncreasingFill(u8),
    Repeat(RepeatKind),
    BackwardsRepeat(RepeatKind),
    BitRevRepeat(RepeatKind),
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
            4 => PacketKind::Repeat(parse_repeat_kind::<ALG>(data)?),
            5 if has_alt_repeats(ALG) => PacketKind::BitRevRepeat(parse_repeat_kind::<ALG>(data)?),
            6 if has_alt_repeats(ALG) => {
                PacketKind::BackwardsRepeat(parse_repeat_kind::<ALG>(data)?)
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
            PacketKind::Repeat(b) => b.encode::<ALG>(buf),
            PacketKind::BitRevRepeat(b) if has_alt_repeats(ALG) => b.encode::<ALG>(buf),
            PacketKind::BackwardsRepeat(b) if has_alt_repeats(ALG) => b.encode::<ALG>(buf),
            _ => panic!("invalid packet for this algorithm"),
        }
    }
    fn decompress<const ALG: u8>(&self, buf: &mut Vec<u8>) {
        match self.kind {
            PacketKind::Direct(c) => buf.extend_from_slice(c),
            PacketKind::ByteFill(b) => buf.extend((0..self.len).map(|_| b)),
            PacketKind::WordFill(b) => {
                if has_double_word(ALG) {
                    buf.extend((0..).flat_map(|_| b).take(self.len * 2))
                } else {
                    buf.extend((0..).flat_map(|_| b).take(self.len))
                }
            }
            PacketKind::ZeroFill => buf.extend((0..self.len).map(|_| 0)),
            PacketKind::IncreasingFill(b) => {
                buf.extend((0..self.len).map(|c| b.wrapping_add(c as u8)))
            }
            PacketKind::Repeat(b) => {
                let off = b.offset(buf.len());
                for i in off..off + self.len {
                    buf.push(buf[i]);
                }
            }
            PacketKind::BitRevRepeat(b) => {
                let off = b.offset(buf.len());
                for i in off..off + self.len {
                    buf.push(buf[i].reverse_bits());
                }
            }
            PacketKind::BackwardsRepeat(b) => {
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
            PacketKind::Repeat(x) => x.len(),
            PacketKind::BackwardsRepeat(x) => x.len(),
            PacketKind::BitRevRepeat(x) => x.len(),
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
            PacketKind::Repeat(..) => 4,
            PacketKind::BitRevRepeat(..) => 5,
            PacketKind::BackwardsRepeat(..) => 6,
            PacketKind::Eof => unreachable!(),
        }
    }
}

pub fn decompress<const ALG: u8>(mut data: &[u8], buf: &mut Vec<u8>) {
    while let Some(c) = Packet::read::<ALG>(&mut data) {
        c.decompress::<ALG>(buf);
    }
}

pub fn analyze<const ALG: u8>(mut data: &[u8], buf: &mut Vec<u8>) {
    while let Some(c) = Packet::read::<ALG>(&mut data) {
        println!("{:?}", c);
        c.decompress::<ALG>(buf);
    }
}

// find an index into data that starts with needle, and starts before needle_pos.
// suff is suffix array of data. inv_suff is inverse suffix array of data.
// lcp is longest-common-prefix array of data.
fn find_repeat<'a, const ALG: u8>(
    orig_len: usize,
    data: &[u8],
    suff: &[i32],
    inv_suff: &[u32],
    lcp: &[u32],
    needle_pos: usize,
    needle: &[u8],
) -> (Option<PacketKind<'a>>, bool) {
    let start_pos = inv_suff[needle_pos] as usize;
    // if we find a valid short encoding, use that. but keep any long encoding
    // we see as fallback in case there is no short encoding.
    let mut out_long: Option<PacketKind<'a>> = None;
    // whether there is any matching substring at all - regardless of
    // constraints on whether we can actually encode a repeat command from it
    let mut match_exists = false;
    let mut do_i = |i: usize, offset: usize| {
        let ind = suff[i] as usize;
        // i love how cursed this is
        // Since `data` is actually 3 copies of `orig_data` concatenated with
        // different transforms, figure out where in the original data this
        // match occurred in, and what transform was applied.
        let (typ, realind) = if has_alt_repeats(ALG) {
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
        // so we'd need realrealind = len-2 i *think*??
        let realind = if typ == 1 {
            orig_len - 1 - realind
        } else {
            realind
        };
        let mk_packet = |repk| match typ {
            0 => PacketKind::Repeat(repk),
            1 => PacketKind::BackwardsRepeat(repk),
            2 => PacketKind::BitRevRepeat(repk),
            3.. => unreachable!(),
        };
        if realind < needle_pos
                // for backwards repeat, we additionally need to check that we
                // don't go past the start of the input.
                && (typ != 1 || realind+1>=needle.len())
            // lcp[i] = len of common part of suff[i] and suff[i-1]
            && lcp[i+offset] as usize >= needle.len()
        {
            match_exists = true;
            if !has_relative_rep(ALG) {
                if realind <= u16::MAX as usize {
                    // if we don't have short mode, the abs encoding is the best
                    // we can do, so return it immediately
                    return Some(Some(mk_packet(RepeatKind::Abs(realind as u16))));
                }
            } else {
                if realind < 32768 {
                    out_long = Some(mk_packet(RepeatKind::Abs(realind as u16)));
                }
                if needle_pos - realind < 129 {
                    let x = needle_pos - realind - 1;
                    return Some(Some(mk_packet(RepeatKind::Rel(x as u8))));
                }
            }
        }
        // once we have a prefix shorter than the needle, the needle doesn't
        // match anymore.
        if (lcp[i + offset] as usize) < needle.len() {
            return Some(None);
        }
        None
    };
    if let Some(Some(v)) = (start_pos + 1..data.len()).find_map(|i| do_i(i, 0)) {
        return (Some(v), true);
    }
    if let Some(Some(v)) = (0..start_pos).rev().find_map(|i| do_i(i, 1)) {
        return (Some(v), true);
    }
    (out_long, match_exists)
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

pub fn compress<'a, const ALG: u8>(data: &'a [u8], offset: usize, buf: &mut Vec<u8>) {
    #[derive(Debug, Copy, Clone)]
    struct DPEntry<'b> {
        cost: usize,
        cmd: Packet<'b>,
        prev: usize,
    }
    // best_for_prefix[i] is the best way to encode the first i bytes of data
    let mut best_for_prefix: Vec<DPEntry> = Vec::with_capacity(data.len() + 1);
    // dummy packet here to avoid special-casing later.
    // you can encode an empty string at a cost of 0, so makes sense to me.
    for _ in 0..offset + 1 {
        best_for_prefix.push(DPEntry {
            cost: 0,
            cmd: Packet::eof(),
            prev: 0,
        });
    }
    let mut fuckedupdata = data.to_vec();
    if has_alt_repeats(ALG) {
        fuckedupdata.extend(data.iter().rev());
        fuckedupdata.extend(data.iter().map(|c| c.reverse_bits()));
    }
    let suff = {
        let mut tmp = vec![0i32; fuckedupdata.len()];
        divsufsort::sort_in_place(&fuckedupdata, &mut tmp);
        tmp
    };
    let inv_suff = inv_suffix_array(&suff);
    let lcp = build_lcp(&fuckedupdata, &suff, &inv_suff);

    for i in offset + 1..=data.len() {
        let mut best: Option<DPEntry> = None;
        // if we fail to do a bytefill from one prefix, we have 2 different
        // bytes in the prefix, and any longer prefix would have to cover those
        // bytes too, which means the bytefill is guaranteed to fail. so don't
        // even bother checking if we already failed once. same logic goes for
        // incfill.
        let mut bytefill_possible = true;
        let mut zerofill_possible = true;
        let mut incfill_possible = true;
        // wordfill is trickier due to the possible parity of where to start,
        // but with similar logic you can conclude that one failure implies the
        // rest are going to fail too.
        let mut wordfill_possible = true;
        let mut repeat_possible = true;
        // number of possible command types to check for. used to speed up
        // searching once all (non-direct copy) command types are impossible.
        // this is initialized to 4, even though there are 5 _possible
        // variables, because incfill and zerofill are mutually exclusive.
        let mut n_possible = 4;
        // another byte/word/zerofill optimization: don't recheck the entire
        // range every time, only check the first byte. we already keep track of
        // whether the fill was valid in the previous iteration, so the only
        // thing we need to check for is whether the newly added byte fits the
        // previous fill's pattern. this optimization requires that we iterate
        // in this order though.
        for j in (i.saturating_sub(1024)..i).rev() {
            // helper that sets the best packet if it's better than the current one.
            let try_packet = |best: &mut Option<DPEntry<'a>>, len: usize, kind: PacketKind<'a>| {
                let cmd = Packet { len, kind };
                let cost = cmd.clen() + best_for_prefix[j].cost;
                if let Some(b) = best {
                    if cost < b.cost {
                        *best = Some(DPEntry { cost, cmd, prev: j });
                    }
                } else {
                    *best = Some(DPEntry { cost, cmd, prev: j });
                }
            };
            // j<i, so our target range is always at least 1 byte
            let thisdata = &data[j..i];
            let len = thisdata.len();
            // if a bytefill or zerofill is possible for this range, it is
            // always going to be at least as good as a repeat. so we don't need
            // to check for repeats at all.
            // (not guaranteed for wordfill though.)
            let mut check_repeat = repeat_possible;
            // direct:
            try_packet(&mut best, len, PacketKind::Direct(thisdata));
            if n_possible == 0 {
                continue;
            }
            // bytefill:
            // we only need to check the first byte. the rest are already
            // assured by the fact that bytefill_possible is true.
            if bytefill_possible && (len < 2 || thisdata[0] == thisdata[1]) {
                try_packet(&mut best, len, PacketKind::ByteFill(thisdata[0]));
                check_repeat = false;
            } else {
                if bytefill_possible {
                    n_possible -= 1;
                }
                bytefill_possible = false;
            }
            // wordfill:
            if len >= 2 {
                if wordfill_possible && (len < 3 || thisdata[0] == thisdata[2]) {
                    let word = &thisdata[..2];
                    // in HAL, the length on a wordfill is the number of words to write. so we
                    // cannot encode an odd wordfill like in the other formats.
                    if has_double_word(ALG) {
                        if len & 1 == 0 {
                            try_packet(&mut best, len / 2, PacketKind::WordFill(&word));
                            check_repeat = false;
                        }
                    } else {
                        try_packet(&mut best, len, PacketKind::WordFill(&word));
                        // if we don't have short relative repeats, a wordfill
                        // is also at least as good as any repeat.
                        if !has_relative_rep(ALG) {
                            check_repeat = false;
                        }
                    }
                } else {
                    if wordfill_possible {
                        n_possible -= 1;
                    }
                    wordfill_possible = false;
                }
            }
            if has_zerofill(ALG) {
                // zerofill:
                if zerofill_possible && thisdata[0] == 0 {
                    try_packet(&mut best, len, PacketKind::ZeroFill);
                    check_repeat = false;
                } else {
                    if zerofill_possible {
                        n_possible -= 1;
                    }
                    zerofill_possible = false;
                }
            }
            if !has_zerofill(ALG) {
                // incfill:
                if incfill_possible && (len < 2 || thisdata[0].wrapping_add(1) == thisdata[1]) {
                    let byte = thisdata[0];
                    try_packet(&mut best, len, PacketKind::IncreasingFill(byte));
                    check_repeat = false;
                } else {
                    if incfill_possible {
                        n_possible -= 1;
                    }
                    incfill_possible = false;
                }
            }
            // repeat:
            // need to find a previous occurrence of data[j..i] (i.e. starting before j)
            if check_repeat {
                let (s, exists) = find_repeat::<ALG>(
                    data.len(),
                    &fuckedupdata,
                    &suff,
                    &inv_suff,
                    &lcp,
                    j,
                    thisdata,
                );
                if let Some(k) = s {
                    try_packet(&mut best, len, k);
                }
                if !exists {
                    if repeat_possible {
                        n_possible -= 1;
                    }
                    repeat_possible = false;
                }
            }
        }
        best_for_prefix.push(best.unwrap());
    }
    //dbg!(best_for_prefix[data.len()].cost);
    let mut pkts = vec![];
    let mut i = data.len();
    while i > offset {
        let pk = best_for_prefix[i].cmd;
        pkts.push(pk);
        i = best_for_prefix[i].prev;
    }
    pkts.reverse();
    pkts.push(Packet::eof());
    //dbg!(&pkts);
    for x in pkts.iter() {
        x.compress::<ALG>(buf);
    }
    //dbg!(buf.len());
}
