//! LookingGlass LZW, as used by compressed resources (`RDF_LZW`).
//!
//! Mirrors `LzwExpand` in shockmac's `Libraries/RES/Source/LZW.C`: fixed
//! 14-bit codes packed most-significant-bit first, codes 0..=255 are literal
//! bytes, new strings start at 256, `0x3FFE` flushes the string table and
//! `0x3FFF` ends the stream.

const LZW_BITS: u32 = 14;
/// End-of-data code.
const MAX_VALUE: u32 = (1 << LZW_BITS) - 1;
/// Highest code the string table can hold.
const MAX_CODE: u32 = MAX_VALUE - 2;
/// Restart the string table.
const FLUSH_CODE: u32 = MAX_VALUE - 1;

struct BitReader<'a> {
    src: &'a [u8],
    pos: usize,
    buffer: u32,
    count: i32,
}

impl BitReader<'_> {
    /// Next 14-bit code. Past the end of the input it reads zero bytes, as the
    /// original's buffer source would read whatever followed; a well-formed
    /// stream ends with `MAX_VALUE` before then.
    fn code(&mut self) -> u32 {
        while self.count <= 24 {
            let byte = self.src.get(self.pos).copied().unwrap_or(0);
            self.pos += 1;
            self.buffer |= u32::from(byte) << (24 - self.count);
            self.count += 8;
        }
        let code = self.buffer >> (32 - LZW_BITS);
        self.buffer <<= LZW_BITS;
        self.count -= LZW_BITS as i32;
        code
    }

    fn exhausted(&self) -> bool {
        // Allow the reader's look-ahead of up to 4 bytes past the end.
        self.pos > self.src.len() + 4
    }
}

/// Expand an LZW stream. `limit` caps the output (the resource's known
/// uncompressed size), like the original's `destSize`.
pub fn expand(src: &[u8], limit: Option<usize>) -> Vec<u8> {
    let limit = limit.unwrap_or(usize::MAX);
    let mut out = Vec::with_capacity(limit.min(1 << 20));
    let table_len = (MAX_CODE + 1) as usize;
    let mut prefix = vec![0u32; table_len];
    let mut append = vec![0u8; table_len];
    let mut stack: Vec<u8> = Vec::with_capacity(4096);

    let mut bits = BitReader {
        src,
        pos: 0,
        buffer: 0,
        count: 0,
    };

    // Walk a code back to its first byte, pushing the string reversed.
    let decode = |code: u32, prefix: &[u32], append: &[u8], stack: &mut Vec<u8>| {
        let mut code = code;
        while code > 255 {
            stack.push(append[code as usize]);
            code = prefix[code as usize];
        }
        stack.push(code as u8);
    };

    let mut next_code = 256u32;
    let mut old_code = bits.code();
    if old_code == MAX_VALUE || out.len() >= limit {
        return out;
    }
    let mut character = old_code as u8;
    out.push(character);

    loop {
        if bits.exhausted() {
            break;
        }
        let new_code = bits.code();
        if new_code == MAX_VALUE {
            break;
        }
        if new_code == FLUSH_CODE {
            next_code = 256;
            old_code = bits.code();
            if old_code == MAX_VALUE {
                break;
            }
            character = old_code as u8;
            if out.len() >= limit {
                break;
            }
            out.push(character);
            continue;
        }

        stack.clear();
        if new_code >= next_code {
            // STRING+CHAR+STRING+CHAR+STRING: the code being defined now.
            decode(old_code, &prefix, &append, &mut stack);
            stack.insert(0, character);
        } else {
            decode(new_code, &prefix, &append, &mut stack);
        }
        character = *stack.last().unwrap();
        for &byte in stack.iter().rev() {
            if out.len() >= limit {
                return out;
            }
            out.push(byte);
        }

        if next_code <= MAX_CODE {
            prefix[next_code as usize] = old_code;
            append[next_code as usize] = character;
            next_code += 1;
        }
        old_code = new_code;
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A straightforward LZW encoder with the same code space, for tests.
    pub(crate) fn compress(data: &[u8]) -> Vec<u8> {
        use std::collections::HashMap;
        let mut codes: Vec<u32> = Vec::new();
        let mut table: HashMap<(u32, u8), u32> = HashMap::new();
        let mut next = 256u32;
        let mut iter = data.iter();
        if let Some(&first) = iter.next() {
            let mut current = u32::from(first);
            for &byte in iter {
                if let Some(&code) = table.get(&(current, byte)) {
                    current = code;
                } else {
                    codes.push(current);
                    if next <= MAX_CODE {
                        table.insert((current, byte), next);
                        next += 1;
                    }
                    current = u32::from(byte);
                }
            }
            codes.push(current);
        }
        codes.push(MAX_VALUE);

        let mut out = Vec::new();
        let (mut buffer, mut count) = (0u64, 0u32);
        for code in codes {
            buffer = (buffer << LZW_BITS) | u64::from(code);
            count += LZW_BITS;
            while count >= 8 {
                count -= 8;
                out.push((buffer >> count) as u8);
            }
        }
        if count > 0 {
            out.push((buffer << (8 - count)) as u8);
        }
        out
    }

    #[test]
    fn round_trips_repetitive_and_varied_data() {
        let mut data = b"TOBEORNOTTOBEORTOBEORNOT#".repeat(40);
        data.extend((0..5000u32).map(|i| (i * 7 % 251) as u8));
        assert_eq!(expand(&compress(&data), None), data);
    }

    #[test]
    fn handles_the_code_defined_by_its_own_use() {
        // "aaaa..." exercises new_code == next_code on every step.
        let data = vec![b'a'; 1000];
        assert_eq!(expand(&compress(&data), None), data);
    }

    #[test]
    fn stops_at_the_size_limit() {
        let data = b"abcabcabcabc".to_vec();
        assert_eq!(expand(&compress(&data), Some(5)), b"abcab");
    }

    #[test]
    fn empty_stream_is_empty() {
        assert!(expand(&compress(&[]), None).is_empty());
    }
}
