//! Port of `org.apache.lucene.util.OfflineSorter`: sorts a file of byte
//! sequences larger than memory by sorting RAM-sized partitions into temp
//! files and merging them `maxTempFiles` at a time.
//!
//! Lives in `lucene-codecs` rather than `lucene-util` because it needs a
//! [`Directory`] (Java's `util` package reaches into `store`; this port's
//! crate graph only allows that from `codecs` down).
//!
//! The file format is Java's `ByteSequencesWriter`: per item a little-endian
//! `short` length then the bytes, closed by a codec footer. What decides the
//! partitions is reproduced exactly -- `BytesRefArray`'s RAM accounting
//! (32 KiB pool blocks, the 64-byte offsets-array header, the offsets array's
//! `ArrayUtil.grow` steps) against the buffer size, or `ramBuffer /
//! valueLength` items for fixed-length values -- and so are the merge
//! schedule (per-level counters, merging the last `maxTempFiles` partitions)
//! and the temp-file names (`<prefix>_sort_<n in base 36>.tmp`, `n` counting
//! from 0 per sorter as Java's per-directory counter does from a fresh
//! directory). The result is therefore Lucene's output file under Lucene's
//! name, with Lucene's `SortInfo` counts.
//!
//! Partitions are sorted in the calling thread (Java's default
//! `SameThreadExecutorService`). A custom comparator under which distinct
//! byte strings compare equal may order those differently from Java's
//! (unstable) radix/intro sort.

use std::cmp::Ordering;

use lucene_store::codec_util::{self, FOOTER_LENGTH, FOOTER_MAGIC};
use lucene_store::data_input::SliceInput;
use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;
use lucene_store::index_output::{FsIndexOutput, IndexOutput};

/// `OfflineSorter.MB`.
pub const MB: u64 = 1024 * 1024;
/// `OfflineSorter.ABSOLUTE_MIN_SORT_BUFFER_SIZE`.
pub const ABSOLUTE_MIN_SORT_BUFFER_SIZE: u64 = MB / 2;
/// `OfflineSorter.MAX_TEMPFILES`.
pub const MAX_TEMPFILES: usize = 10;

const BYTE_BLOCK_SIZE: u64 = 32 * 1024;
/// `RamUsageEstimator.NUM_BYTES_ARRAY_HEADER * Integer.BYTES` on a 64-bit
/// JVM with compressed oops (the header is 16 bytes).
const OFFSETS_HEADER_BYTES: u64 = 16 * 4;

/// Errors of [`OfflineSorter`].
#[derive(Debug, thiserror::Error)]
pub enum SortError {
    /// A configuration Java rejects with `IllegalArgumentException`.
    #[error("{0}")]
    IllegalArgument(String),
    /// Reading, writing or checksumming a file failed.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// An input item does not fit the format.
    #[error("corrupt byte sequences file {file}: {msg}")]
    Corrupt { file: String, msg: String },
}

type Result<T> = std::result::Result<T, SortError>;

/// `OfflineSorter.SortInfo` (the counts; the timings are not kept).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SortInfo {
    /// `tempMergeFiles`.
    pub temp_merge_files: usize,
    /// `mergeRounds`.
    pub merge_rounds: usize,
    /// `lineCount`.
    pub line_count: u64,
    /// `bufferSize`.
    pub buffer_size: u64,
    /// Every temp file written, in order, with its item count -- what Java
    /// hands `getWriter(out, itemCount)`, i.e. the exact partition sizes.
    pub writes: Vec<(String, u64)>,
}

/// A comparator over byte sequences.
pub type BytesComparator = Box<dyn Fn(&[u8], &[u8]) -> Ordering + Send + Sync>;

/// `org.apache.lucene.util.OfflineSorter`.
pub struct OfflineSorter<'d> {
    dir: &'d dyn Directory,
    temp_prefix: String,
    comparator: BytesComparator,
    ram_buffer_bytes: u64,
    max_temp_files: usize,
    value_length: Option<usize>,
    next_temp: u64,
    info: SortInfo,
}

/// `ByteSequencesWriter` over any output: `short` length (LE) + bytes.
pub fn write_byte_sequence(out: &mut impl DataOutput, bytes: &[u8]) -> Result<()> {
    if bytes.len() > i16::MAX as usize {
        return Err(SortError::IllegalArgument(format!(
            "len must be <= {}; got {}",
            i16::MAX,
            bytes.len()
        )));
    }
    out.write_bytes(&(bytes.len() as u16).to_le_bytes());
    out.write_bytes(bytes);
    Ok(())
}

/// `CodecUtil.writeFooter` on a streaming output.
fn write_footer(out: &mut FsIndexOutput) {
    out.write_bytes(&FOOTER_MAGIC.to_be_bytes());
    out.write_bytes(&0u32.to_be_bytes());
    let checksum = out.checksum();
    out.write_bytes(&checksum.to_be_bytes());
}

/// `ByteSequencesReader` over a whole file whose footer has been checked.
struct SeqReader<'a> {
    name: String,
    input: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> SeqReader<'a> {
    fn open(name: &str, bytes: &'a [u8]) -> Result<Self> {
        let end = bytes
            .len()
            .checked_sub(FOOTER_LENGTH)
            .ok_or_else(|| SortError::Corrupt {
                file: name.to_string(),
                msg: "shorter than a codec footer".into(),
            })?;
        let mut footer = SliceInput::new(bytes);
        footer.seek(end)?;
        codec_util::check_footer(&mut footer, bytes.len())?;
        Ok(SeqReader {
            name: name.to_string(),
            input: bytes,
            pos: 0,
            end,
        })
    }

    // ARITH: `pos` advances by lengths checked against `end` first.
    #[allow(clippy::arithmetic_side_effects)]
    fn next(&mut self) -> Result<Option<&'a [u8]>> {
        if self.pos >= self.end {
            return Ok(None);
        }
        let corrupt = |msg: &str| SortError::Corrupt {
            file: self.name.clone(),
            msg: msg.to_string(),
        };
        if self.end - self.pos < 2 {
            return Err(corrupt("truncated length"));
        }
        let len = i16::from_le_bytes([self.input[self.pos], self.input[self.pos + 1]]);
        let len = usize::try_from(len).map_err(|_| corrupt("negative length"))?;
        let start = self.pos + 2;
        if self.end - start < len {
            return Err(corrupt("item runs past the footer"));
        }
        self.pos = start + len;
        Ok(Some(&self.input[start..start + len]))
    }
}

/// One in-RAM partition: the items back to back, with their offsets.
struct Partition {
    data: Vec<u8>,
    offsets: Vec<usize>,
}

impl Partition {
    // ARITH: offsets into `data`, ascending.
    #[allow(clippy::arithmetic_side_effects)]
    fn get(&self, i: usize) -> &[u8] {
        let end = self.offsets.get(i + 1).copied().unwrap_or(self.data.len());
        &self.data[self.offsets[i]..end]
    }
}

/// `ArrayUtil.oversize(minTargetSize, Integer.BYTES)` on a 64-bit JVM.
// ARITH: an offsets-array length far below `i32::MAX`.
#[allow(clippy::arithmetic_side_effects)]
fn oversize_ints(min: usize) -> usize {
    if min == 0 {
        return 0;
    }
    ((min + (min >> 3).max(3)) + 1) & 0x7fff_fffe
}

impl<'d> OfflineSorter<'d> {
    /// `new OfflineSorter(dir, prefix, comparator, BufferSize, maxTempFiles,
    /// valueLength, null, 0)`. `ram_buffer_bytes` is `BufferSize`'s byte
    /// count (at least 0.5 MB, at most `Integer.MAX_VALUE`); `value_length`
    /// is `None` for Java's `-1`.
    pub fn new(
        dir: &'d dyn Directory,
        temp_prefix: &str,
        comparator: Option<BytesComparator>,
        ram_buffer_bytes: u64,
        max_temp_files: usize,
        value_length: Option<usize>,
    ) -> Result<Self> {
        if ram_buffer_bytes > i32::MAX as u64 {
            return Err(SortError::IllegalArgument(format!(
                "Buffer too large for Java ({}MB max): {ram_buffer_bytes}",
                i32::MAX as u64 / MB
            )));
        }
        if ram_buffer_bytes < ABSOLUTE_MIN_SORT_BUFFER_SIZE {
            return Err(SortError::IllegalArgument(format!(
                "At least 0.5MB RAM buffer is needed: {ram_buffer_bytes}"
            )));
        }
        if max_temp_files < 2 {
            return Err(SortError::IllegalArgument(
                "maxTempFiles must be >= 2".into(),
            ));
        }
        if let Some(v) = value_length {
            if v == 0 || v > i16::MAX as usize {
                return Err(SortError::IllegalArgument(format!(
                    "valueLength must be 1 .. {}; got: {v}",
                    i16::MAX
                )));
            }
        }
        Ok(OfflineSorter {
            dir,
            temp_prefix: temp_prefix.to_string(),
            comparator: comparator.unwrap_or_else(|| Box::new(|a: &[u8], b: &[u8]| a.cmp(b))),
            ram_buffer_bytes,
            max_temp_files,
            value_length,
            next_temp: 0,
            info: SortInfo::default(),
        })
    }

    /// `sortInfo` of the last [`Self::sort`].
    pub fn sort_info(&self) -> &SortInfo {
        &self.info
    }

    /// `Directory.createTempOutput(prefix, "sort", ctx)`: the next unused
    /// `<prefix>_sort_<n base 36>.tmp`.
    // ARITH: a temp-file counter.
    #[allow(clippy::arithmetic_side_effects)]
    fn create_temp_output(&mut self) -> Result<FsIndexOutput> {
        let existing = self.dir.list_all()?;
        loop {
            let n = self.next_temp;
            self.next_temp += 1;
            let name = format!("{}_sort_{}.tmp", self.temp_prefix, base36(n));
            if !existing.contains(&name) {
                return Ok(self.dir.create_output(&name)?);
            }
        }
    }

    /// `OfflineSorter.sort(inputFileName)`: sorts the input into a new temp
    /// file and returns its name. On error every temp file created is
    /// deleted.
    pub fn sort(&mut self, input_name: &str) -> Result<String> {
        self.info = SortInfo {
            buffer_size: self.ram_buffer_bytes,
            ..SortInfo::default()
        };
        let mut created: Vec<String> = Vec::new();
        let result = self.sort_inner(input_name, &mut created);
        if result.is_err() {
            for name in created {
                let _ = self.dir.delete_file(&name);
            }
        }
        result
    }

    // ARITH: counts of partitions and levels, bounded by the input.
    #[allow(clippy::arithmetic_side_effects)]
    fn sort_inner(&mut self, input_name: &str, created: &mut Vec<String>) -> Result<String> {
        let input = self.dir.open(input_name)?;
        let mut reader = SeqReader::open(input_name, &input)?;
        // Partitions on disk: (file name, item count).
        let mut segments: Vec<(String, u64)> = Vec::new();
        let mut level_counts: Vec<usize> = vec![0];
        loop {
            let (part, exhausted) = self.read_partition(&mut reader)?;
            if part.offsets.is_empty() {
                break;
            }
            let count = part.offsets.len() as u64;
            let name = self.sort_partition(part)?;
            created.push(name.clone());
            segments.push((name, count));
            self.info.temp_merge_files += 1;
            self.info.line_count += count;
            level_counts[0] += 1;
            let mut merge_level = 0;
            while level_counts[merge_level] == self.max_temp_files {
                self.merge_partitions(&mut segments, created)?;
                if merge_level + 2 > level_counts.len() {
                    level_counts.resize(merge_level + 2, 0);
                }
                level_counts[merge_level + 1] += 1;
                level_counts[merge_level] = 0;
                merge_level += 1;
            }
            if exhausted {
                break;
            }
        }
        while segments.len() > 1 {
            self.merge_partitions(&mut segments, created)?;
        }
        let result = match segments.pop() {
            Some((name, _)) => name,
            None => {
                let mut out = self.create_temp_output()?;
                let name = out.name().to_string();
                created.push(name.clone());
                write_footer(&mut out);
                out.close()?;
                name
            }
        };
        Ok(result)
    }

    /// `readPartition`: items until the RAM accounting passes the buffer
    /// (variable length) or `ramBuffer / valueLength` items (fixed).
    // ARITH: `BytesRefArray`'s byte accounting, bounded by the buffer size.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_partition(&self, reader: &mut SeqReader<'_>) -> Result<(Partition, bool)> {
        let mut part = Partition {
            data: Vec::new(),
            offsets: Vec::new(),
        };
        if let Some(value_length) = self.value_length {
            let limit = self.ram_buffer_bytes as usize / value_length;
            for _ in 0..limit {
                match reader.next()? {
                    None => return Ok((part, true)),
                    Some(item) => {
                        if item.len() != value_length {
                            return Err(SortError::Corrupt {
                                file: reader.name.clone(),
                                msg: format!(
                                    "item of {} bytes, valueLength {value_length}",
                                    item.len()
                                ),
                            });
                        }
                        part.offsets.push(part.data.len());
                        part.data.extend_from_slice(item);
                    }
                }
            }
            return Ok((part, false));
        }
        // `new BytesRefArray(counter)`: one pool block plus the offsets
        // array's header, then `int[1]`.
        let mut offsets_len = 1usize;
        loop {
            let Some(item) = reader.next()? else {
                return Ok((part, true));
            };
            if part.offsets.len() >= offsets_len {
                offsets_len = oversize_ints(offsets_len + 1);
            }
            part.offsets.push(part.data.len());
            part.data.extend_from_slice(item);
            // `ByteBlockPool.append` takes a new block whenever the current
            // one fills exactly, so the pool holds `total / 32K + 1` blocks.
            let blocks = part.data.len() as u64 / BYTE_BLOCK_SIZE + 1;
            // The initial `int[1]` is not counted; only growth is.
            let bytes_used =
                blocks * BYTE_BLOCK_SIZE + OFFSETS_HEADER_BYTES + 4 * (offsets_len as u64 - 1);
            if bytes_used > self.ram_buffer_bytes {
                return Ok((part, false));
            }
        }
    }

    /// `SortPartitionTask`: sorts a partition into a temp file.
    fn sort_partition(&mut self, part: Partition) -> Result<String> {
        let mut order: Vec<usize> = (0..part.offsets.len()).collect();
        let cmp = &self.comparator;
        order.sort_unstable_by(|&a, &b| cmp(part.get(a), part.get(b)));
        let mut out = self.create_temp_output()?;
        let name = out.name().to_string();
        self.info
            .writes
            .push((name.clone(), part.offsets.len() as u64));
        for i in order {
            write_byte_sequence(&mut out, part.get(i))?;
        }
        write_footer(&mut out);
        out.close()?;
        Ok(name)
    }

    /// `mergePartitions`: merges the last `maxTempFiles` partitions (or all
    /// of them, if fewer) into a new one appended at the end.
    // ARITH: indices into `segments`.
    #[allow(clippy::arithmetic_side_effects)]
    fn merge_partitions(
        &mut self,
        segments: &mut Vec<(String, u64)>,
        created: &mut Vec<String>,
    ) -> Result<()> {
        let from = segments.len().saturating_sub(self.max_temp_files);
        let to_merge: Vec<(String, u64)> = segments.drain(from..).collect();
        self.info.merge_rounds += 1;
        let total: u64 = to_merge.iter().map(|s| s.1).sum();
        let mut out = self.create_temp_output()?;
        let new_name = out.name().to_string();
        created.push(new_name.clone());
        self.info.writes.push((new_name.clone(), total));
        let inputs = to_merge
            .iter()
            .map(|(name, _)| self.dir.open(name))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut readers = to_merge
            .iter()
            .zip(&inputs)
            .map(|((name, _), bytes)| SeqReader::open(name, bytes))
            .collect::<Result<Vec<_>>>()?;
        // `PriorityQueue<FileAndTop>`: a binary min-heap by `comparator`,
        // Java's `insertWithOverflow`/`updateTop`/`pop` shape.
        let mut heap: Vec<(usize, &[u8])> = Vec::with_capacity(readers.len());
        for (fd, r) in readers.iter_mut().enumerate() {
            let first = r.next()?.ok_or_else(|| SortError::Corrupt {
                file: r.name.clone(),
                msg: "empty partition".into(),
            })?;
            heap.push((fd, first));
            let mut i = heap.len() - 1;
            while i > 0 {
                let parent = (i - 1) / 2;
                if (self.comparator)(heap[i].1, heap[parent].1) == Ordering::Less {
                    heap.swap(i, parent);
                    i = parent;
                } else {
                    break;
                }
            }
        }
        while let Some(&(fd, top)) = heap.first() {
            write_byte_sequence(&mut out, top)?;
            match readers[fd].next()? {
                Some(next) => heap[0] = (fd, next),
                None => {
                    let last = heap.len() - 1;
                    heap.swap(0, last);
                    heap.pop();
                }
            }
            // Sift down from the root.
            let mut i = 0;
            loop {
                let (l, r) = (2 * i + 1, 2 * i + 2);
                let mut m = i;
                if l < heap.len() && (self.comparator)(heap[l].1, heap[m].1) == Ordering::Less {
                    m = l;
                }
                if r < heap.len() && (self.comparator)(heap[r].1, heap[m].1) == Ordering::Less {
                    m = r;
                }
                if m == i {
                    break;
                }
                heap.swap(i, m);
                i = m;
            }
        }
        write_footer(&mut out);
        out.close()?;
        drop(readers);
        drop(inputs);
        for (name, _) in &to_merge {
            self.dir.delete_file(name)?;
            created.retain(|c| c != name);
        }
        self.info.temp_merge_files += 1;
        segments.push((new_name, total));
        Ok(())
    }
}

/// `Long.toString(n, Character.MAX_RADIX)`.
// ARITH: digit extraction.
#[allow(clippy::arithmetic_side_effects)]
fn base36(mut n: u64) -> String {
    if n == 0 {
        return "0".into();
    }
    let mut digits = Vec::new();
    while n > 0 {
        let d = (n % 36) as u8;
        digits.push(if d < 10 { b'0' + d } else { b'a' + d - 10 });
        n /= 36;
    }
    digits.reverse();
    String::from_utf8(digits).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::directory::FsDirectory;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "offline_sorter_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_input(dir: &FsDirectory, name: &str, items: &[Vec<u8>]) {
        let mut out = dir.create_output(name).unwrap();
        for i in items {
            write_byte_sequence(&mut out, i).unwrap();
        }
        write_footer(&mut out);
        out.close().unwrap();
    }

    fn read_all(dir: &FsDirectory, name: &str) -> Vec<Vec<u8>> {
        let bytes = dir.open(name).unwrap();
        let mut r = SeqReader::open(name, &bytes).unwrap();
        let mut v = Vec::new();
        while let Some(i) = r.next().unwrap() {
            v.push(i.to_vec());
        }
        v
    }

    #[test]
    fn sorts_across_partitions_and_merge_levels() {
        let path = temp_dir("multi");
        let dir = FsDirectory::open(&path);
        let items: Vec<Vec<u8>> = (0..60_000u64)
            .map(|i| format!("{:x}", (i * 2_654_435_761) % 1_000_003).into_bytes())
            .collect();
        write_input(&dir, "in", &items);
        let mut sorter =
            OfflineSorter::new(&dir, "t", None, ABSOLUTE_MIN_SORT_BUFFER_SIZE, 2, None).unwrap();
        let out = sorter.sort("in").unwrap();
        let mut want = items.clone();
        want.sort();
        assert_eq!(read_all(&dir, &out), want);
        let info = sorter.sort_info().clone();
        assert_eq!(info.line_count, 60_000);
        assert!(
            info.merge_rounds >= 1 && info.temp_merge_files > 2,
            "{info:?}"
        );
        // Only the input and the result remain.
        let mut left = dir.list_all().unwrap();
        left.sort();
        assert_eq!(left, vec!["in".to_string(), out.clone()]);
        std::fs::remove_dir_all(&path).unwrap();
    }

    #[test]
    fn fixed_length_empty_and_reverse() {
        let path = temp_dir("fixed");
        let dir = FsDirectory::open(&path);
        let items: Vec<Vec<u8>> = (0..1000u32)
            .map(|i| (i * 7919 % 1000).to_be_bytes().to_vec())
            .collect();
        write_input(&dir, "in", &items);
        let rev: BytesComparator = Box::new(|a: &[u8], b: &[u8]| b.cmp(a));
        let mut sorter = OfflineSorter::new(&dir, "f", Some(rev), MB, 10, Some(4)).unwrap();
        let out = sorter.sort("in").unwrap();
        let mut want = items.clone();
        want.sort_by(|a, b| b.cmp(a));
        assert_eq!(read_all(&dir, &out), want);
        write_input(&dir, "empty", &[]);
        let mut s2 = OfflineSorter::new(&dir, "e", None, MB, 10, None).unwrap();
        let out = s2.sort("empty").unwrap();
        assert_eq!(out, "e_sort_0.tmp");
        assert!(read_all(&dir, &out).is_empty());
        write_input(&dir, "bad", &[vec![1, 2, 3]]);
        let mut s3 = OfflineSorter::new(&dir, "b", None, MB, 10, Some(4)).unwrap();
        assert!(s3.sort("bad").is_err());
        assert!(!dir
            .list_all()
            .unwrap()
            .iter()
            .any(|n| n.starts_with("b_sort")));
        std::fs::remove_dir_all(&path).unwrap();
    }

    #[test]
    fn rejects_bad_configuration_and_corrupt_input() {
        let path = temp_dir("bad");
        let dir = FsDirectory::open(&path);
        assert!(OfflineSorter::new(&dir, "x", None, 1000, 10, None).is_err());
        assert!(OfflineSorter::new(&dir, "x", None, 1 << 40, 10, None).is_err());
        assert!(OfflineSorter::new(&dir, "x", None, MB, 1, None).is_err());
        assert!(OfflineSorter::new(&dir, "x", None, MB, 2, Some(0)).is_err());
        assert!(OfflineSorter::new(&dir, "x", None, MB, 2, Some(40_000)).is_err());
        let mut out = dir.create_output("corrupt").unwrap();
        out.write_bytes(&[1, 2, 3]);
        out.close().unwrap();
        let mut s = OfflineSorter::new(&dir, "c", None, MB, 2, None).unwrap();
        assert!(s.sort("corrupt").is_err());
        let mut v = Vec::new();
        assert!(write_byte_sequence(&mut v, &vec![0u8; 40_000]).is_err());
        v.extend_from_slice(&[5, 0, 1]);
        codec_util::write_footer(&mut v);
        let mut o = dir.create_output("trunc").unwrap();
        o.write_bytes(&v);
        o.close().unwrap();
        assert!(s.sort("trunc").is_err());
        assert_eq!(base36(0), "0");
        assert_eq!(base36(36 * 36 + 35), "10z");
        assert_eq!(oversize_ints(0), 0);
        std::fs::remove_dir_all(&path).unwrap();
    }
}
