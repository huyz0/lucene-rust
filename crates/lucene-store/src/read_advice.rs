//! How a search reader advises the kernel to read each mapped index file,
//! by file extension: Lucene's `ReadAdvice` per file, set process-wide.
//!
//! A mapped file's page faults read ahead of the faulting page by default
//! (Linux reads up to 128 KiB around it). That suits files read in order,
//! and wastes reads and memory on files read a few hundred bytes at a time
//! at scattered offsets once they no longer fit in memory: on network block
//! storage with an IOPS and bandwidth cap (EBS gp3: 3000 IOPS, 125 MB/s)
//! the waste is what caps throughput. Lucene 9.11 to 10.2 advised the
//! vector files, stored fields and term vectors `RANDOM`; 10.3 went back to
//! the OS default and made per-file advice opt-in
//! (`MMapDirectory.ADVISE_BY_CONTEXT`).
//!
//! The defaults here advise random reads for the files a vector search reads
//! a vector at a time: flat vectors (`vec`), the HNSW graph (`vex`), IVF
//! rerank vectors (`ivv`) and quantized vector codes (`veq`, `vqd`). Every
//! other file keeps the OS default. [`set`] replaces the table; `*` sets the
//! advice of every extension not listed (Lucene's
//! `org.apache.lucene.store.defaultReadAdvice`).
//!
//! A file advised random that is small next to available memory is also
//! read in once at open ([`set_warm_random`], on by default), so a cold
//! start still warms it in large reads. The advice only changes how the
//! kernel reads: no reader sees different bytes.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

use crate::error::{Error, Result};

/// How the kernel should read a mapped file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadAdvice {
    /// The OS default: fault-time read-ahead around each page.
    Normal,
    /// One page per fault, no read-ahead (`MADV_RANDOM`).
    Random,
    /// Aggressive read-ahead, pages dropped soon after use
    /// (`MADV_SEQUENTIAL`).
    Sequential,
    /// The whole file is read in at open (`MADV_WILLNEED`), then the OS
    /// default: OpenSearch's `index.store.preload`.
    WillNeed,
}

impl ReadAdvice {
    /// The advice named `name` (Lucene's `ReadAdvice` names, any case,
    /// plus `willneed`/`preload`).
    pub fn parse(name: &str) -> Option<ReadAdvice> {
        match name.trim().to_ascii_lowercase().as_str() {
            "normal" => Some(ReadAdvice::Normal),
            "random" => Some(ReadAdvice::Random),
            "sequential" => Some(ReadAdvice::Sequential),
            "willneed" | "will_need" | "preload" => Some(ReadAdvice::WillNeed),
            _ => None,
        }
    }
}

/// The extensions advised [`ReadAdvice::Random`] unless [`set`] says
/// otherwise.
pub const DEFAULT_RANDOM_EXTENSIONS: [&str; 5] = ["vec", "vex", "ivv", "veq", "vqd"];

struct Table {
    /// `(extension, advice)`, as [`set`] gave them.
    rules: Vec<(String, ReadAdvice)>,
    /// The advice of an extension not in `rules`.
    other: ReadAdvice,
}

static TABLE: RwLock<Option<Table>> = RwLock::new(None);
static WARM_RANDOM: AtomicBool = AtomicBool::new(true);

/// The advice for a file with extension `ext` (the part after the last
/// `.`; a codec suffix before it is ignored, so `_0_Lucene99_0.vec` and
/// `vec` both work).
pub fn for_extension(ext: &str) -> ReadAdvice {
    let ext = ext.rsplit('.').next().unwrap_or(ext);
    let table = TABLE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match &*table {
        Some(t) => t
            .rules
            .iter()
            .find(|(e, _)| e == ext)
            .map_or(t.other, |&(_, a)| a),
        None if DEFAULT_RANDOM_EXTENSIONS.contains(&ext) => ReadAdvice::Random,
        None => ReadAdvice::Normal,
    }
}

/// Replaces the table with `spec`: comma-separated `ext:advice` (or
/// `ext=advice`) entries, `*` for every other extension. An empty spec
/// restores the defaults. Extensions not listed and no `*` get
/// [`ReadAdvice::Normal`], as Lucene's directory does: a spec states the
/// whole table.
///
/// # Errors
/// An entry without an advice or with an unknown one; the table is then
/// unchanged.
pub fn set(spec: &str) -> Result<()> {
    let entries: Vec<&str> = spec
        .split(',')
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .collect();
    if entries.is_empty() {
        *TABLE
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        return Ok(());
    }
    let mut table = Table {
        rules: Vec::with_capacity(entries.len()),
        other: ReadAdvice::Normal,
    };
    for entry in entries {
        let (ext, advice) = entry
            .split_once([':', '='])
            .ok_or_else(|| Error::Io(bad(format!("read advice entry '{entry}' has no advice"))))?;
        let advice = ReadAdvice::parse(advice).ok_or_else(|| {
            Error::Io(bad(format!(
                "unknown read advice '{advice}' (normal, random, sequential or willneed)"
            )))
        })?;
        let ext = ext.trim().trim_start_matches('.');
        if ext == "*" {
            table.other = advice;
        } else {
            table.rules.push((ext.to_string(), advice));
        }
    }
    *TABLE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(table);
    Ok(())
}

fn bad(message: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
}

/// Whether a file advised [`ReadAdvice::Random`] that is small next to
/// available memory is read in once at open (on by default).
pub fn warm_random() -> bool {
    WARM_RANDOM.load(Ordering::Relaxed)
}

/// Sets [`warm_random`].
pub fn set_warm_random(on: bool) {
    WARM_RANDOM.store(on, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test owns the process-wide table, so parallel tests never see
    /// another's.
    #[test]
    fn the_table_defaults_parses_and_restores() {
        assert_eq!(for_extension("vec"), ReadAdvice::Random);
        assert_eq!(
            for_extension("_0_Lucene99HnswVectorsFormat_0.veq"),
            ReadAdvice::Random
        );
        assert_eq!(for_extension("fdt"), ReadAdvice::Normal);

        set("fdt:random, .tvd=Random ,doc:sequential,*:normal,kdd:preload").unwrap();
        assert_eq!(for_extension("fdt"), ReadAdvice::Random);
        assert_eq!(for_extension("tvd"), ReadAdvice::Random);
        assert_eq!(for_extension("doc"), ReadAdvice::Sequential);
        assert_eq!(for_extension("kdd"), ReadAdvice::WillNeed);
        // A spec states the whole table.
        assert_eq!(for_extension("vec"), ReadAdvice::Normal);

        set("*:random").unwrap();
        assert_eq!(for_extension("tim"), ReadAdvice::Random);

        assert!(set("vec").is_err());
        assert!(set("vec:sideways").is_err());
        // A rejected spec leaves the table as it was.
        assert_eq!(for_extension("tim"), ReadAdvice::Random);

        set(" ").unwrap();
        assert_eq!(for_extension("vec"), ReadAdvice::Random);
        assert_eq!(for_extension("tim"), ReadAdvice::Normal);

        assert!(warm_random());
        set_warm_random(false);
        assert!(!warm_random());
        set_warm_random(true);
    }
}
