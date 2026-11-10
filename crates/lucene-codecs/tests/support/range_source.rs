//! Test-only, counted file-backed range source. Uses real Java-written files;
//! it never loads a complete file into the source's heap.
#![allow(clippy::arithmetic_side_effects, dead_code)]
use lucene_store::range_input::{RangeInput, RangeLimits, ReadAt};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

pub struct Source {
    file: Mutex<File>,
    length: usize,
    pub fault: AtomicUsize,
    calls: Mutex<Vec<(usize, usize)>>,
}
impl ReadAt for Source {
    fn len(&self) -> usize {
        self.length
    }
    fn read_at(&self, at: usize, out: &mut [u8]) -> lucene_store::Result<usize> {
        let mut calls = self.calls.lock().unwrap();
        if calls.len() == 32768 {
            return Err(std::io::Error::other("trace capacity").into());
        }
        calls.push((at, out.len()));
        drop(calls);
        match self.fault.load(Ordering::Relaxed) {
            1 => return Err(std::io::Error::other("injected range error").into()),
            2 => return Ok(0),
            _ => {}
        }
        let mut file = self.file.lock().unwrap();
        file.seek(SeekFrom::Start(at as u64))?;
        Ok(file.read(out)?)
    }
}
pub fn open(path: impl AsRef<Path>) -> (Arc<Source>, RangeInput) {
    let file = File::open(path).unwrap();
    let length = usize::try_from(file.metadata().unwrap().len()).unwrap();
    let source = Arc::new(Source {
        file: Mutex::new(file),
        length,
        fault: AtomicUsize::new(0),
        calls: Mutex::new(Vec::new()),
    });
    let input = RangeInput::new(
        source.clone(),
        RangeLimits {
            block_bytes: 16,
            cache_blocks: 2,
            attempts: 32768,
            transferred_bytes: 1024 * 1024,
        },
    )
    .unwrap();
    (source, input)
}
pub fn fail(source: &Source, mode: usize) {
    source.fault.store(mode, Ordering::Relaxed);
}
pub fn calls(source: &Source) -> Vec<(usize, usize)> {
    source.calls.lock().unwrap().clone()
}
