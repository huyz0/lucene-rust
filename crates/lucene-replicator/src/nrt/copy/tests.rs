#![cfg(test)]
use super::*;
use lucene_store::ByteBuffersDirectory;

struct Bytes {
    data: Vec<u8>,
    pos: usize,
}

impl CopyStream for Bytes {
    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        let end = self.pos + buf.len();
        if end > self.data.len() {
            return Err(io_error("stream ended".to_string()));
        }
        buf.copy_from_slice(&self.data[self.pos..end]);
        self.pos = end;
        Ok(())
    }
}

struct Job(CopyJobCore);

impl CopyJob for Job {
    fn core(&self) -> &CopyJobCore {
        &self.0
    }
    fn start(&self) -> Result<()> {
        Ok(())
    }
    fn run_blocking(&self) -> Result<()> {
        Ok(())
    }
    fn visit(&self) -> Result<bool> {
        Ok(true)
    }
    fn finish(&self) -> Result<()> {
        Ok(())
    }
    fn copy_state(&self) -> Option<&CopyState> {
        None
    }
}

fn replica() -> (Arc<ReplicaCore>, Arc<Mutex<Vec<String>>>) {
    let dir = Arc::new(ByteBuffersDirectory::new());
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = Arc::clone(&log);
    let node = Arc::new(NodeCore::new(
        1,
        false,
        dir.clone(),
        Some(Arc::new(move |s: &str| lock(&sink).push(s.to_string()))),
    ));
    node.set_verbose_files(true);
    let deleter = ReplicaFileDeleter::new(Arc::clone(&node), dir);
    (Arc::new(ReplicaCore::new(node, deleter)), log)
}

/// A codec file, its metadata, and the stream a primary sends for it.
fn file(payload: &[u8]) -> (FileMetaData, Vec<u8>) {
    let mut buf = Vec::new();
    lucene_store::codec_util::write_index_header(&mut buf, "Codec", 1, &[7u8; 16], "");
    let header_len = buf.len();
    buf.extend_from_slice(payload);
    lucene_store::codec_util::write_footer(&mut buf);
    let checksum = lucene_store::codec_util::retrieve_checksum(&buf).unwrap();
    let meta = FileMetaData {
        header: buf[..header_len].to_vec(),
        footer: buf[buf.len() - 16..].to_vec(),
        length: buf.len() as u64,
        checksum,
    };
    let mut stream = buf[..buf.len() - 8].to_vec();
    stream.extend_from_slice(&checksum.to_be_bytes());
    (meta, stream)
}

#[test]
fn a_file_is_copied_and_verified() {
    let (dest, _log) = replica();
    let (meta, stream) = file(&[5u8; 200]);
    let mut copy = CopyOneFile::new(Arc::clone(&dest), "_0.dat", meta.clone(), 16).unwrap();
    assert_eq!(copy.bytes_to_copy, meta.length - 8);
    let mut input = Bytes {
        data: stream,
        pos: 0,
    };
    let mut visits = 0;
    while !copy.visit(&mut input).unwrap() {
        visits += 1;
        assert!(copy.bytes_copied() > 0);
    }
    assert!(visits > 0, "ten buffers of 16 bytes do not cover the file");
    assert_eq!(copy.bytes_copied(), meta.length);
    // The temporary file is whole and a second visit finds it closed.
    assert_eq!(dest.dir().file_length(&copy.tmp_name).unwrap(), meta.length);
    assert!(matches!(
        copy.visit(&mut input),
        Err(Error::AlreadyClosed(_))
    ));
    copy.close().unwrap();
    dest.start_copy_file("_0.dat").unwrap();
    dest.finish_copy_file("_0.dat").unwrap();
}

#[test]
fn a_wrong_checksum_fails_the_copy() {
    let (dest, log) = replica();
    let (mut meta, stream) = file(&[1u8; 40]);
    meta.checksum ^= 1;
    let mut copy = CopyOneFile::new(Arc::clone(&dest), "a", meta, 64).unwrap();
    let err = copy
        .visit(&mut Bytes {
            data: stream,
            pos: 0,
        })
        .unwrap_err();
    assert!(err.to_string().contains("checksum mismatch"));
    assert!(lock(&log).iter().any(|l| l.contains("bits flipped")));

    // The primary claims a different checksum than the bytes carry.
    let (meta, mut stream) = file(&[1u8; 40]);
    let n = stream.len();
    stream[n - 1] ^= 1;
    let mut copy = CopyOneFile::new(Arc::clone(&dest), "b", meta, 64).unwrap();
    let err = copy
        .visit(&mut Bytes {
            data: stream,
            pos: 0,
        })
        .unwrap_err();
    assert!(err.to_string().contains("checksum mismatch"));
    assert!(lock(&log).iter().any(|l| l.contains("claimed by primary")));
}

#[test]
fn copies_are_registered_once() {
    let (dest, _log) = replica();
    let (meta, _) = file(b"x");
    let first = CopyOneFile::new(Arc::clone(&dest), "a", meta.clone(), 8).unwrap();
    assert!(matches!(
        CopyOneFile::new(Arc::clone(&dest), "a", meta.clone(), 8),
        Err(Error::IllegalState(_))
    ));
    assert!(matches!(
        dest.finish_copy_file("never"),
        Err(Error::IllegalState(_))
    ));
    drop(first);
    let short = FileMetaData { length: 7, ..meta };
    assert!(matches!(
        CopyOneFile::new(dest, "short", short, 8),
        Err(Error::IllegalState(_))
    ));
}

#[test]
fn vlongs_and_big_endian_longs_are_read() {
    let mut input = Bytes {
        data: vec![0xac, 0x02, 0, 0, 0, 0, 0, 0, 0, 1],
        pos: 0,
    };
    assert_eq!(input.read_vlong().unwrap(), 300);
    assert_eq!(input.read_be_long().unwrap(), 1);
    assert!(input.read_vlong().is_err(), "ended mid-vlong");
    let mut endless = Bytes {
        data: vec![0xff; 11],
        pos: 0,
    };
    assert!(endless.read_vlong().is_err());
    endless.close();
    // Nine bytes carry 63 bits; a ninth continuation bit is rejected.
    let mut max = Bytes {
        data: [vec![0xff; 8], vec![0x7f]].concat(),
        pos: 0,
    };
    assert_eq!(max.read_vlong().unwrap(), i64::MAX as u64);
    let mut ten = Bytes {
        data: [vec![0xff; 9], vec![0x01]].concat(),
        pos: 0,
    };
    assert!(ten.read_vlong().is_err(), "a tenth byte");
}

#[test]
fn a_job_copies_only_what_the_replica_lacks() {
    let (dest, _log) = replica();
    let (meta, bytes) = file(&[3u8; 20]);
    let mut out = dest.dir().create_output("have").unwrap();
    out.write_bytes(&bytes);
    out.close().unwrap();
    let mut files = BTreeMap::new();
    files.insert("have".to_string(), meta.clone());
    files.insert("lack".to_string(), meta.clone());
    let mut other = meta.clone();
    other.footer[0] ^= 1;
    files.insert("differs".to_string(), other);
    let job = Job(
        CopyJobCore::new("test", Some(Arc::new(files)), Arc::clone(&dest), true, None).unwrap(),
    );
    let names: Vec<_> = job.file_names_to_copy().into_iter().collect();
    assert_eq!(names, ["differs", "lack"]);
    assert_eq!(job.file_names().len(), 3);
    assert!(job.description().contains("test"));
    assert!(!job.failed());
    assert_eq!(job.total_bytes_copied(), 0);

    let other_job = Job(CopyJobCore::new(
        "other",
        Some(Arc::new(BTreeMap::from([("lack".to_string(), meta)]))),
        dest,
        false,
        None,
    )
    .unwrap());
    assert!(job.conflicts(&other_job));
    assert_eq!(compare_jobs(&job.0, &other_job.0), std::cmp::Ordering::Less);
    assert_eq!(
        compare_jobs(&other_job.0, &job.0),
        std::cmp::Ordering::Greater
    );
    job.run_blocking().unwrap();
    assert!(job.visit().unwrap());
    job.start().unwrap();
    job.finish().unwrap();
    job.close();
}

#[test]
fn cancel_removes_the_partial_file_and_is_idempotent() {
    let (dest, log) = replica();
    let (meta, stream) = file(&[9u8; 100]);
    let job = Job(CopyJobCore::new(
        "c",
        Some(Arc::new(BTreeMap::from([("f".to_string(), meta.clone())]))),
        Arc::clone(&dest),
        false,
        None,
    )
    .unwrap());
    let mut current = CopyOneFile::new(Arc::clone(&dest), "f", meta, 8).unwrap();
    let tmp = current.tmp_name.clone();
    assert!(!current
        .visit(&mut Bytes {
            data: stream,
            pos: 0
        })
        .unwrap());
    job.0.state().current = Some(current);
    job.0
        .state()
        .copied_files
        .insert("g".to_string(), "g.tmp".to_string());
    job.cancel("because", Some("boom")).unwrap();
    assert!(job.failed());
    assert_eq!(job.0.state().cancel_reason.as_deref(), Some("because"));
    assert!(
        !dest.dir().list_all().unwrap().contains(&tmp),
        "the partial file is gone"
    );
    assert!(lock(&log).iter().any(|l| l.contains("remove partial file")));
    let lines = lock(&log).len();
    job.cancel("again", None).unwrap();
    assert_eq!(lock(&log).len(), lines, "a second cancel does nothing");
}

#[test]
fn a_job_takes_over_what_the_previous_one_copied() {
    let (dest, log) = replica();
    let (meta, stream) = file(&[4u8; 100]);
    let files: BTreeMap<_, _> = ["done", "partial", "fresh"]
        .iter()
        .map(|n| (n.to_string(), meta.clone()))
        .collect();
    let files = Arc::new(files);
    let prev = Job(CopyJobCore::new(
        "prev",
        Some(Arc::clone(&files)),
        Arc::clone(&dest),
        false,
        None,
    )
    .unwrap());
    let next = Job(CopyJobCore::new("next", Some(files), Arc::clone(&dest), true, None).unwrap());
    let mut partial = CopyOneFile::new(Arc::clone(&dest), "partial", meta, 8).unwrap();
    partial
        .visit(&mut Bytes {
            data: stream,
            pos: 0,
        })
        .unwrap();
    let copied = partial.bytes_copied();
    {
        let mut st = prev.0.state();
        st.current = Some(partial);
        st.copied_files
            .insert("done".to_string(), "done.tmp".to_string());
        st.copied_files
            .insert("stale".to_string(), "stale.tmp".to_string());
    }
    next.transfer_and_cancel(&prev).unwrap();
    assert!(prev.failed());
    {
        let st = next.0.state();
        assert_eq!(
            st.copied_files.get("done").map(String::as_str),
            Some("done.tmp")
        );
        assert_eq!(st.current.as_ref().unwrap().bytes_copied(), copied);
        assert_eq!(
            st.to_copy
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            ["fresh"]
        );
    }
    assert!(prev.0.state().copied_files.is_empty());
    assert!(lock(&log)
        .iter()
        .any(|l| l.contains("carry over in-progress file partial")));

    // A job that was already cancelled gives nothing.
    let third = Job(CopyJobCore::new("third", None, Arc::clone(&dest), false, None).unwrap());
    third.transfer_and_cancel(&prev).unwrap();
    assert!(lock(&log).iter().any(|l| l.contains("already cancelled")));
}
