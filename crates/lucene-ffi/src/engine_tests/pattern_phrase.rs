//! The `pattern_phrase` tests that run through the exported entry points.

#[cfg(test)]
pub(crate) mod tests {
    use crate::engine_tests::nodes::tests::{fails, open, search, Blob};
    use crate::jvm_reader::ffi_close_jvm_reader;

    use lucene_engine::analysis::test_support::spec;

    use lucene_engine::pattern_phrase::*;
    use lucene_engine::query_blob::Cursor;
    use lucene_engine::status::Status;

    use lucene_search::query::Clause;

    fn standard() -> Vec<u8> {
        spec(&[], ("standard", &[]), &[("lowercase", &[])], &[])
    }

    fn node(template_parts: u8, phrase: &[(i32, &[&str])]) -> Blob {
        let mut b = Blob::tree()
            .u8(NODE_PATTERN_PHRASE)
            .str("message")
            .bytes(&standard())
            .u8(template_parts)
            .i32(phrase.len() as i32);
        for (p, terms) in phrase {
            b = b.i32(*p).i32(terms.len() as i32);
            for t in *terms {
                b = b.str(t);
            }
        }
        b
    }

    fn decoded(blob: &Blob) -> std::result::Result<Clause, Status> {
        let mut c = Cursor::new(&blob.0[2..]);
        decode(&mut c)
    }

    #[test]
    fn matches_the_messages_that_hold_the_phrase() {
        use lucene_index::document::{Document, Store, StringField};
        use lucene_index::index_writer::IndexWriter;
        use lucene_store::FsDirectory;
        use lucene_util::test_support::TempDir;

        let messages = [
            "Failed password for root from 10.0.0.1 port 22 ssh2",
            "Failed password for invalid user admin1 from 10.0.0.2",
            "connection 3 closed by peer",
            "connection 4 closed by peer",
            "instruction cache parity error corrected",
            "password Failed",
        ];
        let tmp = TempDir::new("pattern_phrase_node");
        let dir = FsDirectory::open(tmp.path());
        let version = lucene_index::segment_info::LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        for (i, m) in messages.iter().enumerate() {
            let mut d = Document::new();
            d.add(StringField::new("id", i.to_string(), Store::Yes));
            lucene_pattern_text::fields(&mut d, "message", m).unwrap();
            w.add_fields_documents(&[d]).unwrap();
        }
        w.commit().unwrap();
        let h = open(&tmp);
        let docs = |blob: &Blob| {
            let (hits, total) = search(h, blob);
            assert!(hits.iter().all(|&(_, s)| s == 0.0), "phrases score 0");
            let mut d: Vec<i32> = hits.into_iter().map(|(d, _)| d).collect();
            d.sort_unstable();
            assert_eq!(total, d.len() as i64);
            d
        };
        for parts in [0, 1] {
            let failed: &[(i32, &[&str])] = &[(0, &["failed"]), (1, &["password"])];
            assert_eq!(docs(&node(parts, failed)), [0, 1]);
            let var: &[(i32, &[&str])] = &[(0, &["connection"]), (1, &["3"]), (2, &["closed"])];
            assert_eq!(docs(&node(parts, var)), [2]);
            // stacked terms at one position match either
            let either: &[(i32, &[&str])] = &[(0, &["3", "4"]), (1, &["closed"])];
            assert_eq!(docs(&node(parts, either)), [2, 3]);
            let none: &[(i32, &[&str])] = &[(0, &["parity"]), (1, &["corrected"])];
            assert!(docs(&node(parts, none)).is_empty());
            let one: &[(i32, &[&str])] = &[(5, &["parity"])];
            assert_eq!(docs(&node(parts, one)), [4]);
        }
        let (_, msg) = fails(h, &node(2, &[(0, &["a"])]));
        assert!(msg.contains("template flag"), "{msg}");
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn refuses_bad_nodes() {
        assert!(decoded(&node(1, &[(0, &["a"]), (1, &["b"])])).is_ok());
        for bad in [
            node(1, &[]),
            node(1, &[(0, &[])]),
            node(1, &[(1, &["a"]), (0, &["b"])]),
        ] {
            assert!(decoded(&bad).is_err());
        }
        let many: Vec<&str> = vec!["t"; MAX_TERMS + 1];
        assert!(decoded(&node(1, &[(0, &many)])).is_err());
    }
}
