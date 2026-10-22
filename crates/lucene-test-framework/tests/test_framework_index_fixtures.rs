//! Differential test against `fixtures/src/testframework/GenTestFrameworkIndex.java`:
//! for 200 seeds, `new_index_writer_config` draws from Java's `r` what
//! Lucene 10.5.0's `newIndexWriterConfig(r, a)` draws (scheduler, buffering,
//! merge policy and its settings, the flags, and where the stream ends), and
//! for 100 of them 40 rounds of `maybe_change_live_index_writer_config`
//! leave the configuration where Java's leaves it. The values Java draws
//! from the test's own `random()` are not compared (see the generator).
//!
//! Regenerate with `scripts/gen-test-framework-fixtures.sh --only TestFrameworkIndex`.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_index::merge_policy::log::LogSizeUnit;
use lucene_index::merge_policy::MergePolicy;
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::analysis::mock_analyzer::MockAnalyzer;
use lucene_test_framework::index::index_writer_config::{
    maybe_change_live_index_writer_config, new_index_writer_config, Draws, LiveFlushMode,
    MergeSchedulerChoice, PolicyKind, RandomIndexWriterConfig,
};
use lucene_test_framework::store::base_lock_factory::open_writer;
use lucene_util::java_random::JavaRandom;

fn fixture() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data/index.txt");
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn bits(d: f64) -> String {
    (d.to_bits() as i64).to_string()
}

fn policy(kind: &PolicyKind) -> String {
    let p: &dyn MergePolicy = kind.policy();
    let s = p.compound_file_settings();
    let cfs = format!("{} {}", bits(s.no_cfs_ratio), s.max_cfs_segment_size);
    match kind {
        PolicyKind::MockRandom(_) => "mock".into(),
        PolicyKind::Tiered(t) => format!(
            "tiered {} {} {} {} {} {} {cfs}",
            t.config.max_merge_at_once,
            t.config.max_merged_segment_size,
            t.config.floor_segment_size,
            bits(t.config.force_merge_deletes_pct_allowed),
            t.config.segments_per_tier,
            t.config.target_search_concurrency,
        ),
        PolicyKind::Log(l) if l.has_size_fn() => format!("alcoholic {} {cfs}", l.max_merge_size),
        PolicyKind::Log(l) => format!(
            "log {} {} {} {cfs}",
            match l.unit() {
                LogSizeUnit::Docs => "doc",
                LogSizeUnit::Bytes => "byte",
            },
            l.calibrate_size_by_deletes,
            l.merge_factor(),
        ),
        other => panic!("not a policy newMergePolicy makes: {other:?}"),
    }
}

fn scheduler(ms: &MergeSchedulerChoice) -> String {
    match ms {
        MergeSchedulerChoice::Serial => "serial".into(),
        MergeSchedulerChoice::Concurrent {
            max_thread_count,
            max_merge_count,
            ..
        } => format!("cms {max_thread_count} {max_merge_count}"),
        MergeSchedulerChoice::Default { .. } => "cms 1 3".into(),
    }
}

fn analyzer() -> Arc<lucene_analysis::Analyzer> {
    Arc::new(MockAnalyzer::new(&mut JavaRandom::new(0)).build())
}

fn config(r: &mut JavaRandom, ctx: &mut JavaRandom) -> RandomIndexWriterConfig {
    new_index_writer_config(
        &mut Draws::with_context(r, ctx).knobs(false, 1),
        analyzer(),
        0,
    )
}

fn iwc_lines(seed: i64, ctx: &mut JavaRandom) -> Vec<String> {
    let mut r = JavaRandom::new(seed);
    let c = config(&mut r, ctx);
    let ms = match &c.merge_scheduler {
        MergeSchedulerChoice::Serial => "serial".to_string(),
        MergeSchedulerChoice::Default { .. } => "default".to_string(),
        MergeSchedulerChoice::Concurrent {
            always_stall,
            max_thread_count,
            max_merge_count,
            ..
        } => format!("concurrent {always_stall} {max_thread_count} {max_merge_count}"),
    };
    vec![
        format!("iwc ms {ms}"),
        format!("iwc mbd {}", c.max_buffered_docs.unwrap_or(-1)),
        format!("iwc mp {}", policy(&c.merge_policy.kind())),
        format!(
            "iwc flags {} {} {} {} {}",
            c.merged_segment_warmer,
            c.use_compound_file,
            c.reader_pooling,
            c.check_pending_flush_update,
            c.event_listener
        ),
        format!("iwc tail {}", r.next_long()),
    ]
}

fn live_lines(seed: i64, ctx: &mut JavaRandom) -> Vec<String> {
    let mode = [
        LiveFlushMode::ByRam,
        LiveFlushMode::ByDocs,
        LiveFlushMode::Either,
    ][(seed % 3) as usize];
    let mut out = vec![format!(
        "mode {}",
        ["BY_RAM", "BY_DOCS", "EITHER"][(seed % 3) as usize]
    )];
    let mut c = config(&mut JavaRandom::new(seed * 31 + 2), ctx);
    let dir = ByteBuffersDirectory::new();
    let mut w = open_writer(&dir).unwrap();
    c.apply(&mut w).unwrap();
    let mut r = JavaRandom::new(seed * 31 + 1);
    for i in 0..40 {
        let mut d = Draws::with_context(&mut r, ctx).knobs(false, 1);
        maybe_change_live_index_writer_config(&mut d, mode, &mut c, &mut w).unwrap();
        out.push(format!(
            "live {i} {} {} {} {} {} {}",
            c.max_buffered_docs.unwrap_or(-1),
            bits(c.ram_buffer_size_mb),
            c.merged_segment_warmer,
            c.use_compound_file,
            scheduler(&c.merge_scheduler),
            policy(&c.merge_policy.kind()),
        ));
    }
    out.push(format!("live tail {}", r.next_long()));
    w.close().unwrap();
    out
}

#[test]
fn new_index_writer_config_and_live_changes_draw_what_java_draws() {
    let text = fixture();
    let mut expected: Vec<(i64, Vec<&str>)> = Vec::new();
    for line in text.lines() {
        if let Some(seed) = line.strip_prefix("seed ") {
            expected.push((seed.parse().unwrap(), Vec::new()));
        } else {
            expected.last_mut().expect("a seed line first").1.push(line);
        }
    }
    assert_eq!(expected.len(), 200);
    // `random()`: any stream, as long as it is not `r`.
    let mut ctx = JavaRandom::new(0x5eed);
    let mut live_seeds = 0;
    for (seed, lines) in &expected {
        let mut actual = iwc_lines(*seed, &mut ctx);
        if lines.len() > actual.len() {
            actual.extend(live_lines(*seed, &mut ctx));
            live_seeds += 1;
        }
        assert_eq!(&actual, lines, "seed {seed}");
    }
    assert_eq!(live_seeds, 100);
}

/// `dvonly.txt`: `updateDocValues` naming a field no document has yet
/// creates it (Java's `verifyOrCreateDvOnlyField`), `updateNumericDocValue`
/// refuses it, and an update reaching a segment without the field gives its
/// documents the value.
#[test]
fn doc_values_updates_create_doc_values_only_fields_as_java_does() {
    use lucene_codecs::stored_fields::{DocumentVisitor, FieldValue};
    use lucene_index::buffered_updates::{DocValuesUpdate, Term};
    use lucene_index::document::{Document, NumericDocValuesField, Store, StringField};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;
    use lucene_search::directory_reader::DirectoryReader;
    use lucene_search::reader::LeafReader;
    use std::collections::BTreeMap;

    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data/dvonly.txt");
    let expected = fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));

    let id_doc = |id: i32| {
        let mut d = Document::new();
        d.add(StringField::new("id", id.to_string(), Store::Yes));
        d
    };
    let id = |i: &str| Term::new("id", i);
    let dir = ByteBuffersDirectory::new();
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    // The document API's mode, which `RandomIndexWriter` opens in.
    w.enable_explicit_documents().unwrap();
    let numeric = |i: &str, field: &str, v: i64| DocValuesUpdate::Numeric {
        term: id(i),
        field: field.to_string(),
        value: Some(v),
    };
    // updateNumericDocValue refuses a field that does not exist;
    // updateDocValues creates it. The port's message is its own, so the
    // line is rebuilt from the refusal's variant and field: those are what
    // is compared, not Java's wording.
    let mut lines = Vec::new();
    match w.update_numeric_doc_value(id("0"), "n", 5) {
        Err(lucene_index::index_writer::Error::UnknownDocValuesUpdateField(f)) => lines.push(
            format!("refused Can't update [NUMERIC] doc values; the field [{f}] doesn't exist."),
        ),
        other => panic!("{other:?}"),
    }
    w.update_doc_values(id("0"), &[numeric("0", "n", 5)])
        .unwrap();
    w.add_fields_document(&id_doc(0)).unwrap();
    w.commit().unwrap();
    let mut one = id_doc(1);
    one.add(NumericDocValuesField::new("n", 1));
    w.add_fields_document(&one).unwrap();
    w.commit().unwrap();
    w.update_numeric_doc_value(id("0"), "n", 7).unwrap();
    w.update_doc_values(id("1"), &[numeric("1", "m", 3)])
        .unwrap();
    w.update_doc_values(
        id("0"),
        &[DocValuesUpdate::Binary {
            term: id("0"),
            field: "b".to_string(),
            value: Some(b"x".to_vec()),
        }],
    )
    .unwrap();
    w.commit().unwrap();
    let mut two = id_doc(2);
    two.add(NumericDocValuesField::new("m", 4));
    w.add_fields_document(&two).unwrap();
    w.close().unwrap();

    let java = |s: String| {
        // `Numeric` -> `NUMERIC`, `DocsAndFreqs` -> `DOCS_AND_FREQS`.
        let mut out = String::new();
        for (i, c) in s.chars().enumerate() {
            if c.is_uppercase() && i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_uppercase());
        }
        out
    };
    let reader = DirectoryReader::open(&dir).unwrap();
    let mut docs = BTreeMap::new();
    let mut fields = BTreeMap::new();
    for leaf in reader.segment_readers() {
        for fi in &leaf.field_infos().fields {
            fields.insert(
                fi.name.clone(),
                format!(
                    "field {} {} {}",
                    fi.name,
                    java(format!("{:?}", fi.doc_values_type)),
                    java(format!("{:?}", fi.index_options))
                ),
            );
        }
        for doc in 0..leaf.max_doc {
            let mut v = DocumentVisitor::all();
            leaf.document(doc, &mut v).unwrap();
            let stored = v.into_document();
            let FieldValue::String(id) = &stored.fields[0].value else {
                panic!("a string id");
            };
            let mut line = format!("doc {id}");
            for f in ["n", "m"] {
                let value = match leaf.numeric_doc_values(f).unwrap() {
                    Some(mut v) => {
                        if v.advance_exact(doc).unwrap() {
                            v.long_value().to_string()
                        } else {
                            "-".to_string()
                        }
                    }
                    None => "-".to_string(),
                };
                line.push_str(&format!(" {f}={value}"));
            }
            let b = match leaf.binary_doc_values("b").unwrap() {
                Some(mut v) => {
                    if v.advance_exact(doc).unwrap() {
                        String::from_utf8(v.binary_value().to_vec()).unwrap()
                    } else {
                        "-".to_string()
                    }
                }
                None => "-".to_string(),
            };
            line.push_str(&format!(" b={b}"));
            docs.insert(id.clone(), line);
        }
    }
    lines.extend(docs.into_values().chain(fields.into_values()));
    let actual = lines;
    assert_eq!(actual.join("\n") + "\n", expected);
}
