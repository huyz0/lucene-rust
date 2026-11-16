#![cfg(test)]
use super::*;
use crate::config::{MonitorConfiguration, MonitorUpdateListener};
use crate::facade::Monitor;
use crate::index::MonitorQuery;
use crate::presearcher::TermFilteredPresearcher;
use crate::serializer::MonitorQuerySerializer;
use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::query::{Clause, TermQuery};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

fn serializer() -> MonitorQuerySerializer {
    MonitorQuerySerializer::from_parser(Arc::new(|t: &str| {
        Ok(Clause::Term(TermQuery::new("f", t.as_bytes().to_vec())))
    }))
}

fn mq(id: &str, text: &str) -> MonitorQuery {
    let clause = Clause::Term(TermQuery::new("f", text.as_bytes().to_vec()));
    MonitorQuery::new(id, clause, Some(text.to_string()), BTreeMap::new()).unwrap()
}

fn open(path: &Path, read_only: bool) -> Result<Monitor> {
    let config = MonitorConfiguration::default().set_directory(path, serializer(), read_only);
    Monitor::with_configuration(
        Arc::new(Analyzer::new(StandardAnalyzer::new())),
        TermFilteredPresearcher::default(),
        Vec::new(),
        &config,
    )
}

#[test]
fn queries_survive_a_reopen_and_deletes_and_replacements_persist() {
    let tmp = lucene_util::test_support::TempDir::new("monitor-persist");
    let path: &Path = &tmp;
    {
        let mut m = open(path, false).unwrap();
        m.register(vec![mq("a", "fox"), mq("b", "dog"), mq("c", "cat")])
            .unwrap();
        m.register(vec![mq("b", "wolf")]).unwrap();
        m.delete_by_id(&["c"]).unwrap();
    }
    let m = open(path, false).unwrap();
    assert_eq!(m.query_ids(), vec!["a".to_string(), "b".to_string()]);
    assert_eq!(
        m.get_query("b").unwrap().query_string.as_deref(),
        Some("wolf")
    );
    let hit = m.match_doc(&[("f", "the wolf")]).unwrap();
    assert_eq!(
        hit.matches
            .iter()
            .map(|x| x.query_id.as_str())
            .collect::<Vec<_>>(),
        vec!["b"]
    );
    drop(m);
    let mut m = open(path, false).unwrap();
    m.clear().unwrap();
    drop(m);
    let m = open(path, false).unwrap();
    assert_eq!(m.query_count(), 0);
    assert_eq!(PersistentStore::open(path).unwrap().num_docs().unwrap(), 0);
}

#[test]
fn a_query_without_text_and_an_index_without_a_serializer_are_refused() {
    let tmp = lucene_util::test_support::TempDir::new("monitor-persist-refuse");
    let path: &Path = &tmp;
    let mut m = open(path, false).unwrap();
    let no_text = MonitorQuery::new(
        "x",
        Clause::Term(TermQuery::new("f", b"x".to_vec())),
        None,
        BTreeMap::new(),
    )
    .unwrap();
    let e = m.register(vec![no_text]).unwrap_err().to_string();
    assert!(e.contains("null string representation"), "{e}");
    m.register(vec![mq("a", "fox")]).unwrap();
    drop(m);
    let config = MonitorConfiguration {
        index_path: Some(path.to_path_buf()),
        ..MonitorConfiguration::default()
    };
    let e = Monitor::with_configuration(
        Arc::new(Analyzer::new(StandardAnalyzer::new())),
        TermFilteredPresearcher::default(),
        Vec::new(),
        &config,
    )
    .err()
    .unwrap()
    .to_string();
    assert!(e.contains("no MonitorQuerySerializer"), "{e}");
    // an unparsable stored query is reported, not skipped
    let bad = MonitorQuerySerializer::from_parser(Arc::new(|_| Err("boom".to_string())));
    let config = MonitorConfiguration::default().set_directory(path, bad, true);
    let e = Monitor::with_configuration(
        Arc::new(Analyzer::new(StandardAnalyzer::new())),
        TermFilteredPresearcher::default(),
        Vec::new(),
        &config,
    )
    .err()
    .unwrap()
    .to_string();
    assert!(
        e.contains("Couldn't parse some queries") && e.contains("boom"),
        "{e}"
    );
}

#[test]
fn listeners_hear_updates_deletes_and_clears() {
    #[derive(Default)]
    struct Log(Mutex<Vec<String>>);
    impl MonitorUpdateListener for Log {
        fn after_update(&self, updates: &[MonitorQuery]) {
            self.0
                .lock()
                .unwrap()
                .push(format!("update {}", updates.len()));
        }
        fn after_delete(&self, ids: &[String]) {
            self.0
                .lock()
                .unwrap()
                .push(format!("delete {}", ids.join(",")));
        }
        fn after_clear(&self) {
            self.0.lock().unwrap().push("clear".into());
        }
    }
    let log = Arc::new(Log::default());
    let mut m = Monitor::new(Arc::new(Analyzer::new(StandardAnalyzer::new())));
    m.add_query_index_update_listener(log.clone());
    m.register(vec![mq("a", "fox"), mq("b", "dog")]).unwrap();
    m.delete_by_id(&["a"]).unwrap();
    m.clear().unwrap();
    assert_eq!(
        *log.0.lock().unwrap(),
        vec!["update 2", "delete a", "clear"]
    );
}

#[test]
fn a_copy_of_lucenes_index_can_be_appended_to() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/monitor/index");
    let tmp = lucene_util::test_support::TempDir::new("monitor-persist-append");
    let path: &Path = &tmp;
    for f in std::fs::read_dir(&src).unwrap().flatten() {
        std::fs::copy(f.path(), path.join(f.file_name())).unwrap();
    }
    let parser = lucene_queryparser::classic::QueryParser::new(
        "body",
        Arc::new(Analyzer::new(StandardAnalyzer::new())),
    );
    let ser = MonitorQuerySerializer::from_parser(Arc::new(move |t: &str| {
        parser
            .parse(t)
            .map(|p| p.to_clause(1))
            .map_err(|e| e.message)
    }));
    let config = MonitorConfiguration::default().set_directory(path, ser.clone(), false);
    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let mut m = Monitor::with_configuration(
        Arc::clone(&analyzer),
        TermFilteredPresearcher::default(),
        Vec::new(),
        &config,
    )
    .unwrap();
    assert_eq!(m.query_count(), 40);
    let fox = ser
        .deserialize(&ser.serialize(&mq("new", "zebra")).unwrap())
        .unwrap();
    m.register(vec![fox]).unwrap();
    m.delete_by_id(&["q0"]).unwrap();
    drop(m);
    let m = Monitor::with_configuration(
        analyzer,
        TermFilteredPresearcher::default(),
        Vec::new(),
        &config,
    )
    .unwrap();
    assert_eq!(m.query_count(), 40);
    assert!(m.get_query("new").is_some() && m.get_query("q0").is_none());
    assert_eq!(m.match_doc(&[("body", "zebra")]).unwrap().match_count(), 1);
}
