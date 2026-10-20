//! The flexible parser against Lucene 10.5.0
//! (`fixtures/src/GenQueryParserFlexible.java`).

use std::path::PathBuf;

use super::token_manager::TokenManager;
use crate::queryparser::javacc::{CharStream, Lexer};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/queryparser_flexible")
}

/// `AnalysisRows.esc` undone.
fn unesc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                out.push(
                    char::from_u32(u32::from_str_radix(&h, 16).unwrap()).unwrap_or('\u{fffd}'),
                );
            }
            Some(o) => out.push(o),
            None => out.push('\\'),
        }
    }
    out
}

/// Every query's token stream (kind, column, image) or lexical error is
/// `StandardSyntaxParserTokenManager`'s.
#[test]
fn the_lexer_splits_every_query_as_lucene_does() {
    let tokens = std::fs::read_to_string(fixtures().join("tokens.tsv")).unwrap();
    // the classic parser's seeded fragment joins: query, then the same two
    // columns (the parse's follow)
    let fragments = std::fs::read_to_string(fixtures().join("fragments.tsv")).unwrap();
    let mut bad = Vec::new();
    let mut rows = 0;
    for line in tokens.lines().chain(fragments.lines()) {
        let cols: Vec<&str> = line.split('\t').take(3).collect();
        let query = unesc(cols[0]);
        let want: Vec<String> = match cols[1] {
            "T" => cols[2]
                .split(' ')
                .map(|t| {
                    let mut parts = t.splitn(3, ':');
                    let kind = parts.next().unwrap();
                    let col = parts.next().unwrap();
                    format!("{kind}:{col}:{}", unesc(parts.next().unwrap_or("")))
                })
                .collect(),
            _ => vec![format!("E {}", unesc(cols[2]))],
        };
        let mut tm = TokenManager::new(CharStream::new(&query));
        let mut got = Vec::new();
        loop {
            match tm.next_token() {
                Ok(t) => {
                    got.push(format!(
                        "{}:{}:{}",
                        t.kind,
                        t.begin_column,
                        String::from_utf16_lossy(&t.image)
                    ));
                    if t.kind == 0 {
                        break;
                    }
                }
                Err(e) => {
                    got = vec![format!("E {}", e.0)];
                    break;
                }
            }
        }
        rows += 1;
        if got != want {
            bad.push(format!("{query:?}\n  want {want:?}\n  got  {got:?}"));
        }
    }
    assert!(rows > 3300, "only {rows} rows");
    let shown: Vec<&str> = bad.iter().take(20).map(String::as_str).collect();
    assert!(
        bad.is_empty(),
        "{} of {rows} differ:\n{}",
        bad.len(),
        shown.join("\n")
    );
}

mod parse {
    use std::sync::Arc;

    use lucene_analysis::{Analyzer, StandardAnalyzer};

    use super::super::builders::build;
    use super::super::escape::CaseLocale;
    use super::super::nodes::{FieldData, IntervalFn, Kind, Modifier, Node, Text};
    use super::super::points::{PointType, PointsConfig, PointsFormat};
    use super::super::{Operator, StandardQueryParser};
    use lucene_analysis::miscellaneous::{DateLocale, DecimalFormat};

    fn std_parser() -> StandardQueryParser {
        StandardQueryParser::with_analyzer(Arc::new(Analyzer::new(StandardAnalyzer::new())))
    }

    fn error(p: &StandardQueryParser, q: &str, field: Option<&str>) -> String {
        let e = p.parse(q, field).unwrap_err();
        format!("{}: {}", e.class, e.message)
    }

    #[test]
    fn the_parser_starts_with_lucenes_defaults() {
        let p = StandardQueryParser::default();
        assert!(p.analyzer.is_none());
        assert_eq!(p.default_operator, Operator::Or);
        assert!(!p.allow_leading_wildcard);
        assert!(p.enable_position_increments);
        assert_eq!(p.fuzzy_min_sim, 2.0);
        // with no analyzer the text is kept as typed
        assert_eq!(p.parse("Foo", Some("f")).unwrap().to_string("f"), "Foo");
    }

    #[test]
    fn a_field_less_term_needs_multi_fields() {
        let p = std_parser();
        assert_eq!(
            error(&p, "a", None),
            "java.lang.IllegalArgumentException: StandardQueryConfigHandler.ConfigurationKeys.MULTI_FIELDS should be set on the QueryConfigHandler"
        );
        // no fields at all leave the field unset, which a field config refuses
        let mut p = std_parser();
        p.set_multi_fields(&[]);
        assert_eq!(
            error(&p, "a", None),
            "java.lang.IllegalArgumentException: field name must not be null!"
        );
        // one field is that field
        p.set_multi_fields(&["t"]);
        assert_eq!(p.parse("a", None).unwrap().to_string("f"), "t:a");
    }

    #[test]
    fn an_int_past_its_range_is_a_number_format_exception() {
        assert_eq!(
            error(&std_parser(), "(a b)@99999999999", Some("f")),
            "java.lang.NumberFormatException: For input string: \"99999999999\""
        );
    }

    #[test]
    fn nodes_the_builders_cannot_build_fail_as_java_does() {
        let p = std_parser();
        let e = build(&p, &Node::leaf(Kind::Deleted), None).unwrap_err();
        assert_eq!(
            e.message,
            "Cannot convert query to lucene syntax: [DELETEDCHILD] error: org.apache.lucene.queryparser.flexible.core.nodes.DeletedQueryNode"
        );
        let interval = Node::leaf(Kind::Interval {
            field: None,
            source: IntervalFn::AnalyzedText("a".into()),
        });
        assert_eq!(
            build(&p, &interval, None).unwrap_err().class,
            "java.lang.NullPointerException"
        );
        // `@` over a group that is not a boolean
        let term = Node::leaf(Kind::Field(FieldData::new(
            Some("f".into()),
            Text::plain("a"),
        )));
        let msm = Node::with(
            Kind::MinShouldMatch(1),
            vec![Node::with(Kind::Group, vec![term.clone()])],
        );
        assert_eq!(
            build(&p, &msm, None).unwrap_err().class,
            "java.lang.ClassCastException"
        );
        // a boolean past the clause limit
        let many = Node::with(Kind::Boolean, vec![term.clone(); 1025]);
        assert!(build(&p, &many, None)
            .unwrap_err()
            .message
            .starts_with("Too many boolean clauses, the maximum supported is 1,024"));
        // a boost with no child builds nothing
        assert_eq!(
            build(&p, &Node::leaf(Kind::Boost(2.0)), None).unwrap(),
            None
        );
        // ... and `@` over that null dereferences it
        let msm = Node::with(Kind::MinShouldMatch(1), vec![Node::leaf(Kind::Boost(2.0))]);
        let e = build(&p, &msm, None).unwrap_err();
        assert_eq!(e.class, "java.lang.NullPointerException");
        assert_eq!(
            e.message,
            "Cannot invoke \"org.apache.lucene.search.BooleanQuery.clauses()\" because \"booleanQuery\" is null"
        );
    }

    fn config() -> PointsConfig {
        let us = DateLocale::for_language_tag("en-US").unwrap();
        PointsConfig::new(
            PointsFormat::Decimal(DecimalFormat::number_instance(us)),
            PointType::Int,
        )
    }

    #[test]
    fn every_node_names_its_java_class() {
        let d = || FieldData::new(None, Text::plain("x"));
        let kinds = vec![
            Kind::Boolean,
            Kind::And,
            Kind::Or,
            Kind::Synonym,
            Kind::Boost(1.0),
            Kind::Group,
            Kind::Modifier(Modifier::Req),
            Kind::BooleanModifier(Modifier::Req),
            Kind::Slop(1),
            Kind::MinShouldMatch(1),
            Kind::TokenizedPhrase,
            Kind::MultiPhrase,
            Kind::Field(d()),
            Kind::Quoted(d()),
            Kind::Fuzzy {
                data: d(),
                similarity: 1.0,
                prefix_length: 0,
            },
            Kind::Wildcard(d()),
            Kind::PrefixWildcard(d()),
            Kind::Regexp {
                field: None,
                text: Text::plain("x"),
            },
            Kind::TermRange {
                lower_inclusive: true,
                upper_inclusive: true,
            },
            Kind::MatchAll,
            Kind::MatchNoDocs,
            Kind::Deleted,
            Kind::NoTokenFound,
            Kind::Interval {
                field: None,
                source: IntervalFn::AnalyzedText("x".into()),
            },
            Kind::Point {
                field: None,
                value: None,
                format: config().number_format().clone(),
            },
            Kind::PointRange {
                lower_inclusive: true,
                upper_inclusive: true,
                config: config(),
            },
        ];
        let mut names = std::collections::HashSet::new();
        for k in kinds {
            let name = Node::leaf(k).class_name();
            assert!(
                name.starts_with("org.apache.lucene.queryparser.flexible."),
                "{name}"
            );
            names.insert(name);
        }
        assert_eq!(names.len(), 26, "one class each");
        // only a wildcard prints itself for a message
        assert_eq!(
            Node::leaf(Kind::MatchAll)
                .to_query_string(None, CaseLocale::Root)
                .unwrap(),
            "*:*"
        );
    }
}
