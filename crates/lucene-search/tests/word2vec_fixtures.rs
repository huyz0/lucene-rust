//! M11 T11.5: the `word2vec` synonym package against
//! `fixtures/src/GenWord2Vec.java`: both model zips read to the same model,
//! `getSynonyms` finds the same synonyms with the same boost bits for every
//! term, and `Word2VecSynonymFilter` gives the same tokens.

use std::sync::Arc;

use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{StrReader, TokenStream};
use lucene_search::word2vec::{read_dl4j_model, Word2VecSynonymFilter, Word2VecSynonymProvider};

fn data(name: &str) -> String {
    format!(
        "{}/../../fixtures/data/word2vec/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn provider(model: &str) -> Arc<Word2VecSynonymProvider> {
    let zip = std::fs::read(data(model)).unwrap();
    Arc::new(Word2VecSynonymProvider::new(read_dl4j_model(&zip).unwrap()).unwrap())
}

#[test]
fn synonyms_match_lucene() {
    let text = std::fs::read_to_string(data("synonyms.txt")).unwrap();
    let mut p = None;
    let mut checked = 0;
    for row in text.lines() {
        let f: Vec<&str> = row.split('\t').collect();
        if f[0] == "#model" {
            p = Some(provider(f[1]));
            continue;
        }
        let p = p.as_ref().unwrap();
        let found: Vec<String> = p
            .synonyms(
                f[0].as_bytes(),
                f[1].parse().unwrap(),
                f[2].parse().unwrap(),
            )
            .unwrap()
            .into_iter()
            .map(|t| {
                format!(
                    "{}:{:x}",
                    String::from_utf8(t.term).unwrap(),
                    t.boost.to_bits()
                )
            })
            .collect();
        assert_eq!(found.join(" "), f[3], "{row}");
        checked += 1;
    }
    assert_eq!(checked, 2 * 81 * 3);
}

/// The generator's `LINES`.
const LINES: [&str; 4] = ["w00 w01 w08 unknown w16", "école 日本 a_b", "w79", ""];

#[test]
fn filter_matches_lucene() {
    let mut actual = String::new();
    for model in ["model_b64.zip", "model_plain.zip"] {
        actual += &format!("#model\t{model}\n");
        let p = provider(model);
        let mut ts = Word2VecSynonymFilter::new(WhitespaceTokenizer::new(), Arc::clone(&p), 2, 0.8);
        for (ln, line) in LINES.iter().enumerate() {
            ts.as_tokenizer()
                .unwrap()
                .set_reader(Box::new(StrReader::new(*line)))
                .unwrap();
            ts.reset().unwrap();
            while ts.increment_token().unwrap() {
                let a = ts.attributes();
                actual += &format!(
                    "{ln}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                    a.term(),
                    a.start_offset(),
                    a.end_offset(),
                    a.position_increment(),
                    a.position_length(),
                    a.token_type()
                );
            }
            ts.end().unwrap();
            ts.close().unwrap();
        }
    }
    assert_eq!(actual, std::fs::read_to_string(data("filter.txt")).unwrap());
}
