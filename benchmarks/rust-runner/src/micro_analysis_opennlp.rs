//! M12 T12.5's pairs (`benchmarks/micro/java/AnalysisOpenNlpMicro.java` is
//! the Java twin): analysis-opennlp chains, built by their factories over
//! the models `fixtures/src/GenAnalysisOpenNLP.java` trains, over that
//! fixture's text, one document per line.

use std::hint::black_box;
use std::time::Duration;

use lucene_analysis::factory::CustomAnalyzer;
use lucene_analysis::Analyzer;

use super::{consume_stream, measure};

const DIR: &str = "fixtures/data/analysis_opennlp";

fn run(name: &str, a: &Analyzer, docs: &[String], w: Duration, m: Duration) {
    measure(name, w, m, || {
        let mut tokens = 0u64;
        for text in docs {
            let mut ts = a.token_stream("body", black_box(text)).unwrap();
            tokens += consume_stream(&mut ts);
        }
        tokens
    });
}

/// The chain: the tokenizer, then `filters` (name and arguments each).
fn chain(filters: &[(&str, &[&str])]) -> Analyzer {
    let mut b = CustomAnalyzer::builder_with_dir(DIR)
        .unwrap()
        .with_tokenizer(
            "openNlp",
            &[
                "sentenceModel",
                "en-sent.bin",
                "tokenizerModel",
                "en-token.bin",
            ],
        )
        .unwrap();
    for (name, args) in filters {
        b = b.add_token_filter(name, args).unwrap();
    }
    b.build().unwrap().into_analyzer()
}

pub(super) fn bench_analysis_opennlp(w: Duration, m: Duration) {
    lucene_analysis_opennlp::register_factories().unwrap();
    let text = std::fs::read_to_string(format!("{DIR}/opennlp_text.txt")).unwrap();
    let docs: Vec<String> = text
        .split('\n')
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    let pos: (&str, &[&str]) = ("openNlppos", &["posTaggerModel", "en-pos-maxent.bin"]);
    run("opennlp_tokenizer", &chain(&[]), &docs, w, m);
    run("opennlp_pos", &chain(&[pos]), &docs, w, m);
    run(
        "opennlp_pos_perceptron",
        &chain(&[("openNlppos", &["posTaggerModel", "en-pos-perceptron.bin"])]),
        &docs,
        w,
        m,
    );
    run(
        "opennlp_chunker",
        &chain(&[pos, ("openNlpChunker", &["chunkerModel", "en-chunker.bin"])]),
        &docs,
        w,
        m,
    );
    run(
        "opennlp_lemmatizer",
        &chain(&[
            pos,
            (
                "openNlpLemmatizer",
                &[
                    "dictionary",
                    "en-lemmas.dict",
                    "lemmatizerModel",
                    "en-lemmatizer.bin",
                ],
            ),
        ]),
        &docs,
        w,
        m,
    );
}
