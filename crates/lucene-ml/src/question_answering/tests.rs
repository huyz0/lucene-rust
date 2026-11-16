#![cfg(test)]
use super::*;

fn texts(context: &str) -> Vec<String> {
    segment(context).into_iter().map(|s| s.text).collect()
}

#[test]
fn splits_after_sentence_ends() {
    let s = segment("The sun is out. Is it warm? Yes!");
    assert_eq!(
        texts("The sun is out. Is it warm? Yes!"),
        vec!["The sun is out.", "Is it warm?", "Yes!"]
    );
    assert_eq!((s[0].start, s[0].end), (0, 15));
    assert_eq!((s[1].start, s[1].end), (16, 27));
}

#[test]
fn keeps_abbreviations_inside_a_sentence_and_marks_them() {
    assert_eq!(
        texts("Dr. Smith came. He left."),
        vec!["Dr@@@ Smith came.", "He left."]
    );
    assert_eq!(texts("See e.g. this one."), vec!["See e@@@g@@@ this one."]);
}

#[test]
fn keeps_initials_and_dotted_words_together() {
    assert_eq!(
        texts("It was J.R. Smith. Done."),
        vec!["It was J.R. Smith.", "Done."]
    );
    assert_eq!(texts("a.b. c"), vec!["a.b. c"]);
}

#[test]
fn moves_sentence_marks_behind_a_closing_quote() {
    assert_eq!(
        texts("He said \"go.\" Then left."),
        vec!["He said \"go\".", "Then left."]
    );
}

#[test]
fn drops_trailing_whitespace_and_keeps_an_unmatched_text_whole() {
    assert_eq!(texts("One. "), vec!["One."]);
    assert_eq!(texts("no ending"), vec!["no ending"]);
    assert_eq!(texts(""), vec![""]);
}

#[test]
fn counts_in_utf16_units() {
    let s = segment("a\u{1F600}. b.");
    assert_eq!((s[1].start, s[1].end), (5, 7));
}

#[test]
fn words_take_the_sentence_that_holds_them() {
    let context = "The sun is out. Is it warm?";
    let ids = word_sentence_ids(&segment(context), context);
    assert_eq!(ids, vec![0, 0, 0, 0, 1, 1, 1]);
}

#[test]
fn splitting_matches_java_for_edge_cases() {
    let lens = |s: &str| split_lengths(&s.encode_utf16().collect::<Vec<_>>(), is_space);
    assert_eq!(lens(""), vec![0]);
    assert_eq!(lens("   "), Vec::<usize>::new());
    assert_eq!(lens(" a  bc "), vec![0, 1, 2]);
    assert_eq!(lens("ab"), vec![2]);
}

#[test]
fn tokens_take_their_words_sentence() {
    let types = [0, 0, 1, 1, 1, 1];
    let words = [Some(0), None, Some(0), Some(1), Some(5), None];
    assert_eq!(
        token_sentence_ids(&types, &words, &[7, 8]),
        vec![-100, -100, 7, 8, -100, -100]
    );
}

struct Fixed(Vec<i64>);

impl TorchScript for Fixed {
    fn forward(&self, inputs: &[Input<'_>]) -> Result<Vec<crate::torchscript::TorchOutput>> {
        assert_eq!(
            inputs.iter().map(|i| i.name).collect::<Vec<_>>(),
            [
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "sentence_ids"
            ]
        );
        Ok(vec![crate::torchscript::TorchOutput {
            name: String::new(),
            group: 0,
            shape: vec![self.0.len()],
            data: Data::I64(self.0.clone()),
        }])
    }
}

fn word_level() -> Vec<u8> {
    br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
            "post_processor":{"type":"TemplateProcessing",
              "single":[{"Sequence":{"id":"A","type_id":0}}],
              "pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],
              "special_tokens":{}},
            "decoder":null,
            "model":{"type":"WordLevel","vocab":{"[UNK]":0,"a":1,"b":2},"unk_token":"[UNK]"}}"#
        .to_vec()
}

#[test]
fn answers_with_the_sentences_the_model_names() {
    let model = HighlighterModel::torchscript(
        Box::new(Fixed(vec![1, 7])),
        &word_level(),
        Config::default(),
    )
    .unwrap();
    let hits = model.highlight("a", "a b. b a. a a.").unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0],
        Highlight {
            text: "b a.".into(),
            position: 1,
            start: 5,
            end: 9
        }
    );
}

#[test]
fn answers_nothing_when_no_sentence_is_named() {
    let model = HighlighterModel::torchscript(
        Box::new(Fixed(vec![-1, 9])),
        &word_level(),
        Config::default(),
    )
    .unwrap();
    assert!(model.highlight("a", "a b.").unwrap().is_empty());
}

#[test]
fn refuses_chunk_sizes_the_tokenizer_cannot_cut() {
    for (max_length, stride) in [(0, 0), (4, 4), (4, 9), (usize::MAX, 0)] {
        let config = Config {
            max_length,
            stride,
            ..Config::default()
        };
        let made = HighlighterModel::torchscript(Box::new(Fixed(vec![])), &word_level(), config);
        assert!(matches!(made, Err(MlError::InvalidArgument(_))));
    }
}

#[test]
fn many_short_sentences_are_tagged_without_a_pass_per_sentence() {
    let context = "a. ".repeat(200_000);
    let ids = word_sentence_ids(&segment(&context), &context);
    assert_eq!(ids.len(), 200_000);
    assert_eq!((ids[0], ids[1], ids[199_999]), (0, 1, 199_999));
}

#[test]
fn tagging_matches_a_pass_over_every_sentence() {
    let pieces = [
        "a", "Dr.", "b.", "e.g.", "c!", "d?", " ", "  ", "\n", "x.y.", "\"", "J.R.",
    ];
    let mut seed = 7_u64;
    for _ in 0..400 {
        let mut context = String::new();
        for _ in 0..12 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            context.push_str(pieces[(seed >> 33) as usize % pieces.len()]);
            context.push(' ');
        }
        let sentences = segment(&context);
        let units: Vec<u16> = context.encode_utf16().collect();
        let lengths = split_lengths(&units, is_space);
        let mut expected = vec![0_i64; lengths.len()];
        for (n, s) in sentences.iter().enumerate() {
            let mut start = 0;
            for (w, &len) in lengths.iter().enumerate() {
                if start >= s.start && start + len <= s.end {
                    expected[w] = n as i64;
                }
                start += len + 1;
            }
        }
        assert_eq!(
            word_sentence_ids(&sentences, &context),
            expected,
            "{context:?}"
        );
    }
}

#[test]
fn marks_can_push_a_sentence_end_past_the_text() {
    let context = "Dr. A. B. c";
    let s = segment(context);
    assert!(s.last().unwrap().end > context.len() - 3);
    let ids = word_sentence_ids(&s, context);
    assert_eq!(ids.len(), 4);
}

#[test]
fn reads_every_chunk_of_a_long_context() {
    let config = Config {
        max_length: 6,
        stride: 1,
        ..Config::default()
    };
    let model =
        HighlighterModel::torchscript(Box::new(Fixed(vec![0])), &word_level(), config).unwrap();
    let context = vec!["a"; 20].join(" ");
    assert_eq!(model.highlight("b", &context).unwrap().len(), 1);
}
