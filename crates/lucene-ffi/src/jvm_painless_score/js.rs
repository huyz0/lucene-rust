//! A JavaScript script's runs on a segment, through `lucene-script-js`: the
//! segment's [`ScoreEnv`] reads the fields, a column of the call's documents
//! at a time, and a script's failure is [`JS_ERROR`] for the plugin to raise
//! as a `ScriptException` of lang `js`.

use lucene_painless::value::ListKind;
use lucene_script_js::{Failure, Fields, Kind, Limits, Out, Param, Params, Script};

use super::*;

/// What a JavaScript script's failure starts with: then its run number, the
/// exception's name, its message, and its line and column (`-1` for none),
/// split by `\u{1}`.
const JS_ERROR: &str = "\u{1}js-script-error\u{1}";

/// The script of a spec reading `fields` (each by its spec's index).
pub(super) fn script(source: &str, timeout_ms: i32, heap_mb: i32, fields: &[FieldSpec]) -> Script {
    let fields = fields
        .iter()
        .map(|f| {
            let kind = match f.kind {
                FieldKind::Long | FieldKind::Double | FieldKind::Float | FieldKind::HalfFloat => {
                    Kind::Number
                }
                FieldKind::Boolean => Kind::Boolean,
                FieldKind::Keyword => Kind::Keyword,
                FieldKind::Unsupported | FieldKind::Unmapped => Kind::Unreadable,
            };
            (f.name.clone(), kind)
        })
        .collect();
    let limits = Limits {
        timeout: std::time::Duration::from_millis(timeout_ms.unsigned_abs().into()),
        max_heap: (heap_mb.unsigned_abs() as usize).saturating_mul(1 << 20),
    };
    Script::new(source, fields, limits)
}

/// A request's params as the script reads them.
pub(super) fn params(c: &ir::Constant) -> Params {
    Params::new(&param(c))
}

fn param(c: &ir::Constant) -> Param {
    match c {
        ir::Constant::Null => Param::Null,
        ir::Constant::Bool(b) => Param::Bool(*b),
        ir::Constant::Byte(n) => Param::Number(f64::from(*n)),
        ir::Constant::Short(n) => Param::Number(f64::from(*n)),
        ir::Constant::Char(n) => Param::String(String::from_utf16_lossy(&[*n])),
        ir::Constant::Int(n) => Param::Number(f64::from(*n)),
        ir::Constant::Long(n) => Param::Number(*n as f64),
        ir::Constant::Float(n) => Param::Number(f64::from(*n)),
        ir::Constant::Double(n) => Param::Number(*n),
        ir::Constant::Str(s) | ir::Constant::Pattern(s, _) => Param::String(s.clone()),
        ir::Constant::List(items) => Param::List(items.iter().map(param).collect()),
        ir::Constant::Map(entries) => Param::Map(
            entries
                .iter()
                .map(|(k, v)| {
                    let k = match k {
                        ir::Constant::Str(s) => s.clone(),
                        k => match param(k) {
                            Param::Number(n) => lucene_painless::number::double_to_string(n),
                            Param::Bool(b) => b.to_string(),
                            Param::String(s) => s,
                            _ => "null".to_string(),
                        },
                    };
                    (k, param(v))
                })
                .collect(),
        ),
    }
}

/// A JavaScript script on one segment.
pub(super) struct JsLeaf {
    spec: Arc<Spec>,
    /// This run's name for the segment, which the thread's term cache keys
    /// a keyword's ordinals by.
    segment: u64,
}

impl JsLeaf {
    pub(super) fn new(spec: &Arc<Spec>, segment: u64) -> Self {
        JsLeaf {
            spec: Arc::clone(spec),
            segment,
        }
    }

    fn js(&self) -> &JsSource {
        self.spec.js.as_ref().expect("a js spec")
    }

    /// What the script returns for the current document.
    pub(super) fn run_object(&self, env: &mut ScoreEnv<'_>) -> Result<Value> {
        let out = self.run_one(env)?;
        value(out).map_err(|what| self.wrong_type(&what))
    }

    /// What the script returns for the current document, as a number (a
    /// filter's boolean as `1` or `0`).
    pub(super) fn run_value(&self, env: &mut ScoreEnv<'_>) -> Result<f64> {
        match (self.run_one(env)?, self.spec.filter) {
            (Out::Bool(b), true) => Ok(f64::from(u8::from(b))),
            (Out::Number(v), false) => Ok(v),
            (other, _) => Err(self.wrong_type(&type_name(&other))),
        }
    }

    fn run_one(&self, env: &mut ScoreEnv<'_>) -> Result<Out> {
        let js = self.js();
        let score = self.spec.needs_score.then_some(env.score);
        let doc = [env.doc];
        let mut docs = Docs { env, docs: &doc };
        lucene_script_js::run_one(&js.script, &js.params, self.segment, score, &mut docs)
            .map_err(|f| self.error(&f))
    }

    /// The scores (or a filter's `1` and `0`) of `docs`, checked as
    /// `ScriptScorer` checks a score, in document order.
    pub(super) fn fill_batch(
        &self,
        env: &mut ScoreEnv<'_>,
        docs: &[i32],
        scores: &[f32],
        out: &mut [f64],
        has: &mut [bool],
    ) -> Result<()> {
        let js = self.js();
        let filter = self.spec.filter;
        let scores = self.spec.needs_score.then_some(&scores[..docs.len()]);
        let n = docs.len();
        let mut fields = Docs { env, docs };
        let r = lucene_script_js::run_batch(
            &js.script,
            &js.params,
            self.segment,
            n,
            scores,
            filter,
            &mut fields,
            &mut out[..n],
        );
        // A score check fails before a later document's exception.
        let upto = match &r {
            Ok(()) => n,
            Err(Failure::Thrown { at, .. }) => *at,
            Err(_) => 0,
        };
        if !filter {
            for (k, o) in out[..upto].iter_mut().enumerate() {
                *o = f64::from(checked(*o as f32, docs[k])?);
            }
        }
        r.map_err(|f| self.error(&f))?;
        has[..n].fill(true);
        Ok(())
    }

    /// The error for a script that returned a `what`.
    fn wrong_type(&self, what: &str) -> Error {
        let spec = &self.spec;
        let message = if spec.filter {
            format!("a filter script must return a boolean, not {what}")
        } else if spec.agg {
            format!("an aggregation script must return a number, a boolean or an array of them, not {what}")
        } else {
            let context = if spec.sort {
                "sort"
            } else if spec.terms_set {
                "terms_set"
            } else {
                "score"
            };
            format!("a {context} script must return a number, not {what}")
        };
        self.error(&Failure::Thrown {
            at: 0,
            name: "TypeError".to_string(),
            message,
            line: None,
            column: None,
        })
    }

    /// The search's error for `f`.
    fn error(&self, f: &Failure) -> Error {
        let run = self.spec.script_id;
        let (name, message, line, column) = match f {
            Failure::Host(m) => return Error::IllegalArgument(format!("js script: {m}")),
            Failure::Thrown {
                name,
                message,
                line,
                column,
                ..
            } => (name.as_str(), message.clone(), *line, *column),
            Failure::Compile {
                message,
                line,
                column,
            } => ("SyntaxError", message.clone(), *line, *column),
            Failure::TimedOut(_) | Failure::OutOfMemory(_) => ("Error", f.to_string(), None, None),
        };
        let at = |p: Option<usize>| p.map_or(-1, |p| i64::try_from(p).unwrap_or(-1));
        Error::IllegalArgument(format!(
            "{JS_ERROR}{run}\u{1}{name}\u{1}{message}\u{1}{}\u{1}{}",
            at(line),
            at(column)
        ))
    }
}

fn type_name(out: &Out) -> String {
    match out {
        Out::Null => "null".to_string(),
        Out::Number(_) => "number".to_string(),
        Out::Bool(_) => "boolean".to_string(),
        Out::String(_) => "string".to_string(),
        Out::List(_) => "an array".to_string(),
        Out::Other(t) => t.clone(),
    }
}

/// What a script returned, as a Painless value: an array as a list.
fn value(out: Out) -> std::result::Result<Value, String> {
    Ok(match out {
        Out::Null => Value::Null,
        Out::Number(v) => Value::Double(v),
        Out::Bool(b) => Value::Bool(b),
        Out::String(s) => Value::string(s),
        Out::List(items) => Value::list(
            ListKind::ArrayList,
            items
                .into_iter()
                .map(value)
                .collect::<std::result::Result<_, _>>()?,
        ),
        Out::Other(t) => return Err(t),
    })
}

/// The call's documents' fields.
struct Docs<'e, 'c> {
    env: &'e mut ScoreEnv<'c>,
    docs: &'e [i32],
}

impl Fields for Docs<'_, '_> {
    fn load(
        &mut self,
        field: usize,
        values: &mut Vec<f64>,
        starts: &mut Vec<i32>,
    ) -> std::result::Result<(), String> {
        let kind = self
            .env
            .source
            .fields
            .get(field)
            .map(|f| f.kind)
            .ok_or_else(|| format!("no field {field}"))?;
        let doc = self.env.doc;
        let mut r = Ok(());
        for &d in self.docs {
            starts.push(i32::try_from(values.len()).map_err(|_| "a column too long".to_string())?);
            self.env.doc = d;
            match self.env.load(field) {
                Ok(cur) => values.extend(cur.items[..cur.count].iter().map(|&v| match kind {
                    FieldKind::Long | FieldKind::Boolean | FieldKind::Keyword => v as f64,
                    _ => f64::from_bits(v as u64),
                })),
                Err(t) => {
                    r = Err(t.message().unwrap_or_default());
                    break;
                }
            }
        }
        self.env.doc = doc;
        r?;
        starts.push(i32::try_from(values.len()).map_err(|_| "a column too long".to_string())?);
        Ok(())
    }

    fn term(&mut self, field: usize, ord: i64) -> std::result::Result<String, String> {
        if field >= self.env.source.fields.len() {
            return Err(format!("no field {field}"));
        }
        self.env
            .term(field, ord)
            .map(|v| v.java_to_string())
            .map_err(|t| t.message().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jvm_nodes::tests::{open, Blob};
    use crate::jvm_reader::ffi_close_jvm_reader;
    use crate::jvm_reader::tests::{run_limit, run_sorted};
    use lucene_util::test_support::TempDir;

    /// The Painless fixture's index (`ScoreCorpus.java`): 300 documents in
    /// two segments, document `i` the `i`-th.
    fn index() -> (TempDir, u64) {
        let tmp = TempDir::new("js-score");
        let dir = format!(
            "{}/../../fixtures/data/painless-score/index",
            env!("CARGO_MANIFEST_DIR")
        );
        for e in std::fs::read_dir(dir).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        let h = open(&tmp);
        (tmp, h)
    }

    /// `{name: value}` as the plugin sends params.
    fn params(entries: &[(&str, f64)]) -> Blob {
        let mut b = Blob::default().u8(12).i32(entries.len() as i32);
        for (k, v) in entries {
            b = b.u8(9).str(k).u8(8).f64(*v);
        }
        b
    }

    /// A JavaScript spec of `context` reading `fields` (name, kind).
    fn spec(context: u8, source: &str, params: &Blob, fields: &[(&str, u8)]) -> Blob {
        let mut b = Blob::default()
            .u8(context | JS_SCRIPT)
            .str(source)
            .bytes(&params.0)
            .str("idx")
            .i32(0)
            .i32(7)
            .i32(fields.len() as i32);
        for (name, kind) in fields {
            b = b.str(name).str(name).u8(*kind);
        }
        b.i32(10_000).i32(64)
    }

    fn likes(i: i32) -> f64 {
        f64::from((i * 7919) % 1000)
    }

    #[test]
    fn a_score_script_scores_every_document() {
        let (_tmp, h) = index();
        let s = spec(
            1,
            "doc['likes'].value * params.w + (doc['flag'].value ? 1 : 0) + _score",
            &params(&[("w", 0.5)]),
            &[("likes", 0), ("flag", 4)],
        );
        let node = Blob::tree().u8(NODE_PAINLESS_SCORE).u8(5).raw(&s);
        let (hits, total, _) = run_limit(h, &node.0, 1000, i64::MAX).unwrap();
        assert_eq!(total, 300);
        assert_eq!(hits.len(), 300);
        for (doc, score) in hits {
            let flag = if doc % 3 == 0 { 1.0 } else { 0.0 };
            // match_all scores 1.
            let want = (likes(doc) * 0.5 + flag + 1.0) as f32;
            assert_eq!(score, want, "doc {doc}");
        }
        assert_eq!(check(&s.0[..]).unwrap(), None);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn a_filter_script_matches_where_it_returns_true() {
        let (_tmp, h) = index();
        let s = spec(
            2,
            "doc['tag'].values.includes('tag1') && doc['vals'].size() === 3",
            &params(&[]),
            &[("tag", 5), ("vals", 0)],
        );
        let node = Blob::tree().u8(NODE_PAINLESS_FILTER).raw(&s);
        let (hits, total, _) = run_limit(h, &node.0, 1000, i64::MAX).unwrap();
        let want: Vec<i32> = (0..300)
            .filter(|i| i % 5 == 1 || (i % 4 == 0 && (i + 2) % 5 == 1))
            .collect();
        let mut got: Vec<i32> = hits.iter().map(|(d, _)| *d).collect();
        got.sort_unstable();
        assert_eq!(got, want);
        assert_eq!(total, want.len() as i64);
        // Counting runs the same script.
        let (_, counted, _) = run_limit(h, &node.0, 0, i64::MAX).unwrap();
        assert_eq!(counted, total);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn a_sort_script_orders_by_its_numbers() {
        let (_tmp, h) = index();
        let s = spec(3, "-doc['likes'].value", &params(&[]), &[("likes", 0)]);
        let mut sort = vec![1, crate::jvm_reader::SORT_SCRIPT, 0];
        sort.extend_from_slice(&s.0);
        sort.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        let (hits, total, _) = run_sorted(h, &Blob::tree().u8(5).0, &sort, 5, i64::MAX).unwrap();
        assert_eq!(total, 300);
        let mut want: Vec<(i32, f64)> = (0..300).map(|i| (i, -likes(i))).collect();
        want.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let got: Vec<(i32, i64)> = hits.iter().map(|(d, v)| (*d, v[0])).collect();
        let want: Vec<(i32, i64)> = want[..5]
            .iter()
            .map(|(d, v)| (*d, lucene_util::numeric_utils::double_to_sortable_long(*v)))
            .collect();
        assert_eq!(got, want);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn a_script_error_is_the_scripts() {
        let (_tmp, h) = index();
        // Every seventh document has no price.
        let s = spec(0, "doc['price'].value", &params(&[]), &[("price", 1)]);
        let node = Blob::tree().u8(NODE_PAINLESS_SCORE).u8(5).raw(&s);
        assert!(run_limit(h, &node.0, 10, i64::MAX).is_err());
        let e = crate::error::last_error();
        assert!(
            e.contains("\u{1}js-script-error\u{1}7\u{1}Error\u{1}A document doesn't have a value for a field!"),
            "{e:?}"
        );
        assert!(e.ends_with("\u{1}1\u{1}13"), "{e:?}");
        // A negative score fails as `ScriptScorer` fails it.
        let s = spec(0, "doc['likes'].value - 500", &params(&[]), &[("likes", 0)]);
        let node = Blob::tree().u8(NODE_PAINLESS_SCORE).u8(5).raw(&s);
        assert!(run_limit(h, &node.0, 10, i64::MAX).is_err());
        let e = crate::error::last_error();
        assert!(
            e.contains(SCORE_ERROR) && e.contains("Must be a non-negative score!"),
            "{e:?}"
        );
        // A score that is not a number.
        let s = spec(0, "'x'", &params(&[]), &[]);
        let node = Blob::tree().u8(NODE_PAINLESS_SCORE).u8(5).raw(&s);
        assert!(run_limit(h, &node.0, 10, i64::MAX).is_err());
        let e = crate::error::last_error();
        assert!(
            e.contains("TypeError\u{1}a score script must return a number, not string"),
            "{e:?}"
        );
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn the_check_refuses_what_v8_or_the_context_cannot_run() {
        let s = spec(0, "return (", &params(&[]), &[]);
        let why = check(&s.0).unwrap().unwrap();
        assert!(why.starts_with("compile error: SyntaxError"), "{why}");
        let s = spec(7, "true", &params(&[]), &[]);
        let why = check(&s.0).unwrap().unwrap();
        assert!(why.contains("js scripts run in the score"), "{why}");
        let mut s = spec(0, "1", &params(&[]), &[]).0;
        // Limits must be positive.
        let n = s.len();
        s[n - 4..].copy_from_slice(&0i32.to_le_bytes());
        assert!(check(&s).is_err());
    }

    #[test]
    fn per_document_runs_convert_what_the_script_returns() {
        let p = params(&[("k", 2.0)]);
        let spec_of = |source: &str, ctx: u8| {
            let blob = spec(ctx, source, &p, &[("likes", 0)]);
            Arc::new(decode_spec(&mut Cursor::new(&blob.0)).unwrap())
        };
        let (tmp, h) = index();
        let dir = lucene_store::FsDirectory::open(tmp.path());
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let seg = &reader.segment_readers()[0];
        for (source, want) in [
            ("[1, true, 'x', null]", "[1.0, true, x, null]"),
            ("params.k", "2.0"),
        ] {
            let spec = spec_of(source, 5);
            let mut s = js_scripts(&spec, seg, Leaf::Ord(0), None).unwrap();
            s.env.doc = 0;
            assert_eq!(s.run_object().unwrap().java_to_string(), want, "{source}");
        }
        let spec = spec_of("({})", 5);
        let mut s = js_scripts(&spec, seg, Leaf::Ord(0), None).unwrap();
        s.env.doc = 0;
        let e = s.run_object().unwrap_err().to_string();
        assert!(e.contains("an aggregation script must return a number, a boolean or an array of them, not object"), "{e}");
        let spec = spec_of("doc['likes'].value > 10000 ? 1 : 0", 3);
        let mut s = js_scripts(&spec, seg, Leaf::Ord(0), None).unwrap();
        s.env.doc = 1;
        assert_eq!(s.run_value().unwrap(), 0.0);
        let spec = spec_of("'s'", 3);
        let mut s = js_scripts(&spec, seg, Leaf::Ord(0), None).unwrap();
        assert!(s.run_value().is_err());
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn params_read_as_their_values() {
        let c = ir::Constant::Map(vec![
            (ir::Constant::Str("a".into()), ir::Constant::Int(1)),
            (
                ir::Constant::Int(2),
                ir::Constant::List(vec![
                    ir::Constant::Null,
                    ir::Constant::Bool(true),
                    ir::Constant::Byte(1),
                    ir::Constant::Short(2),
                    ir::Constant::Char(u16::from(b'c')),
                    ir::Constant::Long(3),
                    ir::Constant::Float(0.5),
                    ir::Constant::Double(1.5),
                    ir::Constant::Pattern("p".into(), 0),
                ]),
            ),
            (ir::Constant::Bool(false), ir::Constant::Null),
            (ir::Constant::Null, ir::Constant::Null),
        ]);
        assert_eq!(
            param(&c),
            Param::Map(vec![
                ("a".into(), Param::Number(1.0)),
                (
                    "2.0".into(),
                    Param::List(vec![
                        Param::Null,
                        Param::Bool(true),
                        Param::Number(1.0),
                        Param::Number(2.0),
                        Param::String("c".into()),
                        Param::Number(3.0),
                        Param::Number(0.5),
                        Param::Number(1.5),
                        Param::String("p".into()),
                    ])
                ),
                ("false".into(), Param::Null),
                ("null".into(), Param::Null),
            ])
        );
    }
}
