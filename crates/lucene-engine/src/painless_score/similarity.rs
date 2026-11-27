//! `ScriptedSimilarity`: a field's similarity scored by a Painless
//! `SimilarityScript`, with the objects such a script reads.

use super::*;

/// OpenSearch's `ScriptedSimilarity`: a field's similarity whose term
/// scores are a Painless `SimilarityScript`, each term's weight an optional
/// `SimilarityWeightScript`.
///
/// `SimScorer.score` returns a `float` and has no way to fail, so a
/// script's exception is kept, the first one, for the search to fail with
/// once it ends ([`Similarity::take_failure`]); the scores after it are `0` and
/// run no script. The plugin raises it as Painless's `ScriptException`,
/// running the script once more in Java on the values the error names.
///
/// The similarity, after its kind byte in the similarity table
/// ([`crate::similarity`]):
///
/// | field | layout |
/// |---|---|
/// | discount overlaps | `u8` |
/// | negative scores | `u8`: [`NegativeScores`] |
/// | description | string: `ScriptedSimilarity.toString()` |
/// | weight script | `u8` `0` none, or `1` and a spec of context `9` |
/// | script | a spec of context `8` |
#[derive(Debug, Clone)]
pub struct ScriptedSimilarity(pub Arc<Scripted>);

#[derive(Debug)]
pub struct Scripted {
    pub script: Spec,
    pub weight: Option<Spec>,
    pub discount_overlaps: bool,
    pub negative: NegativeScores,
    pub description: String,
    /// Whether a script failed: read before every score.
    pub failed: std::sync::atomic::AtomicBool,
    /// The first error, with when it was met ([`next_order`]).
    pub failure: std::sync::Mutex<Option<(u64, Error)>>,
    /// Each term's weight by its statistics: computed once per query, as
    /// Java's `createWeight` does, though each segment makes its scorers.
    pub weights: std::sync::Mutex<HashMap<[u64; 6], f64>>,
    /// This similarity's compiled score script in each thread's [`PROGRAMS`].
    pub id: u64,
}

impl Scripted {
    pub fn fail(&self, e: Error) {
        let mut slot = self.failure.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some((next_order(), e));
        }
        self.failed
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub fn has_failed(&self) -> bool {
        self.failed.load(std::sync::atomic::Ordering::Acquire)
    }
}

impl ScriptedSimilarity {
    /// The similarity after its kind byte, its scripts checked.
    pub fn decode(c: &mut Cursor<'_>) -> std::result::Result<Self, Status> {
        let discount_overlaps = c.u8()? != 0;
        let negative = match c.u8()? {
            0 => NegativeScores::Kept,
            1 => NegativeScores::Fail,
            2 => NegativeScores::Zero,
            other => {
                return Err(invalid(format!(
                    "similarity table: negative scores mode {other} (expected 0 to 2)"
                )))
            }
        };
        let description = string(c)?;
        let weight = match c.u8()? {
            0 => None,
            1 => Some(similarity_spec(c, true)?),
            other => {
                return Err(invalid(format!(
                    "similarity table: weight script flag {other} (expected 0 or 1)"
                )))
            }
        };
        let script = similarity_spec(c, false)?;
        Ok(ScriptedSimilarity(Arc::new(Scripted {
            script,
            weight,
            discount_overlaps,
            negative,
            description,
            failed: std::sync::atomic::AtomicBool::new(false),
            failure: std::sync::Mutex::new(None),
            weights: std::sync::Mutex::new(HashMap::new()),
            id: next_scorer_id(),
        })))
    }
}

/// What a scripted similarity's negative score does. OpenSearch's
/// `SimilarityService` wraps every `scripted` similarity in
/// `NonNegativeScoresSimilarity`, which fails the search on one unless the
/// `opensearch.enforce.positive.scores` system property is `false`, and then
/// scores it `0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegativeScores {
    /// Unwrapped: the score as the script returned it.
    Kept,
    /// `NonNegativeScoresSimilarity` enforcing: an `IllegalArgumentException`.
    Fail,
    /// `NonNegativeScoresSimilarity` not enforcing: `0`.
    Zero,
}

/// A similarity script's spec: of the weight context or the score one, no
/// fields, checked.
pub fn similarity_spec(c: &mut Cursor<'_>, weight: bool) -> std::result::Result<Spec, Status> {
    let spec = decode_spec(c)?;
    if (weight && !spec.similarity_weight) || (!weight && !spec.similarity) {
        return Err(invalid(
            "similarity table: a scripted similarity with another context's script",
        ));
    }
    if !spec.fields.is_empty() {
        return Err(invalid(
            "similarity table: a similarity script reads no fields",
        ));
    }
    spec.check().map_err(|r| {
        invalid(format!(
            "similarity table: painless script not checked: {r}"
        ))
    })?;
    Ok(spec)
}

/// `spec` compiled and specialised to its (empty) params.
pub fn similarity_program(spec: &Spec) -> Result<(Program, lucene_painless::vm::ParamsEnv)> {
    let mut prog = lucene_painless::compile::compile(&spec.script)
        .map_err(|e| Error::IllegalArgument(format!("painless script: {e}")))?;
    let params = spec.params(&Value::Null).map_err(|t| runtime_failure(&t))?;
    specialize(
        &mut prog,
        &Facts {
            params: Some(&params),
            fields: &[],
        },
    );
    Ok((prog, lucene_painless::vm::ParamsEnv { params }))
}

impl lucene_index::similarity::NormSimilarity for ScriptedSimilarity {
    /// `ScriptedSimilarity.computeNorm`: `SmallFloat.intToByte4` of the
    /// length, less the overlaps when they are discounted (a `DOCS` field
    /// too, unlike `Similarity`'s default).
    fn compute_norm(
        &self,
        _field: &str,
        state: &lucene_index::similarity::FieldInvertState,
    ) -> i64 {
        let n = if self.0.discount_overlaps {
            state.length.wrapping_sub(state.num_overlap)
        } else {
            state.length
        };
        i64::from(lucene_util::small_float::int_to_byte4(n as u32) as i8)
    }

    fn discount_overlaps(&self) -> bool {
        self.0.discount_overlaps
    }
}

/// A term's statistics, as `ScriptedSimilarity.scorer` hands its scripts
/// them.
#[derive(Debug, Clone, Copy)]
pub struct SimStats {
    pub boost: f32,
    pub doc_count: i64,
    pub sum_doc_freq: i64,
    pub sum_total_term_freq: i64,
    pub doc_freq: i64,
    pub total_term_freq: i64,
}

impl SimStats {
    /// `query`, `field` and `term`, as the scripts' arguments.
    pub fn objects(self) -> [Value; 3] {
        [
            Value::object(SimQuery(self.boost)),
            Value::object(SimField {
                doc_count: self.doc_count,
                sum_doc_freq: self.sum_doc_freq,
                sum_total_term_freq: self.sum_total_term_freq,
            }),
            Value::object(SimTerm {
                doc_freq: self.doc_freq,
                total_term_freq: self.total_term_freq,
            }),
        ]
    }

    /// The statistics as bits: a [`Scripted::weights`] key.
    pub fn key(self) -> [u64; 6] {
        [
            u64::from(self.boost.to_bits()),
            self.doc_count as u64,
            self.sum_doc_freq as u64,
            self.sum_total_term_freq as u64,
            self.doc_freq as u64,
            self.total_term_freq as u64,
        ]
    }

    /// The statistics where a [`slot_error`] names what the script ran on.
    pub fn slot(self) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}",
            self.boost.to_bits(),
            self.doc_count,
            self.sum_doc_freq,
            self.sum_total_term_freq,
            self.doc_freq,
            self.total_term_freq
        )
    }
}

/// A getter of a similarity script's argument: `name()` with no arguments.
pub fn getter(args: &[Value], value: Option<Value>) -> Option<std::result::Result<Value, Throw>> {
    if args.is_empty() {
        value.map(Ok)
    } else {
        None
    }
}

/// `ScriptedSimilarity.Query`.
#[derive(Debug)]
pub struct SimQuery(pub f32);

impl JObject for SimQuery {
    fn class_name(&self) -> &str {
        "org.opensearch.index.similarity.ScriptedSimilarity$Query"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn call(
        &self,
        _: &mut Vm,
        _: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<std::result::Result<Value, Throw>> {
        getter(args, (name == "getBoost").then_some(Value::Float(self.0)))
    }
    fn field(&self, _: &mut Vm, name: &str) -> Option<std::result::Result<Value, Throw>> {
        (name == "boost").then_some(Ok(Value::Float(self.0)))
    }
}

/// `ScriptedSimilarity.Field`.
#[derive(Debug)]
pub struct SimField {
    pub doc_count: i64,
    pub sum_doc_freq: i64,
    pub sum_total_term_freq: i64,
}

impl SimField {
    pub fn get(&self, property: &str) -> Option<Value> {
        Some(Value::Long(match property {
            "docCount" => self.doc_count,
            "sumDocFreq" => self.sum_doc_freq,
            "sumTotalTermFreq" => self.sum_total_term_freq,
            _ => return None,
        }))
    }
}

impl JObject for SimField {
    fn class_name(&self) -> &str {
        "org.opensearch.index.similarity.ScriptedSimilarity$Field"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn call(
        &self,
        _: &mut Vm,
        _: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<std::result::Result<Value, Throw>> {
        getter(
            args,
            name.strip_prefix("get")
                .and_then(|p| self.get(&decapitalize(p))),
        )
    }
    fn field(&self, _: &mut Vm, name: &str) -> Option<std::result::Result<Value, Throw>> {
        self.get(name).map(Ok)
    }
}

/// `ScriptedSimilarity.Term`.
#[derive(Debug)]
pub struct SimTerm {
    pub doc_freq: i64,
    pub total_term_freq: i64,
}

impl SimTerm {
    pub fn get(&self, property: &str) -> Option<Value> {
        Some(Value::Long(match property {
            "docFreq" => self.doc_freq,
            "totalTermFreq" => self.total_term_freq,
            _ => return None,
        }))
    }
}

impl JObject for SimTerm {
    fn class_name(&self) -> &str {
        "org.opensearch.index.similarity.ScriptedSimilarity$Term"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn call(
        &self,
        _: &mut Vm,
        _: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<std::result::Result<Value, Throw>> {
        getter(
            args,
            name.strip_prefix("get")
                .and_then(|p| self.get(&decapitalize(p))),
        )
    }
    fn field(&self, _: &mut Vm, name: &str) -> Option<std::result::Result<Value, Throw>> {
        self.get(name).map(Ok)
    }
}

/// A getter's property: `DocCount` as `docCount`.
pub fn decapitalize(p: &str) -> String {
    let mut c = p.chars();
    match c.next() {
        Some(f) => f.to_lowercase().chain(c).collect(),
        None => String::new(),
    }
}

/// `ScriptedSimilarity.Doc`: the document a term's scorer is on, set before
/// each run.
#[derive(Debug, Default)]
pub struct SimDoc {
    pub freq: std::cell::Cell<f32>,
    pub norm: std::cell::Cell<i64>,
}

impl SimDoc {
    /// `getLength()`: `SmallFloat.byte4ToInt((byte) norm)`.
    pub fn length(&self) -> i32 {
        lucene_util::small_float::byte4_to_int(self.norm.get() as u8) as i32
    }

    pub fn get(&self, property: &str) -> Option<Value> {
        match property {
            "length" => Some(Value::Int(self.length())),
            "freq" => Some(Value::Float(self.freq.get())),
            _ => None,
        }
    }
}

impl JObject for SimDoc {
    fn class_name(&self) -> &str {
        "org.opensearch.index.similarity.ScriptedSimilarity$Doc"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn call(
        &self,
        _: &mut Vm,
        _: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<std::result::Result<Value, Throw>> {
        getter(
            args,
            name.strip_prefix("get")
                .and_then(|p| self.get(&decapitalize(p))),
        )
    }
    fn field(&self, _: &mut Vm, name: &str) -> Option<std::result::Result<Value, Throw>> {
        self.get(name).map(Ok)
    }
}

impl Similarity for ScriptedSimilarity {
    /// `ScriptedSimilarity.scorer`: per term, its weight from the weight
    /// script (`1` without one), then a scorer running the script; the sum
    /// of the terms' scores, in `double`, for more than one.
    fn scorer(
        &self,
        _field: &str,
        boost: f32,
        collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        let doc_count = if collection.doc_count == -1 {
            collection.max_doc
        } else {
            collection.doc_count
        };
        let mut scorers: Vec<ScriptedScorer> = terms
            .iter()
            .map(|t| {
                let stats = SimStats {
                    boost,
                    doc_count,
                    sum_doc_freq: collection.sum_doc_freq,
                    sum_total_term_freq: collection.sum_total_term_freq,
                    doc_freq: t.doc_freq,
                    total_term_freq: t.total_term_freq,
                };
                ScriptedScorer {
                    weight: self.weight(stats),
                    stats,
                    shared: self.0.clone(),
                    id: next_scorer_id(),
                }
            })
            .collect();
        let scorer: Arc<dyn SimScorer> = if scorers.len() == 1 {
            Arc::new(scorers.pop().expect("one scorer"))
        } else {
            Arc::new(SummedScorer(scorers))
        };
        match self.0.negative {
            NegativeScores::Kept => scorer,
            _ => Arc::new(NonNegativeScorer {
                inner: scorer,
                shared: self.0.clone(),
            }),
        }
    }

    fn simple_name(&self) -> &str {
        "ScriptedSimilarity"
    }

    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        Some(Arc::new(self.clone()))
    }

    fn take_failure(&self) -> Option<Error> {
        if !self.0.has_failed() {
            return None;
        }
        self.0
            .failure
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
            .map(|(_, e)| e)
    }

    fn failure_order(&self) -> Option<u64> {
        if !self.0.has_failed() {
            return None;
        }
        self.0
            .failure
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|&(o, _)| o)
    }
}

impl ScriptedSimilarity {
    /// `computeWeight`: the weight script's value, or `1`.
    pub fn weight(&self, stats: SimStats) -> f64 {
        let Some(spec) = &self.0.weight else {
            return 1.0;
        };
        if self.0.has_failed() {
            return 0.0;
        }
        let key = stats.key();
        let memo = || self.0.weights.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(&w) = memo().get(&key) {
            return w;
        }
        let run = || -> Result<f64> {
            let (prog, mut env) = similarity_program(spec)?;
            let prog = Rc::new(prog);
            let mut vm = Vm::new(prog.clone(), &mut env);
            vm.bindings = vec![None; prog.binding_sites]; // alloc-ok: the compiled program's own count
            match vm.execute(&stats.objects()) {
                Ok(v) => script_double(&v),
                Err(t) => Err(slot_error(spec.script_id, &stats.slot(), vm.loc, &t)),
            }
        };
        match run() {
            Ok(w) => {
                memo().insert(key, w);
                w
            }
            Err(e) => {
                self.0.fail(e);
                0.0
            }
        }
    }
}

/// A similarity script's `double` result.
pub fn script_double(v: &Value) -> Result<f64> {
    match v {
        Value::Double(d) => Ok(*d),
        other => Err(Error::IllegalArgument(format!(
            "painless runtime: a similarity script returned {}",
            other.java_to_string()
        ))),
    }
}

pub fn next_scorer_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// When a scripted similarity failed, in one order across every one of them.
pub fn next_order() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// One term's scorer: `ScriptedSimilarity`'s anonymous `SimScorer`.
#[derive(Debug)]
pub struct ScriptedScorer {
    pub shared: Arc<Scripted>,
    pub weight: f64,
    pub stats: SimStats,
    /// This scorer's state in each thread's [`SCORERS`].
    pub id: u64,
}

/// A [`ScriptedScorer`]'s compiled script and arguments on one thread:
/// the runtime's values are not `Send`, so each thread that scores makes
/// its own.
pub struct ScorerState {
    pub prog: Rc<Program>,
    pub env: lucene_painless::vm::ParamsEnv,
    pub bindings: Vec<Option<Value>>,
    pub pool: Vec<Vec<Value>>,
    pub doc: Rc<SimDoc>,
    pub args: [Value; 5],
}

/// At most this many scorers' states per thread; past it the thread starts
/// over.
pub const MAX_SCORER_STATES: usize = 256;

thread_local! {
    static SCORERS: std::cell::RefCell<HashMap<u64, ScorerState>> =
        std::cell::RefCell::new(HashMap::new());
    /// Each scripted similarity's compiled score script, by its id: compiled
    /// once per thread, not per scorer.
    static PROGRAMS: std::cell::RefCell<HashMap<u64, Rc<Program>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Makes room for one more entry in a per-thread `map` at most
/// [`MAX_SCORER_STATES`] long: one entry goes, not all of them.
pub fn make_room<V>(map: &mut HashMap<u64, V>) {
    if map.len() >= MAX_SCORER_STATES {
        if let Some(&k) = map.keys().next() {
            map.remove(&k);
        }
    }
}

impl ScriptedScorer {
    pub fn state(&self) -> Result<ScorerState> {
        let id = self.shared.id;
        let cached = PROGRAMS.with(|p| p.borrow().get(&id).cloned());
        let (prog, env) = match cached {
            // Its own params: a script may change them.
            Some(prog) => {
                let params = self
                    .shared
                    .script
                    .params(&Value::Null)
                    .map_err(|t| runtime_failure(&t))?;
                (prog, lucene_painless::vm::ParamsEnv { params })
            }
            None => {
                let (prog, env) = similarity_program(&self.shared.script)?;
                let prog = Rc::new(prog);
                PROGRAMS.with(|p| {
                    let mut p = p.borrow_mut();
                    make_room(&mut p);
                    p.insert(id, prog.clone());
                });
                (prog, env)
            }
        };
        let doc = Rc::new(SimDoc::default());
        let [query, field, term] = self.stats.objects();
        Ok(ScorerState {
            bindings: vec![None; prog.binding_sites], // alloc-ok: the compiled program's own count
            prog,
            env,
            pool: Vec::new(),
            args: [
                Value::Double(self.weight),
                query,
                field,
                term,
                Value::Object(doc.clone()),
            ],
            doc,
        })
    }

    pub fn run(&self, freq: f32, norm: i64) -> Result<f32> {
        SCORERS.with(|states| {
            let mut states = states.borrow_mut();
            if !states.contains_key(&self.id) {
                make_room(&mut states);
                states.insert(self.id, self.state()?);
            }
            let st = states.get_mut(&self.id).expect("inserted above");
            st.doc.freq.set(freq);
            st.doc.norm.set(norm);
            let mut vm = Vm::new(st.prog.clone(), &mut st.env);
            vm.bindings = std::mem::take(&mut st.bindings);
            vm.pool = std::mem::take(&mut st.pool);
            let r = vm.execute(&st.args);
            let loc = vm.loc;
            st.bindings = std::mem::take(&mut vm.bindings);
            st.pool = std::mem::take(&mut vm.pool);
            match r {
                Ok(v) => Ok(script_double(&v)? as f32),
                Err(t) => Err(slot_error(
                    self.shared.script.script_id,
                    &format!(
                        "{}:{}:{}:{}",
                        self.weight.to_bits(),
                        self.stats.slot(),
                        freq.to_bits(),
                        norm
                    ),
                    loc,
                    &t,
                )),
            }
        })
    }
}

impl SimScorer for ScriptedScorer {
    fn score(&self, freq: f32, norm: i64) -> f32 {
        if self.shared.has_failed() {
            return 0.0;
        }
        self.run(freq, norm).unwrap_or_else(|e| {
            self.shared.fail(e);
            0.0
        })
    }

    /// `ScriptedSimilarity`'s explanation: the score, then each value the
    /// script read.
    fn explain(&self, freq: Explanation, norm: i64) -> Explanation {
        let score = self.score(freq.value, norm);
        let s = self.stats;
        let length = lucene_util::small_float::byte4_to_int(norm as u8) as i64;
        Explanation::match_(
            score,
            format!("score from {} computed from:", self.shared.description),
        )
        .with_details(vec![
            Explanation::match_(self.weight as f32, "weight"),
            Explanation::match_(s.boost, "query.boost"),
            Explanation::match_long(s.doc_count, "field.docCount"),
            Explanation::match_long(s.sum_doc_freq, "field.sumDocFreq"),
            Explanation::match_long(s.sum_total_term_freq, "field.sumTotalTermFreq"),
            Explanation::match_long(s.doc_freq, "term.docFreq"),
            Explanation::match_long(s.total_term_freq, "term.totalTermFreq"),
            Explanation::match_value_of(&freq, "doc.freq").with_details(freq.details.clone()),
            Explanation::match_long(length, "doc.length"),
        ])
    }
}

/// More than one term's scorers: their scores summed in `double`.
#[derive(Debug)]
pub struct SummedScorer(pub Vec<ScriptedScorer>);

impl SimScorer for SummedScorer {
    fn score(&self, freq: f32, norm: i64) -> f32 {
        // From `0.0` as Java's `double` starts: `Sum` starts from `-0.0`.
        self.0
            .iter()
            .fold(0.0f64, |sum, s| sum + f64::from(s.score(freq, norm))) as f32
    }

    fn explain(&self, freq: Explanation, norm: i64) -> Explanation {
        let details = self
            .0
            .iter()
            .map(|s| s.explain(freq.clone(), norm))
            .collect();
        Explanation::match_(self.score(freq.value, norm), "Sum of:").with_details(details)
    }
}

/// `NonNegativeScoresSimilarity`'s scorer around a scripted similarity's.
pub struct NonNegativeScorer {
    pub inner: Arc<dyn SimScorer>,
    pub shared: Arc<Scripted>,
}

impl std::fmt::Debug for NonNegativeScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NonNegativeScorer")
            .field("negative", &self.shared.negative)
            .finish_non_exhaustive()
    }
}

impl SimScorer for NonNegativeScorer {
    /// A negative score fails the search, its explanation the message, or
    /// is `0` when not enforced; `NaN` and `-0.0` pass.
    fn score(&self, freq: f32, norm: i64) -> f32 {
        let score = self.inner.score(freq, norm);
        if score >= 0.0 || score.is_nan() {
            return score;
        }
        if self.shared.negative == NegativeScores::Fail {
            let why = self
                .inner
                .explain(Explanation::match_(freq, "term frequency"), norm);
            self.shared.fail(score_error(format!(
                "Similarities must not produce negative scores, but got:\n{why}"
            )));
        }
        0.0
    }

    /// A negative explanation as the `max of` it and `0`.
    fn explain(&self, freq: Explanation, norm: i64) -> Explanation {
        let e = self.inner.explain(freq, norm);
        if e.matched && e.value < 0.0 {
            Explanation::match_(0.0, "max of:")
                .with_details(vec![e, Explanation::match_(0.0, "Minimum allowed score")])
        } else {
            e
        }
    }
}

impl Drop for Scripted {
    fn drop(&mut self) {
        // The dropping thread's program; another thread's goes when that
        // thread passes MAX_SCORER_STATES.
        let _ = PROGRAMS.try_with(|p| p.borrow_mut().remove(&self.id));
    }
}

impl Drop for ScriptedScorer {
    fn drop(&mut self) {
        // The dropping thread's state; another thread's goes when that
        // thread passes MAX_SCORER_STATES.
        let _ = SCORERS.try_with(|s| s.borrow_mut().remove(&self.id));
    }
}
