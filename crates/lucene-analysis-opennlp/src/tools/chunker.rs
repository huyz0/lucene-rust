//! `opennlp.tools.chunker`: `ChunkerME.chunk` over a loaded
//! `ChunkerModel` -- a beam search (`BeamSize`, default 10) over
//! `DefaultChunkerContextGenerator`'s 41 features of words, tags and the
//! previous chunk tags, an `I-` tag only after a tag of its own phrase
//! (`DefaultChunkerSequenceValidator`).

use std::sync::Arc;

use lucene_analysis::factory::FactoryError;
use lucene_analysis::AnalysisError;

use super::ml::{BeamSearch, MaxentModel, SequenceContext};
use super::model::{BaseModel, Component};
use super::text::JString;

/// A loaded `ChunkerModel`.
#[derive(Debug, Clone)]
pub struct ChunkerModel {
    base: Arc<BaseModel>,
}

impl ChunkerModel {
    /// `new ChunkerModel(InputStream)`.
    pub fn load(bytes: &[u8]) -> Result<ChunkerModel, FactoryError> {
        Ok(ChunkerModel {
            base: Arc::new(BaseModel::load(Component::Chunker, bytes)?),
        })
    }
}

/// `ChunkerME`.
#[derive(Debug, Clone)]
pub struct Chunker {
    model: Arc<MaxentModel>,
    beam: BeamSearch,
}

impl Chunker {
    /// `new ChunkerME(model)`.
    pub fn new(model: &ChunkerModel) -> Result<Chunker, AnalysisError> {
        Ok(Chunker {
            beam: BeamSearch::new(model.base.beam_size(10)?),
            model: model.base.maxent_model(),
        })
    }

    /// `chunk(toks, tags)`: a chunk tag per token.
    pub fn chunk(
        &mut self,
        toks: &[JString],
        tags: &[JString],
    ) -> Result<Vec<JString>, AnalysisError> {
        let mut ctx = ChunkContext {
            toks,
            tags,
            error: std::cell::Cell::new(false),
        };
        let outcomes = self.beam.best_sequence(&self.model, toks.len(), &mut ctx)?;
        if ctx.error.get() {
            return Err(AnalysisError::IllegalState(
                "StringIndexOutOfBoundsException: a chunk tag shorter than its prefix precedes an I- tag"
                    .into(),
            ));
        }
        Ok(outcomes
            .into_iter()
            .map(|o| self.model.outcome(o as usize).to_vec())
            .collect())
    }
}

struct ChunkContext<'a> {
    toks: &'a [JString],
    tags: &'a [JString],
    error: std::cell::Cell<bool>,
}

impl SequenceContext for ChunkContext<'_> {
    // ARITH: positions are below the sentence length (`i + 2` of a token index).
    #[allow(clippy::arithmetic_side_effects)]
    fn context(&mut self, i: usize, prior: &[u32], outcomes: &[JString], f: &mut Vec<JString>) {
        // `key + value`, or `key + edge` past the sentence: one allocation.
        let keyed = |key: &str, value: &[u16]| {
            let mut out = Vec::with_capacity(key.len() + value.len());
            out.extend(key.encode_utf16());
            out.extend_from_slice(value);
            out
        };
        let edged = |key: &str, edge: &str| key.encode_utf16().chain(edge.encode_utf16()).collect();
        let feature = |key: &str, at: Option<usize>, values: &[JString], edge: &str| match at
            .and_then(|a| values.get(a))
        {
            Some(v) => keyed(key, v),
            None => edged(key, edge),
        };
        // The prior outcomes by id: only `p_2`/`p_1` are read, so they are
        // not copied out per call.
        let pred = |key: &str, at: Option<usize>| match at.and_then(|a| prior.get(a)) {
            Some(&o) => keyed(key, &outcomes[o as usize]),
            None => edged(key, "bos"),
        };
        let w_2 = feature("w_2=", i.checked_sub(2), self.toks, "bos");
        let t_2 = feature("t_2=", i.checked_sub(2), self.tags, "bos");
        let p_2 = pred("p_2=", i.checked_sub(2));
        let w_1 = feature("w_1=", i.checked_sub(1), self.toks, "bos");
        let t_1 = feature("t_1=", i.checked_sub(1), self.tags, "bos");
        let p_1 = pred("p_1=", i.checked_sub(1));
        let w0 = feature("w0=", Some(i), self.toks, "");
        let t0 = feature("t0=", Some(i), self.tags, "");
        let w1 = feature("w1=", Some(i + 1), self.toks, "eos");
        let t1 = feature("t1=", Some(i + 1), self.tags, "eos");
        let w2 = feature("w2=", Some(i + 2), self.toks, "eos");
        let t2 = feature("t2=", Some(i + 2), self.tags, "eos");
        let c = |parts: &[&JString]| {
            let mut out = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
            for p in parts {
                out.extend_from_slice(p);
            }
            out
        };
        // The combinations first, over the parts; then the parts are moved
        // into place rather than cloned (the features' order is Java's).
        let [w_1w0, w0w1] = [c(&[&w_1, &w0]), c(&[&w0, &w1])];
        let t_combos = [
            c(&[&t_2, &t_1]),
            c(&[&t_1, &t0]),
            c(&[&t0, &t1]),
            c(&[&t1, &t2]),
            c(&[&t_2, &t_1, &t0]),
            c(&[&t_1, &t0, &t1]),
            c(&[&t0, &t1, &t2]),
        ];
        let p_combos = [
            c(&[&p_2, &p_1]),
            c(&[&p_1, &t_2]),
            c(&[&p_1, &t_1]),
            c(&[&p_1, &t0]),
            c(&[&p_1, &t1]),
            c(&[&p_1, &t2]),
            c(&[&p_1, &t_2, &t_1]),
            c(&[&p_1, &t_1, &t0]),
            c(&[&p_1, &t0, &t1]),
            c(&[&p_1, &t1, &t2]),
            c(&[&p_1, &t_2, &t_1, &t0]),
            c(&[&p_1, &t_1, &t0, &t1]),
            c(&[&p_1, &t0, &t1, &t2]),
            c(&[&p_1, &w_2]),
            c(&[&p_1, &w_1]),
            c(&[&p_1, &w0]),
            c(&[&p_1, &w1]),
            c(&[&p_1, &w2]),
            c(&[&p_1, &w_1, &w0]),
            c(&[&p_1, &w0, &w1]),
        ];
        f.reserve(41);
        f.extend([w_2, w_1, w0, w1, w2, w_1w0, w0w1, t_2, t_1, t0, t1, t2]);
        f.extend(t_combos);
        f.extend([p_2, p_1]);
        f.extend(p_combos);
    }

    fn valid(&self, _i: usize, prior: &[u32], outcome: u32, outcomes: &[JString]) -> bool {
        const I_DASH: [u16; 2] = [b'I' as u16, b'-' as u16];
        const O: [u16; 1] = [b'O' as u16];
        let outcome = &outcomes[outcome as usize];
        let Some(rest) = outcome.strip_prefix(&I_DASH[..]) else {
            return true;
        };
        let Some(&prev) = prior.last() else {
            return false;
        };
        let prev = &outcomes[prev as usize];
        if prev[..] == O {
            return false;
        }
        match prev.get(2..) {
            Some(p) => p == rest,
            None => {
                self.error.set(true);
                false
            }
        }
    }
}
