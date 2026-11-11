//! Picking a field's cluster code quantizer from its own vectors.
//!
//! [`AutoQuantizer`] stands for a choice among quantizers. For each field
//! of at least [`AUTO_MIN_VECTORS`] vectors it writes, at a flush or a
//! merge, [`crate::IvfWriter`] encodes the field's clusters with every
//! candidate and calibrates each ([`crate::calibrate`]) against the same
//! sampled queries. Of the candidates whose rerank reaches the calibration
//! recall, it keeps the one whose search touches the fewest bytes per
//! query: the codes its probes scan, plus the vectors its rerank reads at
//! [`crate::IvfParams::auto_rerank_read_bytes`] each at least (a disk page,
//! by default, for vectors read from disk). When none reaches it, the one
//! with the highest recall. A smaller field gets the first candidate.
//!
//! Every segment records the quantizer it was written with, so segments of
//! one field may differ, and each merge picks again from the merged
//! vectors.

use lucene_util::quantization::ScalarEncoding;
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::rabitq::ExRabitqFactory;
use lucene_vector_quant::{Error, Quantizer, QuantizerFactory, Result, Similarity};

/// The fewest vectors a field needs for [`AutoQuantizer`] to compare its
/// candidates; a smaller one gets the first.
pub const AUTO_MIN_VECTORS: usize = 10_000;

/// A choice among quantizers, made per field when it is written.
#[derive(Debug)]
pub struct AutoQuantizer {
    candidates: Vec<Box<dyn QuantizerFactory>>,
}

impl AutoQuantizer {
    /// A choice among `candidates`, the first being what a field below
    /// [`AUTO_MIN_VECTORS`] vectors gets.
    ///
    /// # Panics
    ///
    /// When `candidates` is empty.
    pub fn new(candidates: Vec<Box<dyn QuantizerFactory>>) -> Self {
        assert!(!candidates.is_empty(), "no candidate quantizers");
        AutoQuantizer { candidates }
    }

    /// Extended RaBitQ at 2 bits first, then BBQ, OSQ at 2 bits and
    /// Extended RaBitQ at 3 and 4 (`docs/benchmarks/ivf.md`, "Picking the
    /// quantizer per segment (auto)", compares them).
    pub fn with_defaults() -> Self {
        Self::new(vec![
            Box::new(ExRabitqFactory::new(2)),
            Box::new(OsqFactory::bbq()),
            Box::new(OsqFactory::new(ScalarEncoding::DibitQueryNibble)),
            Box::new(ExRabitqFactory::new(3)),
            Box::new(ExRabitqFactory::new(4)),
        ])
    }
}

impl QuantizerFactory for AutoQuantizer {
    fn name(&self) -> &str {
        "auto"
    }

    fn train(
        &self,
        dimension: usize,
        similarity: Similarity,
        sample: &[f32],
    ) -> Result<Box<dyn Quantizer>> {
        self.candidates[0].train(dimension, similarity, sample)
    }

    fn read(&self, _: usize, _: Similarity, _: &[u8]) -> Result<Box<dyn Quantizer>> {
        Err(Error::CorruptState(
            "an index records the quantizer picked, never auto".into(),
        ))
    }

    fn candidates(&self) -> Option<&[Box<dyn QuantizerFactory>]> {
        Some(&self.candidates)
    }
}
