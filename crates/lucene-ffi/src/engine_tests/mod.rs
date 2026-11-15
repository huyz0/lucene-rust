//! Engine tests that drive the exported entry points: they open a reader
//! through [`crate::jvm_reader`], so they live beside the boundary rather
//! than in `lucene-engine`.

// Tests index and count freely: the gate is about values read off disk.
#![allow(clippy::arithmetic_side_effects)]

mod function_score;
mod max_sim;
pub(crate) mod nodes;
mod painless_score;
#[cfg(feature = "js")]
mod painless_score_js;
mod pattern_phrase;
mod queries;
