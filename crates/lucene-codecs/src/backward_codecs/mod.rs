//! Readers for the retired formats of `lucene-backward-codecs` that an index
//! Lucene 10.5.0 opens can still hold: everything Lucene 9.0 through 10.3
//! wrote whose wire format the current readers do not already accept
//! (M8, `docs/milestones/m8-backward-codecs.md`).
//!
//! Most retired components need no module here, because their current
//! reader was widened to the older versions instead -- the wire format did
//! not change, only the accepted range or one branch did:
//!
//! - `.si`: `Lucene90SegmentInfoFormat` is `Lucene99SegmentInfoFormat` minus
//!   the `hasBlocks` byte (`lucene_index::segment_info::parse_for_codec`,
//!   dispatched by codec name through [`crate::codecs`]).
//! - `.fnm`: `Lucene90FieldInfosFormat` ([`crate::field_infos::parse`],
//!   dispatched on the header's codec name).
//! - `.kdm`/`.kdi`/`.kdd`: `Lucene90PointsFormat` version 0, i.e. BKD
//!   version 9 ([`crate::points`]).
//! - `.pos`/`.pay` of every retired postings format
//!   ([`crate::postings::read_positions`] at a 128-value block).
//!
//! What is here is what did change: the postings `.doc` framing and bit
//! packing ([`postings`], [`for_util`]) and the FST terms index of
//! `Lucene90BlockTreeTermsReader` ([`blocktree`]), and the retired
//! `Lucene90`..`Lucene95` HNSW vector formats' metadata, graphs and searches
//! ([`hnsw_vectors`]), whose vectors the current flat reader serves. The
//! opt-in quantized formats (`Lucene99` scalar, `Lucene102` binary) are not
//! ported yet.

pub mod blocktree;
pub mod for_util;
pub mod hnsw_vectors;
pub mod postings;
