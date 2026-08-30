//! `ffi_open_vectors`/`ffi_knn_float_vector_search`/
//! `ffi_knn_byte_vector_search`: KNN (approximate nearest neighbour) search
//! over one segment's vector field.
//!
//! **A thin wrapper, like every other entry point in this crate.** The query
//! itself is [`lucene_search::vector_query`] -- the port of Java's
//! `AbstractKnnVectorQuery`/`KnnFloatVectorQuery`/`KnnByteVectorQuery`, which
//! in turn sits on `lucene_codecs::vectors`/`hnsw`/`hnsw_vectors` (the
//! `Lucene99FlatVectorsFormat` store and the HNSW graph, verified arc-for-arc
//! against real Lucene by the M2 sweep batch `c5-vectors`). Nothing about
//! *which* collector size, *which* accept set, or graph-walk-versus-exact
//! lives here: this module owns handle validation, argument decoding and the
//! results handle, exactly as `points_query.rs` wraps
//! `lucene_search::points_query` and `sort.rs` wraps
//! `lucene_search::doc_value_query`.
//!
//! (It did not always. The batch that first exposed KNN over this ABI had to
//! keep the query policy here, because `lucene-search` had no vector module
//! and belonged to a concurrently-running batch; `c16-knn-query` moved it
//! down. The exported symbols and their behaviour are unchanged, which
//! `knn_search_reproduces_lucene_knn_vector_query_results` -- the differential
//! test that runs every query the fixture records *through the exported C
//! symbols* and matches real Lucene doc-for-doc -- is what proves.)
//!
//! Mirrors `KnnFloatVectorQuery`/`KnnByteVectorQuery`: a field, a target
//! vector, and `k`. Results come back as `(doc_id, score)` through the
//! existing [`crate::registry::ScoredResultsHandle`] and
//! `results_scored.rs`'s accessor trio -- a KNN hit *is* a scored doc
//! (Java's `KnnFloatVectorQuery` produces a `TopDocs` like any other query),
//! so no new results shape was invented.
//!
//! **Why a separate handle from [`crate::segment::ffi_open_segment`]**: that
//! function requires `.tim`/`.tip`/`.tmd`, because everything built on it
//! needs a term dictionary. A vector field needs none -- real Lucene's
//! `KnnVectorsReader` is a per-segment reader entirely independent of
//! `FieldsProducer`, and a segment can carry vectors and no postings at all
//! (this crate's own `fixtures/data/vectors_index` is exactly that). Folding
//! vectors into `SegmentHandle` would therefore have made a vectors-only
//! segment unopenable. [`ffi_open_vectors`] takes the `.fnm` it needs for
//! the field-name -> field-number mapping and nothing else.
//!
//! **Deletions** are honoured: [`ffi_vectors_set_live_docs`] attaches this
//! segment's `.liv` exactly as [`crate::segment::ffi_segment_set_live_docs`]
//! does for a `SegmentHandle` (both go through the same
//! [`crate::segment::decode_live_docs`]), and the search layer hands the
//! bitset to the graph walk as Java's `acceptOrds`, so a deleted node never
//! enters the collector at all.

use std::os::raw::c_char;

use lucene_codecs::field_infos::{self, FieldInfos, VectorSimilarityFunction};
use lucene_codecs::hnsw_vectors::HnswVectorsReader;
use lucene_codecs::vectors::FlatVectorsReader;
use lucene_search::vector_query;

use crate::directory::read_whole_file;
use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::str_from_raw;
use crate::registry::{lock_recovering, read_recovering, scored_results, vectors, VectorsHandle};
use crate::segment::decode_live_docs;
use lucene_util::fixed_bit_set::FixedBitSet;

/// "Use the similarity function the field itself was written with" -- the
/// value [`ffi_knn_float_vector_search`]'s `similarity` parameter takes when
/// the caller has no opinion, which is what real Lucene always does
/// (`KnnFloatVectorQuery` has no similarity parameter; `FieldInfo` owns it).
pub const SIMILARITY_FROM_FIELD: i32 = -1;

fn map_vectors_error(what: &str, e: lucene_codecs::vectors::Error) -> FfiStatus {
    set_last_error(format!("{what}: {e}"));
    FfiStatus::Decode
}

/// Opens one segment's vector files behind a new handle.
///
/// - `dir_handle`: an [`crate::directory::ffi_open_directory`] handle.
/// - `fnm_name`/`fnm_name_len`: the segment's `.fnm`, required -- the only
///   place a field *name* (all a caller has over the C ABI) maps to the
///   field *number* the vector formats key everything by.
/// - `vemf_name`/`vec_name`: the flat vector store's metadata and data files
///   (`Lucene99FlatVectorsFormat`), both required. Together they are enough
///   to serve exact, exhaustive KNN.
/// - `vem_name`/`vex_name`: the HNSW graph's metadata and index files
///   (`Lucene99HnswVectorsFormat`), or a **null** pointer (any `len`) for
///   either to open neither. Opened together or not at all, the same
///   convention `ffi_open_segment` uses for `.nvm`/`.nvd` and `.dvm`/`.dvd`.
///   Without them every search is the exhaustive scan Java also falls back
///   to for a field written with no graph -- correct, just `O(size)`.
/// - `segment_id`: the segment's 16-byte ID (`SegmentInfo.getId()`).
/// - `vector_suffix`: the per-field codec suffix in the vector files' index
///   headers (e.g. `"Lucene99HnswVectorsFormat_0"`). Vector files *do* carry
///   one, unlike `.nvm`/`.nvd`; the `.fnm` is always validated against the
///   empty suffix, as everywhere else in this crate.
/// - `max_doc`: the segment's `SegmentInfo.maxDoc()`, used to size the
///   `.liv` bitset [`ffi_vectors_set_live_docs`] may later attach.
///
/// # Safety
/// Every `(*const c_char, len)` pair must be valid for reads of `len` bytes
/// (or null where allowed above); `segment_id` must be valid for 16 bytes;
/// `out_handle` must be valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_open_vectors(
    dir_handle: u64,
    fnm_name: *const c_char,
    fnm_name_len: usize,
    vemf_name: *const c_char,
    vemf_name_len: usize,
    vec_name: *const c_char,
    vec_name_len: usize,
    vem_name: *const c_char,
    vem_name_len: usize,
    vex_name: *const c_char,
    vex_name_len: usize,
    segment_id: *const u8,
    vector_suffix: *const c_char,
    vector_suffix_len: usize,
    max_doc: i32,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() || segment_id.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // Same reasoning as `ffi_open_segment`: `max_doc` is a count, and a
        // negative one would silently empty every later `0..max_doc` range.
        if max_doc < 0 {
            set_last_error(format!(
                "ffi_open_vectors: max_doc {max_doc} is negative (SegmentInfo.maxDoc() is a count)"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        // SAFETY: caller contract guarantees each name pointer is valid for
        // its paired length.
        let (fnm, vemf, vec_file, suffix) = unsafe {
            (
                str_from_raw(fnm_name, fnm_name_len)?,
                str_from_raw(vemf_name, vemf_name_len)?,
                str_from_raw(vec_name, vec_name_len)?,
                str_from_raw(vector_suffix, vector_suffix_len)?,
            )
        };
        let mut id = [0u8; 16];
        // SAFETY: caller contract guarantees `segment_id` is valid for 16 bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(segment_id, id.as_mut_ptr(), 16);
        }

        let fnm_bytes = read_whole_file(dir_handle, fnm)?;
        let field_infos: FieldInfos = field_infos::parse(&fnm_bytes, &id, "").map_err(|e| {
            set_last_error(format!("parsing .fnm: {e}"));
            FfiStatus::Decode
        })?;

        let vemf_bytes = read_whole_file(dir_handle, vemf)?;
        let vec_bytes = read_whole_file(dir_handle, vec_file)?;
        // Validated once here (then discarded, like `ffi_open_segment` does
        // for `.doc`/points) so a corrupt file is a `Decode` status at open
        // time rather than at the first search.
        FlatVectorsReader::open(&vemf_bytes, &vec_bytes, &id, suffix)
            .map_err(|e| map_vectors_error("opening flat vectors (.vemf/.vec)", e))?;

        // Both null means "no graph, search exhaustively" -- a real shape (a
        // field below `HNSW_GRAPH_THRESHOLD` has no graph at all). Exactly one
        // null is not a shape any correct caller produces, and silently
        // downgrading it would leave that caller with `O(size)` searches and
        // no signal at all. Deliberately stricter than the `.nvm`/`.nvd` and
        // `.dvm`/`.dvd` pairs `ffi_open_segment` established, which do
        // silently ignore a half-specified pair; this entry point is new, so
        // there is no caller to keep compatible with the weaker contract.
        if vem_name.is_null() != vex_name.is_null() {
            set_last_error(concat!(
                "ffi_open_vectors: the HNSW graph's .vem and .vex must be given together ",
                "or not at all (a half-specified pair would silently fall back to an ",
                "exhaustive scan)"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        let hnsw = if vem_name.is_null() {
            None
        } else {
            // SAFETY: caller contract guarantees both are valid for their
            // paired lengths.
            let (vem, vex) = unsafe {
                (
                    str_from_raw(vem_name, vem_name_len)?,
                    str_from_raw(vex_name, vex_name_len)?,
                )
            };
            let vem_bytes = read_whole_file(dir_handle, vem)?;
            let vex_bytes = read_whole_file(dir_handle, vex)?;
            HnswVectorsReader::open(&vem_bytes, &vex_bytes, &id, suffix)
                .map_err(|e| map_vectors_error("opening HNSW graph (.vem/.vex)", e))?;
            Some((vem_bytes, vex_bytes))
        };

        let handle = vectors_registry_insert(VectorsHandle {
            field_infos,
            vemf: vemf_bytes,
            vec: vec_bytes,
            hnsw,
            segment_id: id,
            suffix: suffix.to_string(),
            max_doc,
            live_docs: None,
        })?;
        // SAFETY: caller contract guarantees `out_handle` is valid for one write.
        unsafe {
            *out_handle = handle;
        }
        Ok(())
    })
}

fn vectors_registry_insert(handle: VectorsHandle) -> Result<u64, FfiStatus> {
    lock_recovering(vectors()).insert_checked(handle)
}

/// Attaches (or clears) this vector reader's `.liv` live-docs bitset, so
/// every later KNN search skips deleted documents. Exactly
/// [`crate::segment::ffi_segment_set_live_docs`]'s contract and validation
/// (both call [`decode_live_docs`]) -- see that function's doc comment for
/// `liv_name`/`del_gen`/`del_count`, including that a **null** `liv_name`
/// clears the bitset.
///
/// # Safety
/// `liv_name` must be valid for reads of `liv_name_len` bytes, or null.
#[no_mangle]
pub unsafe extern "C" fn ffi_vectors_set_live_docs(
    vectors_handle: u64,
    dir_handle: u64,
    liv_name: *const c_char,
    liv_name_len: usize,
    del_gen: i64,
    del_count: i32,
) -> i32 {
    guard(|| {
        // Read the two scalars under a scoped read guard, drop it, then do
        // the I/O and take the write guard -- `std`'s `RwLock` neither
        // upgrades nor re-enters, so holding both at once would deadlock.
        // Same shape (and same reason) as `ffi_segment_set_live_docs`.
        let (segment_id, max_doc) = {
            let registry = read_recovering(vectors());
            let handle = registry.get(vectors_handle).ok_or_else(|| {
                set_last_error("ffi_vectors_set_live_docs: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })?;
            (handle.segment_id, handle.max_doc)
        };
        // SAFETY: forwarded from this function's own caller contract.
        let parsed = unsafe {
            decode_live_docs(
                "ffi_vectors_set_live_docs",
                dir_handle,
                liv_name,
                liv_name_len,
                del_gen,
                del_count,
                &segment_id,
                max_doc,
            )?
        };

        let mut registry = lock_recovering(vectors());
        let handle = registry.get_mut(vectors_handle).ok_or_else(|| {
            set_last_error("ffi_vectors_set_live_docs: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        handle.live_docs = parsed;
        Ok(())
    })
}

/// Decodes the `similarity` argument every KNN entry point takes into what
/// [`lucene_search::vector_query`] wants -- argument decoding, which is this
/// boundary's job, over policy, which is not.
///
/// Real Lucene has no such parameter: `FieldInfo` owns the similarity and
/// `KnnFloatVectorQuery` uses it unconditionally. It exists here as a
/// **cross-check**, not an override -- see
/// [`lucene_search::vector_query::KnnFloatVectorQuery::similarity`] for why
/// (the short version: an HNSW graph's arcs encode the build-time
/// similarity's neighbourhood, so walking it under another one silently
/// degrades recall with no error at all). [`SIMILARITY_FROM_FIELD`] (`-1`)
/// means "the field's own"; anything else must be one of the four ordinals
/// *and* must equal the field's, which `lucene-search` checks because only it
/// has the `.vemf` open.
fn decode_similarity(requested: i32) -> Result<Option<VectorSimilarityFunction>, FfiStatus> {
    if requested == SIMILARITY_FROM_FIELD {
        return Ok(None);
    }
    match vector_query::similarity_from_ordinal(requested) {
        Some(s) => Ok(Some(s)),
        None => {
            set_last_error(format!(
                "similarity {requested} is not a VectorSimilarityFunction ordinal (0=EUCLIDEAN, \
                 1=DOT_PRODUCT, 2=COSINE, 3=MAXIMUM_INNER_PRODUCT, or -1 for the field's own)"
            ));
            Err(FfiStatus::InvalidArgument)
        }
    }
}

/// Reopens this handle's readers as the [`vector_query::VectorsInput`] the
/// search layer takes.
///
/// The readers are views over the byte buffers the handle already owns, so
/// "reopening" is parsing the `.vemf`/`.vem` metadata again (a few hundred
/// bytes), not re-reading the index. Doing it per search rather than storing
/// the readers on the handle is what keeps [`crate::registry::VectorsHandle`]
/// free of self-referential borrows -- the same trade every other reader
/// handle in this crate makes.
fn vectors_input<'h>(
    handle: &'h VectorsHandle,
) -> Result<vector_query::VectorsInput<'h>, FfiStatus> {
    let flat = FlatVectorsReader::open(
        &handle.vemf,
        &handle.vec,
        &handle.segment_id,
        &handle.suffix,
    )
    .map_err(|e| map_vectors_error("reopening flat vectors", e))?;
    let hnsw = match &handle.hnsw {
        None => None,
        Some((vem, vex)) => Some(
            HnswVectorsReader::open(vem, vex, &handle.segment_id, &handle.suffix)
                .map_err(|e| map_vectors_error("reopening HNSW graph", e))?,
        ),
    };
    Ok(vector_query::VectorsInput {
        flat,
        hnsw,
        field_infos: &handle.field_infos,
        live_docs: handle.live_docs.as_ref(),
        // Unfiltered. The filtered entry points
        // ([`ffi_knn_float_vector_search_filtered`] and its byte sibling)
        // resolve their clause array through [`resolve_filter`] and set this
        // field on the returned value, because the bitset they build has to
        // outlive this call and cannot be borrowed from inside it.
        filter: None,
        max_doc: handle.max_doc,
    })
}

/// `lucene_search::Error` -> this crate's status codes.
///
/// The split is the point of [`lucene_search::Error::InvalidKnnQuery`]: a
/// caller mistake (an unknown field, a wrong-length query vector, a `k` under
/// one) is [`FfiStatus::InvalidArgument`] with Java's own message, and only a
/// genuine decode failure is [`FfiStatus::Decode`] -- which a JNI caller
/// reads as "this index is corrupt" and may fail a shard over.
fn map_knn_error(e: lucene_search::Error) -> FfiStatus {
    match e {
        lucene_search::Error::InvalidKnnQuery(msg) => {
            set_last_error(msg);
            FfiStatus::InvalidArgument
        }
        lucene_search::Error::Vectors(inner) => {
            set_last_error(format!("KNN search: {inner}"));
            FfiStatus::Decode
        }
        other => crate::query::map_search_error(other),
    }
}

/// Runs a `KnnFloatVectorQuery`-equivalent search against `vectors_handle`.
///
/// - `field_name`/`field_name_len`: the FLOAT32 vector field's name.
/// - `query`/`query_len`: the target vector, `query_len` `f32`s. Must be
///   exactly the field's dimension.
/// - `k`: how many hits to return, `>= 1` (Java's own bound).
/// - `ef_search`: the beam width (OpenSearch's `num_candidates`). `0` means
///   "use `k`", which is exactly `KnnFloatVectorQuery`'s behaviour; a larger
///   value trades work for recall. Clamped to the field's vector count.
/// - `similarity`: [`SIMILARITY_FROM_FIELD`] (`-1`) to use the field's own,
///   or a `VectorSimilarityFunction` ordinal that must match it -- see
///   [`decode_similarity`] for why this is a cross-check and not an override.
/// - `visited_limit`: the collector's `visitLimit`; `0` means unlimited,
///   which is Java's default.
///
/// Writes a [`crate::registry::ScoredResultsHandle`] to
/// `*out_scored_results_handle`, read back with
/// `ffi_scored_results_len`/`ffi_scored_results_copy` and released with
/// `ffi_close_scored_results` like any other scored search's results.
///
/// # Safety
/// `field_name` must be valid for reads of `field_name_len` bytes; `query`
/// must be valid for reads of `query_len` `f32`s (or null iff
/// `query_len == 0`); `out_scored_results_handle` must be valid for one
/// `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_knn_float_vector_search(
    vectors_handle: u64,
    field_name: *const c_char,
    field_name_len: usize,
    query: *const f32,
    query_len: usize,
    k: usize,
    ef_search: usize,
    similarity: i32,
    visited_limit: u64,
    out_scored_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_scored_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        if query.is_null() && query_len != 0 {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `field_name` is valid for
        // `field_name_len` bytes.
        let field = unsafe { str_from_raw(field_name, field_name_len)? };
        // SAFETY: caller contract guarantees `query` is valid for `query_len`
        // `f32`s; the empty slice is used for a null pointer with length 0.
        let target: &[f32] = if query_len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(query, query_len) }
        };

        let similarity = decode_similarity(similarity)?;

        let registry = read_recovering(vectors());
        let handle = registry.get(vectors_handle).ok_or_else(|| {
            set_last_error("ffi_knn_float_vector_search: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;

        let input = vectors_input(handle)?;
        let query = vector_query::KnnFloatVectorQuery {
            field: field.to_string(),
            target: target.to_vec(),
            k,
            ef_search,
            visited_limit,
            similarity,
        };
        let hits =
            vector_query::search_knn_float_vector_query(&input, &query).map_err(map_knn_error)?;
        drop(registry);

        let handle =
            scored_results().insert_checked(crate::registry::ScoredResultsHandle { hits })?;
        // SAFETY: caller contract guarantees the out pointer is valid for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// `KnnByteVectorQuery`'s equivalent: identical to
/// [`ffi_knn_float_vector_search`] except that the field must be BYTE-encoded
/// and `query`/`query_len` is a byte vector.
///
/// The bytes are Java's *signed* `byte[]`, reinterpreted exactly as
/// `lucene_codecs::vectors`'s byte kernels do (`as i8 as i32`, Java's own
/// sign extension) -- a caller passes the same bytes it would hand
/// `KnnByteVectorQuery`, with no re-encoding. `*const u8` rather than
/// `*const i8` because `c_char`'s signedness is target-dependent and this
/// crate never `as`-casts across it (see the `ffi-safety` skill).
///
/// # Safety
/// `field_name` must be valid for reads of `field_name_len` bytes; `query`
/// must be valid for reads of `query_len` bytes (or null iff
/// `query_len == 0`); `out_scored_results_handle` must be valid for one
/// `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_knn_byte_vector_search(
    vectors_handle: u64,
    field_name: *const c_char,
    field_name_len: usize,
    query: *const u8,
    query_len: usize,
    k: usize,
    ef_search: usize,
    similarity: i32,
    visited_limit: u64,
    out_scored_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_scored_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `field_name`/`query` are valid
        // for their paired lengths.
        let (field, target) = unsafe {
            (
                str_from_raw(field_name, field_name_len)?,
                crate::raw::bytes_from_raw(query, query_len)?,
            )
        };

        let similarity = decode_similarity(similarity)?;

        let registry = read_recovering(vectors());
        let handle = registry.get(vectors_handle).ok_or_else(|| {
            set_last_error("ffi_knn_byte_vector_search: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;

        let input = vectors_input(handle)?;
        let query = vector_query::KnnByteVectorQuery {
            field: field.to_string(),
            target: target.to_vec(),
            k,
            ef_search,
            visited_limit,
            similarity,
        };
        let hits =
            vector_query::search_knn_byte_vector_query(&input, &query).map_err(map_knn_error)?;
        drop(registry);

        let handle =
            scored_results().insert_checked(crate::registry::ScoredResultsHandle { hits })?;
        // SAFETY: caller contract guarantees the out pointer is valid for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// Resolves a filter `BooleanQuery`, given in this crate's occur-tagged
/// clause-array wire format, into the accept bitset
/// [`vector_query::VectorsInput::filter`] takes.
///
/// **Why a second handle.** A vector field needs no term dictionary, so
/// [`ffi_open_vectors`] deliberately opens none (see this module's header) --
/// but a filter clause is a `TermQuery`, and resolving one needs the
/// segment's `.tim`/`.tip`/`.tmd`/`.doc`. Rather than widen the vectors
/// handle (which would make a vectors-only segment unopenable again, the very
/// thing that handle exists to allow), the filtered entry points take the
/// **same segment's** [`crate::segment::SegmentHandle`] as a second argument.
/// Java has no equivalent choice: its `LeafReader` is one object with both.
///
/// The two handles must describe the same segment, and `max_doc` is the check
/// for that: a filter resolved against a different segment yields doc ids that
/// mean something else, which no downstream code can detect (the accept set
/// would simply be wrong, silently). It is the same hazard
/// `lucene_search::vector_query::accept_bitset` documents for its own inputs.
///
/// **Deletions** are the vectors handle's, not this one's: the search layer
/// intersects the filter with its own `live_docs` when it builds the accept
/// set (Java's `AcceptDocs.fromIteratorSupplier(.., liveDocs, maxDoc)`). The
/// segment handle's `.liv` is still passed to the clause resolution, because
/// that is what every other `ffi_search_*` entry point does and intersecting
/// twice is idempotent -- but a caller that attaches deletions to only one of
/// the two handles should attach them to the **vectors** one, which is the
/// handle that decides what a KNN search may return.
///
/// # Safety
/// Every clause array must satisfy [`crate::query::read_boolean_query`]'s
/// contract for `clause_count` elements.
#[allow(clippy::too_many_arguments)]
unsafe fn resolve_filter(
    filter_segment_handle: u64,
    vectors_max_doc: i32,
    clause_occurs: *const u8,
    clause_kinds: *const u8,
    clause_fields: *const *const c_char,
    clause_field_lens: *const usize,
    clause_terms: *const *const u8,
    clause_term_lens: *const usize,
    clause_parents: *const i32,
    clause_params: *const i32,
    clause_count: usize,
    minimum_should_match: i32,
) -> Result<FixedBitSet, FfiStatus> {
    // Per-query clause cap before any decoding, exactly as
    // `ffi_search_boolean_query` applies it -- one array, one length.
    crate::query::check_clause_count(clause_count)?;
    // SAFETY: forwarded verbatim from this function's own contract.
    let query = unsafe {
        crate::query::read_boolean_query(
            clause_occurs,
            clause_kinds,
            clause_fields,
            clause_field_lens,
            clause_terms,
            clause_term_lens,
            clause_parents,
            clause_params,
            clause_count,
            minimum_should_match,
        )?
    };

    let segments = read_recovering(crate::registry::segments());
    let segment = segments.get(filter_segment_handle).ok_or_else(|| {
        set_last_error("KNN filter: unknown or already-closed segment handle");
        FfiStatus::InvalidHandle
    })?;
    if segment.max_doc != vectors_max_doc {
        set_last_error(format!(
            "KNN filter: the filter segment's maxDoc ({}) is not the vector segment's ({}), so \
             the two handles are not the same segment and its doc ids would be meaningless here",
            segment.max_doc, vectors_max_doc
        ));
        return Err(FfiStatus::InvalidArgument);
    }
    let doc_in = segment
        .doc_bytes
        .as_deref()
        .map(|b| {
            lucene_codecs::postings::DocInput::open(b, &segment.segment_id, &segment.segment_suffix)
        })
        .transpose()
        .map_err(|e| {
            set_last_error(format!("KNN filter: reopening .doc: {e}"));
            FfiStatus::Decode
        })?;

    let mut collector = lucene_search::VecCollector::default();
    lucene_search::search_boolean_query(
        &segment.fields,
        doc_in.as_ref(),
        None,
        None,
        segment.live_docs.as_ref(),
        None,
        &query,
        &mut collector,
    )
    .map_err(crate::query::map_search_error)?;
    // An empty clause list is a `BooleanQuery` with no clauses, which matches
    // nothing -- Java's `rewrite` turns that filter into `MatchNoDocsQuery`
    // and returns no hits, and an all-zero accept set is the same answer
    // (cost 0, so `getLeafResults` takes `exactSearch` over nothing).
    Ok(vector_query::accept_bitset(collector.docs, vectors_max_doc))
}

/// [`ffi_knn_float_vector_search`] with a filter: only documents matching the
/// `BooleanQuery` described by the clause arrays may be returned.
///
/// This is Java's `new KnnFloatVectorQuery(field, target, k, filter)`, whose
/// per-leaf policy `lucene_search::vector_query` ports in full -- the
/// `cost <= perLeafTopK` short circuit into an exact scan, the graph walk
/// with `acceptOrds` and `visitedLimit = cost + 1`, and the exact-search
/// fallback when that walk came back early-terminated or short. None of that
/// is decided here.
///
/// - `filter_segment_handle`: the **same** segment, opened by
///   [`crate::segment::ffi_open_segment`] with at least its
///   `.fnm`/`.tim`/`.tip`/`.tmd`/`.doc` -- see [`resolve_filter`] for why a
///   second handle rather than a wider one, and for the `maxDoc` cross-check
///   that catches a mismatched pair.
/// - the eight `clause_*` arrays plus `clause_count` and
///   `minimum_should_match`: the identical occur-tagged clause-array wire
///   format [`crate::query::read_boolean_query`] documents and every
///   `ffi_search_boolean_query*` entry point already takes. **No second
///   encoding was invented**: `c13-ffi-surface` rebuilt that format precisely
///   so the next clause-shaped addition would need no new one, and this is
///   that addition.
/// - every other parameter is [`ffi_knn_float_vector_search`]'s, unchanged.
///
/// Java resolves the filter to a `Weight` and evaluates it per leaf inside
/// `rewrite`; this port has no `IndexSearcher`, so the filter is resolved to
/// a doc set here and handed down as one, which is the shape
/// `lucene_search::vector_query::VectorsInput::filter` documents. A caller
/// that wants Java's "a `MatchAllDocsQuery` filter is dropped" short circuit
/// should call [`ffi_knn_float_vector_search`] instead, which is that path
/// exactly.
///
/// # Safety
/// [`ffi_knn_float_vector_search`]'s contract, plus
/// [`crate::query::read_boolean_query`]'s for the clause arrays.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_knn_float_vector_search_filtered(
    vectors_handle: u64,
    filter_segment_handle: u64,
    field_name: *const c_char,
    field_name_len: usize,
    query: *const f32,
    query_len: usize,
    k: usize,
    ef_search: usize,
    similarity: i32,
    visited_limit: u64,
    clause_occurs: *const u8,
    clause_kinds: *const u8,
    clause_fields: *const *const c_char,
    clause_field_lens: *const usize,
    clause_terms: *const *const u8,
    clause_term_lens: *const usize,
    clause_parents: *const i32,
    clause_params: *const i32,
    clause_count: usize,
    minimum_should_match: i32,
    out_scored_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_scored_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        if query.is_null() && query_len != 0 {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `field_name` is valid for
        // `field_name_len` bytes.
        let field = unsafe { str_from_raw(field_name, field_name_len)? };
        // SAFETY: caller contract guarantees `query` is valid for `query_len`
        // `f32`s; the empty slice is used for a null pointer with length 0.
        let target: &[f32] = if query_len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(query, query_len) }
        };
        let similarity = decode_similarity(similarity)?;
        let max_doc = vectors_max_doc(vectors_handle, "ffi_knn_float_vector_search_filtered")?;
        // SAFETY: the clause arrays are forwarded exactly as received.
        let filter = unsafe {
            resolve_filter(
                filter_segment_handle,
                max_doc,
                clause_occurs,
                clause_kinds,
                clause_fields,
                clause_field_lens,
                clause_terms,
                clause_term_lens,
                clause_parents,
                clause_params,
                clause_count,
                minimum_should_match,
            )?
        };

        let registry = read_recovering(vectors());
        let handle = registry.get(vectors_handle).ok_or_else(|| {
            set_last_error(
                "ffi_knn_float_vector_search_filtered: unknown or already-closed handle",
            );
            FfiStatus::InvalidHandle
        })?;
        let mut input = vectors_input(handle)?;
        input.filter = Some(&filter);
        let query = vector_query::KnnFloatVectorQuery {
            field: field.to_string(),
            target: target.to_vec(),
            k,
            ef_search,
            visited_limit,
            similarity,
        };
        let hits =
            vector_query::search_knn_float_vector_query(&input, &query).map_err(map_knn_error)?;
        drop(registry);

        let handle =
            scored_results().insert_checked(crate::registry::ScoredResultsHandle { hits })?;
        // SAFETY: caller contract guarantees the out pointer is valid for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// `KnnByteVectorQuery`'s equivalent of
/// [`ffi_knn_float_vector_search_filtered`]: identical except that the field
/// must be BYTE-encoded and `query`/`query_len` is a byte vector, exactly as
/// [`ffi_knn_byte_vector_search`] is to [`ffi_knn_float_vector_search`].
///
/// # Safety
/// [`ffi_knn_byte_vector_search`]'s contract, plus
/// [`crate::query::read_boolean_query`]'s for the clause arrays.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_knn_byte_vector_search_filtered(
    vectors_handle: u64,
    filter_segment_handle: u64,
    field_name: *const c_char,
    field_name_len: usize,
    query: *const u8,
    query_len: usize,
    k: usize,
    ef_search: usize,
    similarity: i32,
    visited_limit: u64,
    clause_occurs: *const u8,
    clause_kinds: *const u8,
    clause_fields: *const *const c_char,
    clause_field_lens: *const usize,
    clause_terms: *const *const u8,
    clause_term_lens: *const usize,
    clause_parents: *const i32,
    clause_params: *const i32,
    clause_count: usize,
    minimum_should_match: i32,
    out_scored_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_scored_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `field_name`/`query` are valid
        // for their paired lengths.
        let (field, target) = unsafe {
            (
                str_from_raw(field_name, field_name_len)?,
                crate::raw::bytes_from_raw(query, query_len)?,
            )
        };
        let similarity = decode_similarity(similarity)?;
        let max_doc = vectors_max_doc(vectors_handle, "ffi_knn_byte_vector_search_filtered")?;
        // SAFETY: the clause arrays are forwarded exactly as received.
        let filter = unsafe {
            resolve_filter(
                filter_segment_handle,
                max_doc,
                clause_occurs,
                clause_kinds,
                clause_fields,
                clause_field_lens,
                clause_terms,
                clause_term_lens,
                clause_parents,
                clause_params,
                clause_count,
                minimum_should_match,
            )?
        };

        let registry = read_recovering(vectors());
        let handle = registry.get(vectors_handle).ok_or_else(|| {
            set_last_error("ffi_knn_byte_vector_search_filtered: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        let mut input = vectors_input(handle)?;
        input.filter = Some(&filter);
        let query = vector_query::KnnByteVectorQuery {
            field: field.to_string(),
            target: target.to_vec(),
            k,
            ef_search,
            visited_limit,
            similarity,
        };
        let hits =
            vector_query::search_knn_byte_vector_query(&input, &query).map_err(map_knn_error)?;
        drop(registry);

        let handle =
            scored_results().insert_checked(crate::registry::ScoredResultsHandle { hits })?;
        // SAFETY: caller contract guarantees the out pointer is valid for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// This vectors handle's `maxDoc`, read and released before the filter is
/// resolved.
///
/// Taking the vectors registry's guard, dropping it, and taking the segments
/// registry's inside [`resolve_filter`] -- rather than holding both at once --
/// is deliberate: two registries held together in one order here and the
/// other order anywhere else is a lock cycle, and this is the only entry
/// point in the crate that needs two.
fn vectors_max_doc(vectors_handle: u64, what: &str) -> Result<i32, FfiStatus> {
    let registry = read_recovering(vectors());
    let handle = registry.get(vectors_handle).ok_or_else(|| {
        set_last_error(format!("{what}: unknown or already-closed handle"));
        FfiStatus::InvalidHandle
    })?;
    Ok(handle.max_doc)
}

/// Closes a vectors handle opened by [`ffi_open_vectors`]. Returns
/// [`FfiStatus::InvalidHandle`] for an unknown/already-closed handle.
#[no_mangle]
pub extern "C" fn ffi_close_vectors(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(vectors())
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| {
                set_last_error("ffi_close_vectors: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::{ffi_close_directory, ffi_open_directory};
    use crate::results_scored::{ffi_close_scored_results, ffi_scored_results_copy};
    use lucene_util::fixed_bit_set::FixedBitSet;

    /// `fixtures/data/vectors_index` -- a real `IndexWriter`-written segment
    /// carrying five vector fields (dense/sparse FLOAT32 with all three float
    /// similarities, a BYTE field, and a below-threshold field with no graph)
    /// and no term dictionary at all. Its `manifest.properties` records, per
    /// query, the exact `(doc, score)` list real Lucene's
    /// `KnnFloatVectorQuery`/`KnnByteVectorQuery` returned -- so these tests
    /// are differential against Java, not against our own expectations.
    fn fixture_dir_path() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/vectors_index"
        )
        .to_string()
    }

    /// The calling thread's last-error message, read back through the real
    /// exported accessor (not the thread-local directly), so these tests also
    /// prove the message actually reaches a JNI caller.
    fn last_error() -> String {
        let mut buf = [0 as c_char; 512];
        let rc = unsafe {
            crate::ffi_get_last_error_message(buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    struct Manifest(Vec<(String, String)>);

    impl Manifest {
        fn load() -> Self {
            let text =
                std::fs::read_to_string(format!("{}/manifest.properties", fixture_dir_path()))
                    .expect("vectors fixture manifest");
            Manifest(
                text.lines()
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .filter_map(|l| l.split_once('='))
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            )
        }
        fn opt(&self, key: &str) -> Option<&str> {
            self.0
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
        }
        fn get(&self, key: &str) -> &str {
            self.opt(key)
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
        }
        fn int(&self, key: &str) -> i32 {
            self.get(key).parse().expect("integer manifest value")
        }
    }

    fn segment_id(m: &Manifest) -> [u8; 16] {
        let hex = m.get("id_hex");
        let mut id = [0u8; 16];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).expect("hex byte");
        }
        id
    }

    /// `doc:scoreBits;doc:scoreBits;...` -- scores are `Float.floatToIntBits`
    /// so the fixture is exact, not rounded.
    fn parse_hits(spec: &str) -> Vec<(i32, f32)> {
        spec.split(';')
            .filter(|s| !s.is_empty())
            .map(|pair| {
                let (d, s) = pair.split_once(':').expect("doc:score");
                (
                    d.parse().unwrap(),
                    f32::from_bits(s.parse::<i32>().unwrap() as u32),
                )
            })
            .collect()
    }

    fn open_dir() -> u64 {
        let path = fixture_dir_path();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_directory(
                path.as_ptr() as *const c_char,
                path.len(),
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        handle
    }

    /// Opens the fixture's vectors, with or without the HNSW graph files.
    fn open_vectors(dir_handle: u64, with_graph: bool) -> u64 {
        let m = Manifest::load();
        let id = segment_id(&m);
        let fnm = format!("{}.fnm", m.get("segment_name"));
        let vemf = m.get("vemf_file").to_string();
        let vec_file = m.get("vec_file").to_string();
        let vem = m.get("vem_file").to_string();
        let vex = m.get("vex_file").to_string();
        let suffix = m.get("segment_suffix").to_string();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_vectors(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                vemf.as_ptr() as *const c_char,
                vemf.len(),
                vec_file.as_ptr() as *const c_char,
                vec_file.len(),
                if with_graph {
                    vem.as_ptr() as *const c_char
                } else {
                    std::ptr::null()
                },
                vem.len(),
                if with_graph {
                    vex.as_ptr() as *const c_char
                } else {
                    std::ptr::null()
                },
                vex.len(),
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                m.int("max_doc"),
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code(), "ffi_open_vectors");
        handle
    }

    fn read_scored(handle: u64) -> Vec<(i32, f32)> {
        let len = {
            let mut len: usize = 0;
            let rc = unsafe {
                crate::results_scored::ffi_scored_results_len(handle, &mut len as *mut _)
            };
            assert_eq!(rc, FfiStatus::Ok.code());
            len
        };
        let mut docs = vec![0i32; len];
        let mut scores = vec![0f32; len];
        let rc =
            unsafe { ffi_scored_results_copy(handle, docs.as_mut_ptr(), scores.as_mut_ptr(), len) };
        assert_eq!(rc, FfiStatus::Ok.code());
        docs.into_iter().zip(scores).collect()
    }

    fn float_query(m: &Manifest, key: &str) -> Vec<f32> {
        m.get(key)
            .split(',')
            .map(|s| f32::from_bits(s.parse::<i32>().unwrap() as u32))
            .collect()
    }

    fn byte_query(m: &Manifest, key: &str) -> Vec<u8> {
        m.get(key)
            .split(',')
            .map(|s| s.parse::<i32>().unwrap() as i8 as u8)
            .collect()
    }

    fn search_float(vh: u64, field: &str, q: &[f32], k: usize, ef: usize, sim: i32) -> (i32, u64) {
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_knn_float_vector_search(
                vh,
                field.as_ptr() as *const c_char,
                field.len(),
                q.as_ptr(),
                q.len(),
                k,
                ef,
                sim,
                0,
                &mut out as *mut _,
            )
        };
        (rc, out)
    }

    fn search_byte(vh: u64, field: &str, q: &[u8], k: usize, ef: usize, sim: i32) -> (i32, u64) {
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_knn_byte_vector_search(
                vh,
                field.as_ptr() as *const c_char,
                field.len(),
                q.as_ptr(),
                q.len(),
                k,
                ef,
                sim,
                0,
                &mut out as *mut _,
            )
        };
        (rc, out)
    }

    fn assert_hits_match(got: &[(i32, f32)], expected: &[(i32, f32)], what: &str) {
        assert_eq!(got.len(), expected.len(), "{what}: hit count");
        for (i, ((gd, gs), (ed, es))) in got.iter().zip(expected).enumerate() {
            assert_eq!(gd, ed, "{what}: doc at rank {i}");
            assert!(
                (gs - es).abs() <= 1e-6 * es.abs().max(1.0),
                "{what}: score at rank {i}: {gs} vs {es}"
            );
        }
    }

    /// The headline differential test: every FLOAT32 and BYTE query the
    /// fixture records, run through the exported C ABI, must return exactly
    /// what real Lucene's `KnnFloatVectorQuery`/`KnnByteVectorQuery` returned
    /// -- same doc ids, same order, same scores.
    #[test]
    fn knn_search_reproduces_lucene_knn_vector_query_results() {
        let m = Manifest::load();
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let k = 10usize;
        let mut checked = 0;
        for i in 0..m.int("field_count") {
            let fk = format!("f{i}");
            let Some(count) = m.opt(&format!("q.{fk}.count")) else {
                continue;
            };
            let field = m.get(&format!("{fk}.name")).to_string();
            let float = m.get(&format!("{fk}.encoding")) == "FLOAT32";
            for q in 0..count.parse::<i32>().unwrap() {
                let qk = format!("q.{fk}.{q}");
                let expected = parse_hits(m.get(&format!("{qk}.hnsw")));
                let (rc, handle) = if float {
                    let query = float_query(&m, &format!("{qk}.vec"));
                    search_float(vh, &field, &query, k, 0, SIMILARITY_FROM_FIELD)
                } else {
                    let query = byte_query(&m, &format!("{qk}.vec"));
                    search_byte(vh, &field, &query, k, 0, SIMILARITY_FROM_FIELD)
                };
                assert_eq!(rc, FfiStatus::Ok.code(), "{qk}");
                assert_hits_match(&read_scored(handle), &expected, &qk);
                ffi_close_scored_results(handle);
                checked += 1;
            }
        }
        assert!(checked >= 60, "expected several fields' worth of queries");
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    /// With no `.vem`/`.vex` opened, `hnsw_vectors::search` takes its
    /// exhaustive branch -- which is *exact*, so it must reproduce the
    /// fixture's brute-force expectations rather than the graph's.
    #[test]
    fn without_a_graph_the_search_is_lucene_s_exact_brute_force() {
        let m = Manifest::load();
        let dir = open_dir();
        let vh = open_vectors(dir, false);
        let field = m.get("f0.name").to_string();
        let mut checked = 0;
        for q in 0..3 {
            let qk = format!("q.f0.{q}");
            let expected = parse_hits(m.get(&format!("{qk}.exact")));
            let query = float_query(&m, &format!("{qk}.vec"));
            let (rc, handle) = search_float(vh, &field, &query, 10, 0, SIMILARITY_FROM_FIELD);
            assert_eq!(rc, FfiStatus::Ok.code());
            assert_hits_match(&read_scored(handle), &expected, &qk);
            ffi_close_scored_results(handle);
            checked += 1;
        }
        assert_eq!(checked, 3);
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    /// A field written below `HNSW_GRAPH_THRESHOLD` carries no graph even
    /// when `.vem`/`.vex` are opened -- and must still search, exactly.
    #[test]
    fn a_field_with_no_graph_in_an_opened_vex_still_searches_exactly() {
        let m = Manifest::load();
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let field = m.get("f4.name").to_string();
        let size = m.int("f4.count");
        assert!(size < 100, "f4 is the below-threshold, graphless field");
        let query = vec![0.5f32; m.int("f4.dim") as usize];
        let (rc, handle) = search_float(vh, &field, &query, 3, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored(handle);
        assert_eq!(hits.len(), 3);
        // Exhaustive search is exact: strictly descending by score.
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
        ffi_close_scored_results(handle);
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    /// A sparse field's ordinals are not its doc ids -- every returned doc
    /// must be one the fixture says actually has a vector.
    #[test]
    fn a_sparse_field_returns_doc_ids_not_ordinals() {
        let m = Manifest::load();
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let field = m.get("f1.name").to_string();
        let expected = parse_hits(m.get("q.f1.0.hnsw"));
        let query = float_query(&m, "q.f1.0.vec");
        let (rc, handle) = search_float(vh, &field, &query, 10, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored(handle);
        assert_hits_match(&hits, &expected, "q.f1.0");
        // The sparse mapping is real: at least one hit's doc id exceeds the
        // field's vector count, so an ordinal would have been a different
        // (wrong) number.
        let count = m.int("f1.count");
        assert!(
            hits.iter().any(|(d, _)| *d >= count),
            "expected a doc id past the ordinal range"
        );
        ffi_close_scored_results(handle);
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    /// A wider beam is allowed to be *better* than `KnnFloatVectorQuery`'s
    /// `k`-wide one, never worse: every hit must still come back in
    /// descending score order, and the top hit must be at least as good.
    #[test]
    fn a_wider_ef_search_never_returns_worse_hits() {
        let m = Manifest::load();
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let field = m.get("f0.name").to_string();
        let query = float_query(&m, "q.f0.0.vec");
        let (rc, narrow) = search_float(vh, &field, &query, 10, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::Ok.code());
        let (rc, wide) = search_float(vh, &field, &query, 10, 200, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::Ok.code());
        let (narrow_hits, wide_hits) = (read_scored(narrow), read_scored(wide));
        assert_eq!(wide_hits.len(), 10);
        for (i, (n, w)) in narrow_hits.iter().zip(&wide_hits).enumerate() {
            assert!(w.1 >= n.1 - 1e-6, "rank {i}: wider beam scored worse");
        }
        ffi_close_scored_results(narrow);
        ffi_close_scored_results(wide);
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    #[test]
    fn k_zero_is_rejected_like_java_s_knn_vector_query() {
        let m = Manifest::load();
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let field = m.get("f0.name").to_string();
        let query = float_query(&m, "q.f0.0.vec");
        let (rc, _) = search_float(vh, &field, &query, 0, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(last_error().contains("k must be at least 1"));
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    /// A `k` a caller could only produce by accident (a negative Java `int`
    /// widened to `usize`) must be a status code, never the heap allocation
    /// `KnnCollector::new` would otherwise attempt -- an allocation failure
    /// aborts, and `catch_unwind` cannot contain an abort.
    #[test]
    fn an_absurd_k_is_clamped_to_the_field_size_not_allocated() {
        let m = Manifest::load();
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let field = m.get("f4.name").to_string();
        let query = vec![0.25f32; m.int("f4.dim") as usize];
        let (rc, handle) = search_float(vh, &field, &query, usize::MAX, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::Ok.code());
        // Clamped to the field's own vector count, which is all there is.
        assert_eq!(read_scored(handle).len(), m.int("f4.count") as usize);
        ffi_close_scored_results(handle);
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    /// Java's `AbstractKnnVectorQuery` raises `IllegalArgumentException` for a
    /// query vector of the wrong length, so this must be `InvalidArgument`
    /// with Java's message -- not the `Decode` the reader would report, which
    /// a JNI caller reads as "the index is corrupt".
    #[test]
    fn a_wrong_dimension_query_is_an_invalid_argument_not_a_decode_error() {
        let m = Manifest::load();
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let field = m.get("f0.name").to_string();
        let (rc, _) = search_float(vh, &field, &[1.0, 2.0], 5, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(last_error().contains("differs from field dimension: 16"));
        // Same for the byte entry point.
        let byte_field = m.get("f3.name").to_string();
        let (rc, _) = search_byte(vh, &byte_field, &[1, 2], 5, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    /// Exactly one of `.vem`/`.vex` is a caller bug, and must say so rather
    /// than silently degrading every later search to an exhaustive scan.
    #[test]
    fn a_half_specified_hnsw_graph_pair_is_rejected() {
        let m = Manifest::load();
        let id = segment_id(&m);
        let dir = open_dir();
        let fnm = format!("{}.fnm", m.get("segment_name"));
        let vemf = m.get("vemf_file").to_string();
        let vec_file = m.get("vec_file").to_string();
        let vem = m.get("vem_file").to_string();
        let suffix = m.get("segment_suffix").to_string();
        let mut h: u64 = 0;
        let rc = unsafe {
            ffi_open_vectors(
                dir,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                vemf.as_ptr() as *const c_char,
                vemf.len(),
                vec_file.as_ptr() as *const c_char,
                vec_file.len(),
                vem.as_ptr() as *const c_char,
                vem.len(),
                std::ptr::null(), // .vex missing
                0,
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                m.int("max_doc"),
                &mut h as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(last_error().contains("together or not at all"));
        ffi_close_directory(dir);
    }

    #[test]
    fn searching_a_byte_field_as_float32_is_rejected() {
        let m = Manifest::load();
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let field = m.get("f3.name").to_string();
        let query = vec![0.0f32; m.int("f3.dim") as usize];
        let (rc, _) = search_float(vh, &field, &query, 5, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(last_error().contains("Byte"));
        // ... and the mirror image.
        let float_field = m.get("f0.name").to_string();
        let bq = vec![0u8; m.int("f0.dim") as usize];
        let (rc, _) = search_byte(vh, &float_field, &bq, 5, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    #[test]
    fn a_matching_similarity_is_accepted_and_a_mismatching_one_is_not() {
        let m = Manifest::load();
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let field = m.get("f0.name").to_string(); // EUCLIDEAN == ordinal 0
        let query = float_query(&m, "q.f0.0.vec");
        let (rc, handle) = search_float(vh, &field, &query, 5, 0, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(read_scored(handle).len(), 5);
        ffi_close_scored_results(handle);
        // COSINE (2) against a EUCLIDEAN field.
        let (rc, _) = search_float(vh, &field, &query, 5, 0, 2);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(last_error().contains("does not match the field's own"));
        // Not an ordinal at all.
        let (rc, _) = search_float(vh, &field, &query, 5, 0, 9);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(last_error().contains("VectorSimilarityFunction"));
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    #[test]
    fn every_similarity_function_in_the_fixture_round_trips_its_own_ordinal() {
        let m = Manifest::load();
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        for (fk, ordinal) in [("f0", 0), ("f2", 3), ("f1", 2), ("f3", 1)] {
            let field = m.get(&format!("{fk}.name")).to_string();
            let float = m.get(&format!("{fk}.encoding")) == "FLOAT32";
            let (rc, handle) = if float {
                let q = vec![0.1f32; m.int(&format!("{fk}.dim")) as usize];
                search_float(vh, &field, &q, 2, 0, ordinal)
            } else {
                let q = vec![1u8; m.int(&format!("{fk}.dim")) as usize];
                search_byte(vh, &field, &q, 2, 0, ordinal)
            };
            assert_eq!(rc, FfiStatus::Ok.code(), "{fk} with ordinal {ordinal}");
            ffi_close_scored_results(handle);
        }
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    #[test]
    fn a_field_name_absent_from_the_fnm_is_an_invalid_argument() {
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let (rc, _) = search_float(vh, "no_such_field", &[0.0; 4], 5, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(last_error().contains("unknown field"));
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    /// A field the `.fnm` declares but the `.vemf` has no entry for is a
    /// *different* error from an unknown name, and a real one: a segment's
    /// `.fnm` lists every field, vector or not. This fixture happens to
    /// contain nothing but vector fields, so the state is fabricated on the
    /// handle -- the same test-only technique `query.rs`'s
    /// `corrupt_doc_bytes` uses to reach a branch the public API cannot
    /// produce on its own.
    #[test]
    fn a_field_declared_in_the_fnm_but_carrying_no_vectors_is_an_invalid_argument() {
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        {
            let mut registry = lock_recovering(vectors());
            let handle = registry.get_mut(vh).unwrap();
            let mut declared = handle.field_infos.fields[0].clone();
            declared.name = "stored_only".to_string();
            // A field number no `.vemf` entry can possibly use.
            declared.number = 4242;
            handle.field_infos.fields.push(declared);
        }
        let (rc, _) = search_float(vh, "stored_only", &[0.0; 16], 5, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(last_error().contains("has no vectors in this segment"));
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    #[test]
    fn unknown_handles_and_null_pointers_are_rejected() {
        let m = Manifest::load();
        let query = float_query(&m, "q.f0.0.vec");
        let (rc, _) = search_float(0xDEAD_BEEF, "dense_f32", &query, 5, 0, -1);
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        let (rc, _) = search_byte(0xDEAD_BEEF, "byte_dot", &[1, 2, 3], 5, 0, -1);
        assert_eq!(rc, FfiStatus::InvalidHandle.code());

        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let field = "dense_f32";
        let rc = unsafe {
            ffi_knn_float_vector_search(
                vh,
                field.as_ptr() as *const c_char,
                field.len(),
                query.as_ptr(),
                query.len(),
                5,
                0,
                -1,
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let rc = unsafe {
            ffi_knn_float_vector_search(
                vh,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                4,
                5,
                0,
                -1,
                0,
                &mut 0u64 as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let rc = unsafe {
            ffi_knn_byte_vector_search(
                vh,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                4,
                5,
                0,
                -1,
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    #[test]
    fn open_vectors_rejects_a_negative_max_doc_and_null_out_pointers() {
        let m = Manifest::load();
        let id = segment_id(&m);
        let dir = open_dir();
        let fnm = format!("{}.fnm", m.get("segment_name"));
        let vemf = m.get("vemf_file").to_string();
        let vec_file = m.get("vec_file").to_string();
        let suffix = m.get("segment_suffix").to_string();
        let call = |max_doc: i32, out: *mut u64, id_ptr: *const u8| unsafe {
            ffi_open_vectors(
                dir,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                vemf.as_ptr() as *const c_char,
                vemf.len(),
                vec_file.as_ptr() as *const c_char,
                vec_file.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                id_ptr,
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                max_doc,
                out,
            )
        };
        let mut h: u64 = 0;
        assert_eq!(
            call(-1, &mut h as *mut _, id.as_ptr()),
            FfiStatus::InvalidArgument.code()
        );
        assert_eq!(
            call(10, std::ptr::null_mut(), id.as_ptr()),
            FfiStatus::NullPointer.code()
        );
        assert_eq!(
            call(10, &mut h as *mut _, std::ptr::null()),
            FfiStatus::NullPointer.code()
        );
        ffi_close_directory(dir);
    }

    #[test]
    fn a_corrupt_vemf_is_a_decode_error_at_open_time() {
        let m = Manifest::load();
        let id = segment_id(&m);
        let dir = open_dir();
        // Point `vemf_name` at the `.vec` file: a real file with the wrong
        // codec header.
        let fnm = format!("{}.fnm", m.get("segment_name"));
        let vec_file = m.get("vec_file").to_string();
        let suffix = m.get("segment_suffix").to_string();
        let mut h: u64 = 0;
        let rc = unsafe {
            ffi_open_vectors(
                dir,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                vec_file.as_ptr() as *const c_char,
                vec_file.len(),
                vec_file.as_ptr() as *const c_char,
                vec_file.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                m.int("max_doc"),
                &mut h as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());
        ffi_close_directory(dir);
    }

    #[test]
    fn a_corrupt_hnsw_index_is_a_decode_error_at_open_time() {
        let m = Manifest::load();
        let id = segment_id(&m);
        let dir = open_dir();
        let fnm = format!("{}.fnm", m.get("segment_name"));
        let vemf = m.get("vemf_file").to_string();
        let vec_file = m.get("vec_file").to_string();
        let suffix = m.get("segment_suffix").to_string();
        let mut h: u64 = 0;
        let rc = unsafe {
            ffi_open_vectors(
                dir,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                vemf.as_ptr() as *const c_char,
                vemf.len(),
                vec_file.as_ptr() as *const c_char,
                vec_file.len(),
                // `.vemf` where `.vem` belongs.
                vemf.as_ptr() as *const c_char,
                vemf.len(),
                vec_file.as_ptr() as *const c_char,
                vec_file.len(),
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                m.int("max_doc"),
                &mut h as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());
        ffi_close_directory(dir);
    }

    #[test]
    fn closing_twice_is_an_invalid_handle_and_a_segment_handle_is_rejected() {
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        assert_eq!(ffi_close_vectors(vh), FfiStatus::Ok.code());
        assert_eq!(ffi_close_vectors(vh), FfiStatus::InvalidHandle.code());
        // A directory handle carries a different registry tag, so it can
        // never be accepted here.
        assert_eq!(ffi_close_vectors(dir), FfiStatus::InvalidHandle.code());
        assert_eq!(
            unsafe { ffi_vectors_set_live_docs(0xBEEF, dir, std::ptr::null(), 0, -1, 0) },
            FfiStatus::InvalidHandle.code()
        );
        ffi_close_directory(dir);
    }

    /// Deletions: attaching a `.liv` must remove deleted docs from the KNN
    /// result, and the widened beam must still fill `k`.
    #[test]
    fn attached_live_docs_remove_deleted_docs_from_knn_results() {
        let m = Manifest::load();
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let field = m.get("f0.name").to_string();
        let query = float_query(&m, "q.f0.0.vec");
        let (rc, handle) = search_float(vh, &field, &query, 10, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::Ok.code());
        let before = read_scored(handle);
        ffi_close_scored_results(handle);

        // Fabricate deletions directly on the handle: the fixture carries no
        // `.liv`, and what is under test is the filtering, not `.liv`
        // decoding (which `segment.rs`'s own tests cover against a real
        // Java-written `.liv`).
        let deleted: Vec<i32> = before.iter().take(3).map(|(d, _)| *d).collect();
        {
            let mut registry = lock_recovering(vectors());
            let h = registry.get_mut(vh).unwrap();
            let mut bits = FixedBitSet::new(m.int("max_doc") as usize);
            for d in 0..m.int("max_doc") {
                if !deleted.contains(&d) {
                    bits.set(d as usize);
                }
            }
            h.live_docs = Some(bits);
        }

        let (rc, handle) = search_float(vh, &field, &query, 10, 0, SIMILARITY_FROM_FIELD);
        assert_eq!(rc, FfiStatus::Ok.code());
        let after = read_scored(handle);
        for (d, _) in &after {
            assert!(!deleted.contains(d), "deleted doc {d} was returned");
        }
        // The beam was widened by the deleted count, so `k` is still filled.
        assert_eq!(after.len(), 10);
        // And the survivors of the original top-10 are still there, in order.
        let survivors: Vec<i32> = before
            .iter()
            .map(|(d, _)| *d)
            .filter(|d| !deleted.contains(d))
            .collect();
        let got: Vec<i32> = after.iter().map(|(d, _)| *d).collect();
        assert_eq!(&got[..survivors.len()], &survivors[..]);
        ffi_close_scored_results(handle);
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    #[test]
    fn set_live_docs_validates_del_gen_and_del_count() {
        let dir = open_dir();
        let vh = open_vectors(dir, true);
        let name = "_0_1.liv";
        let rc = unsafe {
            ffi_vectors_set_live_docs(vh, dir, name.as_ptr() as *const c_char, name.len(), 0, 0)
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        let rc = unsafe {
            ffi_vectors_set_live_docs(vh, dir, name.as_ptr() as *const c_char, name.len(), 1, -1)
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        // A null name clears, and is always accepted.
        assert_eq!(
            unsafe { ffi_vectors_set_live_docs(vh, dir, std::ptr::null(), 0, -1, 0) },
            FfiStatus::Ok.code()
        );
        ffi_close_vectors(vh);
        ffi_close_directory(dir);
    }

    /// The `similarity` argument is decoded here (boundary work) and
    /// cross-checked one layer down (policy). This covers the decode half:
    /// `-1` is "the field's own", `0..=3` are the `.vemf`'s own pinned
    /// ordinals, and nothing else is accepted at all.
    #[test]
    fn the_similarity_argument_decodes_the_pinned_file_format_ordinals() {
        assert_eq!(decode_similarity(SIMILARITY_FROM_FIELD).unwrap(), None);
        for (ordinal, expected) in [
            (0, VectorSimilarityFunction::Euclidean),
            (1, VectorSimilarityFunction::DotProduct),
            (2, VectorSimilarityFunction::Cosine),
            (3, VectorSimilarityFunction::MaximumInnerProduct),
        ] {
            assert_eq!(decode_similarity(ordinal).unwrap(), Some(expected));
        }
        for bad in [-2, 4, 9, i32::MIN, i32::MAX] {
            assert_eq!(decode_similarity(bad), Err(FfiStatus::InvalidArgument));
            assert!(last_error().contains("VectorSimilarityFunction"));
        }
    }

    /// `ffi_knn_*_vector_search` must keep a *caller* mistake and a *corrupt
    /// index* apart, because a JNI caller reads `Decode` as "fail this shard".
    #[test]
    fn a_caller_error_is_an_invalid_argument_and_a_decode_error_is_a_decode() {
        assert_eq!(
            map_knn_error(lucene_search::Error::InvalidKnnQuery(
                "k must be at least 1".into()
            )),
            FfiStatus::InvalidArgument
        );
        assert_eq!(last_error(), "k must be at least 1");
        assert_eq!(
            map_knn_error(lucene_search::Error::Vectors(
                lucene_codecs::vectors::Error::OrdOutOfRange(7, 3)
            )),
            FfiStatus::Decode
        );
        assert!(last_error().starts_with("KNN search: "));
        // Anything else is this crate's existing generic search status.
        assert_eq!(
            map_knn_error(lucene_search::Error::MissingPosInput),
            FfiStatus::Search
        );
    }

    // -----------------------------------------------------------------------
    // Filtered KNN over the C ABI (`fixtures/data/vectors_filter_index`)
    // -----------------------------------------------------------------------

    /// A second fixture, and a second set of helpers, because the filtered
    /// entry points need something `vectors_index` deliberately does not
    /// have: a **term dictionary**. `fixtures/data/vectors_filter_index` is
    /// one 1200-document segment carrying a FLOAT32 and a BYTE vector field
    /// *and* two `StringField`s, written by `GenVectorsFiltered.java` and
    /// queried through a real `IndexSearcher` -- so the recorded results are
    /// what Lucene's `new KnnFloatVectorQuery(field, target, k, filter)`
    /// actually returned, on a one-leaf index where `perLeafTopK == k` and no
    /// re-entry pass runs, which is exactly what this ABI's one-segment
    /// handle can express.
    ///
    /// Both of Java's filtered branches are covered: `bucket:b0` accepts 6
    /// documents against `k = 10` (the `cost <= perLeafTopK` short circuit
    /// into `exactSearch`) and `group:g0` accepts a quarter of the index (the
    /// graph walk with `acceptOrds` and `visitedLimit = cost + 1`).
    mod filtered {
        use super::*;
        use crate::segment::{ffi_close_segment, ffi_open_segment};

        fn dir_path() -> String {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../fixtures/data/vectors_filter_index"
            )
            .to_string()
        }

        struct M(Vec<(String, String)>);

        impl M {
            fn load() -> Self {
                let text = std::fs::read_to_string(format!("{}/manifest.properties", dir_path()))
                    .expect("run scripts/gen-fixtures.sh first (GenVectorsFiltered)");
                M(text
                    .lines()
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .filter_map(|l| l.split_once('='))
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect())
            }
            fn get(&self, key: &str) -> &str {
                self.0
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v.as_str())
                    .unwrap_or_else(|| panic!("manifest key {key} missing"))
            }
            fn int(&self, key: &str) -> i32 {
                self.get(key).parse().unwrap()
            }
            fn id(&self) -> [u8; 16] {
                let hex = self.get("id_hex");
                let mut id = [0u8; 16];
                for (i, slot) in id.iter_mut().enumerate() {
                    *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
                }
                id
            }
        }

        fn open_dir() -> u64 {
            let path = dir_path();
            let mut handle: u64 = 0;
            let rc = unsafe {
                ffi_open_directory(
                    path.as_ptr() as *const c_char,
                    path.len(),
                    &mut handle as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code());
            handle
        }

        fn open_vectors(m: &M, dir: u64) -> u64 {
            open_vectors_with_max_doc(m, dir, m.int("max_doc"))
        }

        fn open_vectors_with_max_doc(m: &M, dir: u64, max_doc: i32) -> u64 {
            let id = m.id();
            let fnm = format!("{}.fnm", m.get("segment_name"));
            let (vemf, vec_file) = (m.get("vemf_file"), m.get("vec_file"));
            let (vem, vex) = (m.get("vem_file"), m.get("vex_file"));
            let suffix = m.get("segment_suffix");
            let mut handle: u64 = 0;
            let rc = unsafe {
                ffi_open_vectors(
                    dir,
                    fnm.as_ptr() as *const c_char,
                    fnm.len(),
                    vemf.as_ptr() as *const c_char,
                    vemf.len(),
                    vec_file.as_ptr() as *const c_char,
                    vec_file.len(),
                    vem.as_ptr() as *const c_char,
                    vem.len(),
                    vex.as_ptr() as *const c_char,
                    vex.len(),
                    id.as_ptr(),
                    suffix.as_ptr() as *const c_char,
                    suffix.len(),
                    max_doc,
                    &mut handle as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code(), "ffi_open_vectors");
            handle
        }

        /// The same segment's postings, which is where a filter clause's term
        /// is actually resolved.
        fn open_segment(m: &M, dir: u64, max_doc: i32) -> u64 {
            let id = m.id();
            let fnm = format!("{}.fnm", m.get("segment_name"));
            let (tim, tip, tmd, doc) = (
                m.get("tim_file"),
                m.get("tip_file"),
                m.get("tmd_file"),
                m.get("doc_file"),
            );
            let suffix = m.get("postings_suffix");
            let mut handle: u64 = 0;
            let rc = unsafe {
                ffi_open_segment(
                    dir,
                    fnm.as_ptr() as *const c_char,
                    fnm.len(),
                    tim.as_ptr() as *const c_char,
                    tim.len(),
                    tip.as_ptr() as *const c_char,
                    tip.len(),
                    tmd.as_ptr() as *const c_char,
                    tmd.len(),
                    doc.as_ptr() as *const c_char,
                    doc.len(),
                    std::ptr::null(), // .pos
                    0,
                    std::ptr::null(), // .pay
                    0,
                    std::ptr::null(), // .nvm
                    0,
                    std::ptr::null(), // .nvd
                    0,
                    std::ptr::null(), // .dvm
                    0,
                    std::ptr::null(), // .dvd
                    0,
                    std::ptr::null(), // dv suffix
                    0,
                    std::ptr::null(), // .kdm
                    0,
                    std::ptr::null(), // .kdi
                    0,
                    std::ptr::null(), // .kdd
                    0,
                    id.as_ptr(),
                    suffix.as_ptr() as *const c_char,
                    suffix.len(),
                    max_doc,
                    &mut handle as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code(), "ffi_open_segment");
            handle
        }

        /// One `Occur.FILTER` `TermQuery` clause, in the occur-tagged
        /// clause-array wire format `c13-ffi-surface` established --
        /// deliberately the *same* arrays `ffi_search_boolean_query` takes,
        /// not a second encoding.
        struct Clauses {
            occurs: Vec<u8>,
            kinds: Vec<u8>,
            fields: Vec<*const c_char>,
            field_lens: Vec<usize>,
            terms: Vec<*const u8>,
            term_lens: Vec<usize>,
        }

        impl Clauses {
            fn one(field: &str, term: &str) -> Self {
                Clauses {
                    occurs: vec![crate::query::OCCUR_FILTER],
                    kinds: vec![crate::query::CLAUSE_KIND_TERM],
                    fields: vec![field.as_ptr() as *const c_char],
                    field_lens: vec![field.len()],
                    terms: vec![term.as_ptr()],
                    term_lens: vec![term.len()],
                }
            }

            fn none() -> Self {
                Clauses {
                    occurs: Vec::new(),
                    kinds: Vec::new(),
                    fields: Vec::new(),
                    field_lens: Vec::new(),
                    terms: Vec::new(),
                    term_lens: Vec::new(),
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn search_float(
            vh: u64,
            sh: u64,
            field: &str,
            q: &[f32],
            k: usize,
            c: &Clauses,
        ) -> (i32, u64) {
            let mut out: u64 = 0;
            let rc = unsafe {
                ffi_knn_float_vector_search_filtered(
                    vh,
                    sh,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    q.as_ptr(),
                    q.len(),
                    k,
                    0,
                    SIMILARITY_FROM_FIELD,
                    0,
                    c.occurs.as_ptr(),
                    c.kinds.as_ptr(),
                    c.fields.as_ptr(),
                    c.field_lens.as_ptr(),
                    c.terms.as_ptr(),
                    c.term_lens.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    c.occurs.len(),
                    0,
                    &mut out as *mut _,
                )
            };
            (rc, out)
        }

        #[allow(clippy::too_many_arguments)]
        fn search_byte(
            vh: u64,
            sh: u64,
            field: &str,
            q: &[u8],
            k: usize,
            c: &Clauses,
        ) -> (i32, u64) {
            let mut out: u64 = 0;
            let rc = unsafe {
                ffi_knn_byte_vector_search_filtered(
                    vh,
                    sh,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    q.as_ptr(),
                    q.len(),
                    k,
                    0,
                    SIMILARITY_FROM_FIELD,
                    0,
                    c.occurs.as_ptr(),
                    c.kinds.as_ptr(),
                    c.fields.as_ptr(),
                    c.field_lens.as_ptr(),
                    c.terms.as_ptr(),
                    c.term_lens.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    c.occurs.len(),
                    0,
                    &mut out as *mut _,
                )
            };
            (rc, out)
        }

        fn hits(m: &M, key: &str) -> Vec<(i32, f32)> {
            parse_hits(m.get(key))
        }

        fn floats(m: &M, key: &str) -> Vec<f32> {
            m.get(key)
                .split(',')
                .map(|s| f32::from_bits(s.parse::<i32>().unwrap() as u32))
                .collect()
        }

        fn bytes(m: &M, key: &str) -> Vec<u8> {
            m.get(key)
                .split(',')
                .map(|s| s.parse::<i32>().unwrap() as i8 as u8)
                .collect()
        }

        /// The headline differential for this batch's ABI addition: every
        /// filtered query the fixture records, run through the exported
        /// symbols with the filter given as a clause array, must return what
        /// real Lucene returned -- doc for doc, score for score, on both
        /// encodings and both of Java's filtered branches.
        #[test]
        fn filtered_knn_over_the_c_abi_reproduces_lucene() {
            let m = M::load();
            let dir = open_dir();
            let vh = open_vectors(&m, dir);
            let sh = open_segment(&m, dir, m.int("max_doc"));
            let k = m.int("k") as usize;
            let selective = Clauses::one(m.get("selective_field"), m.get("selective_term"));
            let permissive = Clauses::one(m.get("permissive_field"), m.get("permissive_term"));

            let mut checked = 0;
            for (fk, is_byte) in [("f0", false), ("f1", true)] {
                let field = m.get(&format!("{fk}.name")).to_string();
                let count = m.int(&format!("q.{fk}.count"));
                for q in 0..count {
                    let qk = format!("q.{fk}.{q}");
                    for (key, clauses) in [("selective", &selective), ("permissive", &permissive)] {
                        let (rc, out) = if is_byte {
                            let target = bytes(&m, &format!("{qk}.vec"));
                            search_byte(vh, sh, &field, &target, k, clauses)
                        } else {
                            let target = floats(&m, &format!("{qk}.vec"));
                            search_float(vh, sh, &field, &target, k, clauses)
                        };
                        assert_eq!(rc, FfiStatus::Ok.code(), "{qk}.{key}: {}", last_error());
                        let got = read_scored(out);
                        assert_hits_match(
                            &got,
                            &hits(&m, &format!("{qk}.{key}")),
                            &format!("{qk}.{key}"),
                        );
                        assert_eq!(ffi_close_scored_results(out), FfiStatus::Ok.code());
                        checked += 1;
                    }
                }
            }
            assert_eq!(checked, 80, "the fixture's whole filtered query set");
            assert_eq!(ffi_close_segment(sh), FfiStatus::Ok.code());
            assert_eq!(ffi_close_vectors(vh), FfiStatus::Ok.code());
            assert_eq!(ffi_close_directory(dir), FfiStatus::Ok.code());
        }

        /// The filter really is resolved through this port's own term
        /// dictionary: the accepted documents are Lucene's own postings list
        /// for that term, recorded in the manifest, and every returned hit is
        /// one of them. A graph walk that decoded the clause array and then
        /// ignored the accept set would pass the score comparison above on a
        /// permissive filter and fail here.
        #[test]
        fn every_filtered_hit_comes_from_the_terms_own_postings() {
            let m = M::load();
            let dir = open_dir();
            let vh = open_vectors(&m, dir);
            let sh = open_segment(&m, dir, m.int("max_doc"));
            let k = m.int("k") as usize;
            let field = m.get("f0.name").to_string();
            for (docs_key, clauses) in [
                (
                    "selective_docs",
                    Clauses::one(m.get("selective_field"), m.get("selective_term")),
                ),
                (
                    "permissive_docs",
                    Clauses::one(m.get("permissive_field"), m.get("permissive_term")),
                ),
            ] {
                let accepted: Vec<i32> = m
                    .get(docs_key)
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse().unwrap())
                    .collect();
                for q in 0..m.int("q.f0.count") {
                    let target = floats(&m, &format!("q.f0.{q}.vec"));
                    let (rc, out) = search_float(vh, sh, &field, &target, k, &clauses);
                    assert_eq!(rc, FfiStatus::Ok.code(), "{}", last_error());
                    let got = read_scored(out);
                    assert!(!got.is_empty());
                    for (doc, _) in &got {
                        assert!(
                            accepted.contains(doc),
                            "{docs_key}: doc {doc} is not in the term's postings"
                        );
                    }
                    assert_eq!(ffi_close_scored_results(out), FfiStatus::Ok.code());
                }
            }
            assert_eq!(ffi_close_segment(sh), FfiStatus::Ok.code());
            assert_eq!(ffi_close_vectors(vh), FfiStatus::Ok.code());
            assert_eq!(ffi_close_directory(dir), FfiStatus::Ok.code());
        }

        /// An empty clause list matches nothing, which is Java's
        /// `MatchNoDocsQuery` rewrite: no hits, and no error. The degenerate
        /// case a graph walk that quietly dropped the filter would fail.
        #[test]
        fn an_empty_clause_list_is_a_filter_that_accepts_nothing() {
            let m = M::load();
            let dir = open_dir();
            let vh = open_vectors(&m, dir);
            let sh = open_segment(&m, dir, m.int("max_doc"));
            let field = m.get("f0.name").to_string();
            let target = floats(&m, "q.f0.0.vec");
            let (rc, out) = search_float(vh, sh, &field, &target, 10, &Clauses::none());
            assert_eq!(rc, FfiStatus::Ok.code(), "{}", last_error());
            assert!(read_scored(out).is_empty());
            assert_eq!(ffi_close_scored_results(out), FfiStatus::Ok.code());
            assert_eq!(ffi_close_segment(sh), FfiStatus::Ok.code());
            assert_eq!(ffi_close_vectors(vh), FfiStatus::Ok.code());
            assert_eq!(ffi_close_directory(dir), FfiStatus::Ok.code());
        }

        /// Every caller mistake at this boundary is a typed status with a
        /// retrievable message, never a panic and never a silent wrong
        /// answer: a bad handle on either side, a mismatched pair of handles,
        /// a null out pointer, an unknown `Occur` tag, and a clause count past
        /// `maxClauseCount`.
        #[test]
        fn the_filtered_entry_points_reject_every_caller_mistake_by_status() {
            let m = M::load();
            let dir = open_dir();
            let vh = open_vectors(&m, dir);
            let sh = open_segment(&m, dir, m.int("max_doc"));
            let field = m.get("f0.name").to_string();
            let target = floats(&m, "q.f0.0.vec");
            let ok = Clauses::one(m.get("selective_field"), m.get("selective_term"));

            // An unknown vectors handle.
            let (rc, _) = search_float(vh + 12345, sh, &field, &target, 10, &ok);
            assert_eq!(rc, FfiStatus::InvalidHandle.code());
            assert!(last_error().contains("unknown or already-closed"));

            // An unknown filter-segment handle.
            let (rc, _) = search_float(vh, sh + 12345, &field, &target, 10, &ok);
            assert_eq!(rc, FfiStatus::InvalidHandle.code());
            let e = last_error();
            assert!(e.contains("segment handle"), "{e}");

            // A pair of handles that do not describe the same segment,
            // caught by `maxDoc`. The mismatch is staged on the *vectors*
            // side deliberately: `ffi_open_segment` cross-checks `maxDoc`
            // against the term dictionary's own metadata and refuses a wrong
            // one outright, so a mis-described segment handle cannot even be
            // constructed -- which is worth knowing, and is exactly why this
            // check has to exist on the side that has no such cross-check.
            let other_vh = open_vectors_with_max_doc(&m, dir, m.int("max_doc") - 1);
            let (rc, _) = search_float(other_vh, sh, &field, &target, 10, &ok);
            assert_eq!(rc, FfiStatus::InvalidArgument.code());
            let e = last_error();
            assert!(e.contains("not the vector segment's"), "{e}");
            assert_eq!(ffi_close_vectors(other_vh), FfiStatus::Ok.code());

            // A null out pointer, on both encodings.
            let rc = unsafe {
                ffi_knn_float_vector_search_filtered(
                    vh,
                    sh,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    target.as_ptr(),
                    target.len(),
                    10,
                    0,
                    SIMILARITY_FROM_FIELD,
                    0,
                    ok.occurs.as_ptr(),
                    ok.kinds.as_ptr(),
                    ok.fields.as_ptr(),
                    ok.field_lens.as_ptr(),
                    ok.terms.as_ptr(),
                    ok.term_lens.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    1,
                    0,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(rc, FfiStatus::NullPointer.code());
            let byte_field = m.get("f1.name").to_string();
            let byte_target = bytes(&m, "q.f1.0.vec");
            let rc = unsafe {
                ffi_knn_byte_vector_search_filtered(
                    vh,
                    sh,
                    byte_field.as_ptr() as *const c_char,
                    byte_field.len(),
                    byte_target.as_ptr(),
                    byte_target.len(),
                    10,
                    0,
                    SIMILARITY_FROM_FIELD,
                    0,
                    ok.occurs.as_ptr(),
                    ok.kinds.as_ptr(),
                    ok.fields.as_ptr(),
                    ok.field_lens.as_ptr(),
                    ok.terms.as_ptr(),
                    ok.term_lens.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    1,
                    0,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(rc, FfiStatus::NullPointer.code());

            // A null query pointer with a non-zero length.
            let rc = unsafe {
                ffi_knn_float_vector_search_filtered(
                    vh,
                    sh,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    std::ptr::null(),
                    16,
                    10,
                    0,
                    SIMILARITY_FROM_FIELD,
                    0,
                    ok.occurs.as_ptr(),
                    ok.kinds.as_ptr(),
                    ok.fields.as_ptr(),
                    ok.field_lens.as_ptr(),
                    ok.terms.as_ptr(),
                    ok.term_lens.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    1,
                    0,
                    &mut 0u64 as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::NullPointer.code());

            // An unknown `Occur` tag, rejected by the shared clause decoder.
            let mut bad = Clauses::one(m.get("selective_field"), m.get("selective_term"));
            bad.occurs[0] = 9;
            let (rc, _) = search_float(vh, sh, &field, &target, 10, &bad);
            assert_eq!(rc, FfiStatus::InvalidArgument.code());
            let e = last_error();
            assert!(e.contains("unknown Occur tag"), "{e}");

            // More clauses than `IndexSearcher.maxClauseCount`, rejected
            // before any array is dereferenced.
            let huge = Clauses {
                occurs: vec![crate::query::OCCUR_FILTER; 2],
                kinds: vec![crate::query::CLAUSE_KIND_TERM; 2],
                ..Clauses::none()
            };
            let mut out: u64 = 0;
            let rc = unsafe {
                ffi_knn_byte_vector_search_filtered(
                    vh,
                    sh,
                    byte_field.as_ptr() as *const c_char,
                    byte_field.len(),
                    byte_target.as_ptr(),
                    byte_target.len(),
                    10,
                    0,
                    SIMILARITY_FROM_FIELD,
                    0,
                    huge.occurs.as_ptr(),
                    huge.kinds.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    usize::MAX,
                    0,
                    &mut out as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::InvalidArgument.code());
            let e = last_error();
            assert!(e.contains("maxClauseCount"), "{e}");

            // A negative `minimumNumberShouldMatch`.
            let mut out: u64 = 0;
            let rc = unsafe {
                ffi_knn_float_vector_search_filtered(
                    vh,
                    sh,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    target.as_ptr(),
                    target.len(),
                    10,
                    0,
                    SIMILARITY_FROM_FIELD,
                    0,
                    ok.occurs.as_ptr(),
                    ok.kinds.as_ptr(),
                    ok.fields.as_ptr(),
                    ok.field_lens.as_ptr(),
                    ok.terms.as_ptr(),
                    ok.term_lens.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    1,
                    -1,
                    &mut out as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::InvalidArgument.code());
            let e = last_error();
            assert!(e.contains("is negative"), "{e}");

            // And a query-level mistake is still an argument error, not a
            // decode error: `k = 0` is Java's own rejection.
            let (rc, _) = search_float(vh, sh, &field, &target, 0, &ok);
            assert_eq!(rc, FfiStatus::InvalidArgument.code());
            let e = last_error();
            assert!(e.contains("k must be at least 1"), "{e}");
            // As is an unknown field name.
            let (rc, _) = search_float(vh, sh, "no_such_field", &target, 10, &ok);
            assert_eq!(rc, FfiStatus::InvalidArgument.code());

            assert_eq!(ffi_close_segment(sh), FfiStatus::Ok.code());
            assert_eq!(ffi_close_vectors(vh), FfiStatus::Ok.code());
            assert_eq!(ffi_close_directory(dir), FfiStatus::Ok.code());
        }

        /// A filter that accepts everything is *not* the same call as no
        /// filter at all -- Java drops a `MatchAllDocsQuery` filter in
        /// `rewrite` and takes the unfiltered path, where the cost heuristic
        /// and the `visitedLimit` cap do not apply. Both are exercised here,
        /// and the point of the test is that the ABI's unfiltered entry point
        /// remains the way to ask for Java's unfiltered path.
        #[test]
        fn the_unfiltered_entry_point_is_still_javas_unfiltered_path() {
            let m = M::load();
            let dir = open_dir();
            let vh = open_vectors(&m, dir);
            let sh = open_segment(&m, dir, m.int("max_doc"));
            let field = m.get("f0.name").to_string();
            let k = m.int("k") as usize;
            for q in 0..m.int("q.f0.count") {
                let target = floats(&m, &format!("q.f0.{q}.vec"));
                let (rc, out) =
                    super::search_float(vh, &field, &target, k, 0, SIMILARITY_FROM_FIELD);
                assert_eq!(rc, FfiStatus::Ok.code(), "{}", last_error());
                assert_hits_match(
                    &read_scored(out),
                    &hits(&m, &format!("q.f0.{q}.hnsw")),
                    &format!("q.f0.{q}.hnsw"),
                );
                assert_eq!(ffi_close_scored_results(out), FfiStatus::Ok.code());
            }
            assert_eq!(ffi_close_segment(sh), FfiStatus::Ok.code());
            assert_eq!(ffi_close_vectors(vh), FfiStatus::Ok.code());
            assert_eq!(ffi_close_directory(dir), FfiStatus::Ok.code());
        }
    }
}
