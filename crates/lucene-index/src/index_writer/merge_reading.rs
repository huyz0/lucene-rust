//! The reading half of a merge: [`IndexingConfig::run_merge_reading`] opens every
//! source segment of a [`MergePlan`] (raw files, deletes, field infos, doc
//! values), opens a reader of each format over those files, hands them to
//! [`merge::merge_segments_mapped`] as [`merge::MergeSource`]s, and finishes
//! the merged segment (compound file, warmer).
//!
//! The stages below run in the order Java's `SegmentMerger` needs them, and
//! the order is part of the behaviour: which corruption is reported first
//! depends on it.

use super::*;

/// Raw `.tim`/`.tip`/`.tmd`/`.doc` bytes for a source that has
/// postings, plus its `.pos`/`.pay` when the segment has them --
/// `None` when that source's `.si` lists no `.tim` file.
///
/// A positional field always has a `.pos`; `.pay` exists only when
/// some term's `total_term_freq` spans a full 256-position block (see
/// `postings::read_positions`), so it is separately optional.
type RawPostingsFiles = Option<RawPostings>;
struct RawPostings {
    /// The terms dictionary's files, in
    /// `per_field_postings::dictionary_extensions` order for the
    /// suffix's format (`.tim`/`.tip`/`.tmd` for a block tree).
    dictionary: Vec<Input>,
    /// `None` for a format keeping its own postings (`RAMOnly`).
    doc: Option<Input>,
    pos: Option<Input>,
    pay: Option<Input>,
    /// The codec suffix the files were written under -- the
    /// segment's own postings format, which for a segment an older
    /// Lucene wrote is not this writer's (`Lucene90_0`, ...).
    suffix: String,
    /// `PerFieldPostingsFormat`: the segment's further postings
    /// formats, when its fields were routed to more than one. Empty
    /// for the usual segment.
    other_groups: Vec<RawPostings>,
}
/// Raw `.tvd`/`.tvx`/`.tvm` bytes for a source that has term vectors
/// -- `None` when that source's `.si` lists no `.tvd` file.
// `.tvd`, `.tvx`, `.tvm` and their segment suffix.
type RawTermVectorFiles = Option<(Input, Input, Input, String)>;
/// Raw `.nvm`/`.nvd` bytes for a source that has norms.
type RawNormsFiles = Option<(Input, Input)>;
/// One current-format `PerFieldKnnVectorsFormat` instance of a
/// source: its raw `.vec`/`.vemf`, its `.vem`/`.vex` graph when it
/// has one, and its `.vemq`/`.veq` codes when it is a
/// scalar-quantized one.
struct RawVectorGroup {
    suffix: String,
    /// `Lucene104ScalarQuantizedVectorsFormat`: the one format whose
    /// centroid a merge sees (`getCentroid`).
    flat_quantized: bool,
    vec: Input,
    vemf: Input,
    graph: Option<(Input, Input)>,
    quantized: Option<(Input, Input)>,
    /// A `IvfVectorsFormat` group's `.ivm`, `.ivp`, `.ivv`.
    ivf: Option<crate::ivf_vectors::IvfInputs>,
}
type RawVectorFiles = Vec<RawVectorGroup>;
/// A retired (9.0-9.8) HNSW format's `.vem`/`.vec`/`.vex` triple,
/// with the format and codec suffix it was written under.
type RawRetiredVectorFiles = Option<(RetiredHnswFormat, String, Input, Input, Input)>;
/// One retired quantized per-field group (Lucene 9.9-10.3's
/// `Lucene99(Hnsw)ScalarQuantizedVectorsFormat`,
/// `Lucene102(Hnsw)BinaryQuantizedVectorsFormat`): format, codec
/// suffix, `.vemf`, `.vec`, the quantized meta and data, and the
/// `.vem`/`.vex` graph of an HNSW wrapper.
struct RawQuantizedVectorFiles {
    format: QuantizedFormat,
    suffix: String,
    vemf: Input,
    vec: Input,
    quantized_meta: Input,
    quantized_data: Input,
    graph: Option<(Input, Input)>,
}

struct OpenedSegment {
    sci: SegmentCommitInfo,
    fdt: Input,
    fdx: Input,
    fdm: Input,
    /// The stored-fields files' segment suffix (a `CompressingCodec`
    /// format's, or empty).
    stored_suffix: String,
    live_docs: Option<FixedBitSet>,
    /// The documents whose postings for the pruned field survive --
    /// see [`SoftDeletesRetention::prune_postings_field`].
    postings_live: Option<FixedBitSet>,
    /// The pruned recovery-source field and who keeps it -- see
    /// [`SoftDeletesRetention::prune_recovery_source_field`].
    recovery_source: Option<(i32, Option<FixedBitSet>)>,
    /// The documents and fields rewritten to sentinels -- see
    /// [`SoftDeletesRetention::seq_no_sentinel`].
    seq_no_floor: Option<SentinelDocs>,
    postings: RawPostingsFiles,
    term_vectors: RawTermVectorFiles,
    doc_values: SourceDocValueColumns,
    norms: RawNormsFiles,
    vectors: RawVectorFiles,
    retired_vectors: RawRetiredVectorFiles,
    quantized_vectors: Vec<RawQuantizedVectorFiles>,
    /// Raw `.kdm`/`.kdi`/`.kdd`, when the segment has points.
    points: Option<(Input, Input, Input)>,
    /// This source's fields under **its own numbers** -- see where it
    /// is built.
    field_infos: Vec<FieldInfo>,
    index_sort: Option<Vec<segment_info::IndexSortField>>,
    has_blocks: bool,
    /// Whether the source's `.si` records
    /// [`segment_writer::LZ4_HC_ATTRIBUTE`].
    lz4_hc: bool,
    /// This source's `SegmentInfo.minVersion` -- Java's
    /// `LeafMetaData.minVersion()`, which `SegmentMerger` folds into
    /// the merged segment's.
    min_version: Option<LuceneVersion>,
    /// This source's own `SegmentInfo::files`, for
    /// [`merge::check_format_coverage`] -- the segment's own claim
    /// about which formats it has, which is the same set
    /// `IndexFileDeleter` reference-counts.
    files: Vec<String>,
}

type OpenedPostings<'a> = Option<(
    lucene_codecs::blocktree::BlockTreeFields,
    lucene_codecs::postings::DocInput<'a>,
    Option<lucene_codecs::postings::PosInput<'a>>,
    Option<lucene_codecs::postings::PayInput<'a>>,
)>;

type OpenedVectorGroup<'a> = (
    vectors::FlatVectorsReader<'a>,
    Option<hnsw_vectors::HnswVectorsReader<'a>>,
    Option<lucene_codecs::scalar_quantized_vectors::ScalarQuantizedVectorsReader<'a>>,
    Option<&'a crate::ivf_vectors::IvfInputs>,
);

/// What the merge hooks hand back: the reorder map (new doc to old doc) and
/// the diagnostics to record on the merged segment.
type MergeHookOutput = (Option<Vec<i32>>, Vec<(String, String)>);

impl IndexingConfig {
    /// Merge `plan`'s sources into one segment written under `dir`, reading
    /// every source through its own `.si`, field numbers and codec files.
    pub(super) fn run_merge_reading(
        &self,
        dir: &dyn Directory,
        plan: &MergePlan,
    ) -> Result<MergeOutcome> {
        let mut opened = Vec::with_capacity(plan.names.len());
        for (name, sci) in plan.names.iter().zip(&plan.sources) {
            opened.push(self.open_merge_source(dir, plan, name, sci.clone())?);
        }

        // `IndexWriter.validateIndexSort`: every source must agree, and a
        // merge of segments that disagree is refused rather than silently
        // producing an unsorted segment (or one whose `.si` lies).
        let merge_sort = opened.first().and_then(|o| o.index_sort.clone());

        let (reorder, hook_diagnostics) =
            self.apply_merge_hooks(dir, plan, &mut opened, merge_sort.as_deref())?;
        check_index_sorts_agree(&opened, merge_sort.as_deref())?;

        let readers: Vec<_> = opened
            .iter()
            .map(|o| {
                stored_fields::open(&o.fdt, &o.fdx, &o.fdm, &o.sci.segment_id, &o.stored_suffix)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // `IndexWriter.mergeMiddle`: `if (merger.shouldMerge()) merger.merge();`
        // -- `SegmentMerger.shouldMerge()` is `segmentInfo.maxDoc() > 0`, and
        // that `maxDoc` is the sum of the readers' *live* document counts.
        // When it is zero, Java writes **nothing at all** and goes straight to
        // `commitMerge`, whose `allDeleted` branch drops the merged segment and
        // still removes every source from the commit ("Merge would produce a
        // 0-doc segment, so we do nothing except commit the merge to remove all
        // the 0-doc segments that we merged").
        //
        // Publishing the empty result would leave a real zero-document segment
        // that every later open, merge and `CheckIndex` pays for, and a
        // `.si`/`.fnm`/`.fdt` set nothing will ever read.
        // Placed exactly where Java's is: after the readers exist and before
        // the merge writes anything. It is deliberately *not* hoisted above
        // the `opened` loop, even though each source's live count is
        // `si.doc_count - sci.del_count` and both are known there. Hoisting it
        // would skip the loop's cross-checks -- the `.fdm`-against-`.si`
        // `maxDoc` agreement, the `.liv` parse, `validate_index_sort` and
        // `check_format_coverage` -- so a corrupt or unmergeable source would
        // be *silently dropped from the commit* instead of reported. Refusing
        // to look before deciding to throw the sources away is the wrong
        // trade; the cost is reading files for a merge that will not happen,
        // which only occurs when every source is fully deleted.
        let live_doc_count: usize = opened
            .iter()
            .zip(readers.iter())
            .map(|(o, reader)| match &o.live_docs {
                Some(bits) => bits.cardinality(),
                None => usize::try_from(reader.max_doc()).unwrap_or(0),
            })
            .sum();
        if live_doc_count == 0 {
            return Ok(MergeOutcome::AllDeleted);
        }

        let postings_field_infos = postings_field_infos(&opened);
        let combined_postings = combine_group_postings(&opened, &readers, &postings_field_infos)?;
        let opened_postings =
            open_source_postings(&opened, &readers, &postings_field_infos, &combined_postings)?;
        let per_source_postings =
            source_postings(plan, &opened, &opened_postings, &postings_field_infos);
        let opened_term_vectors = open_term_vectors(&opened)?;
        let doc_values = SourceDocValueLists::new(&opened);
        let opened_norms = open_norm_metas(&opened)?;
        let per_source_norms = source_norms(&opened, &opened_norms);
        let opened_vectors = OpenedVectors::open(&opened)?;
        let per_source_vectors = opened_vectors.sources();
        let opened_points = open_points(&opened)?;
        let per_source_points = source_points(&opened, &opened_points);

        // Indexed rather than zipped: eight parallel `Vec`s in one `zip`
        // chain is a tuple nobody can read, and the index is the source's own
        // identity anyway.
        let sources: Vec<merge::MergeSource> = (0..opened.len())
            .map(|i| merge::MergeSource {
                field_infos: &opened[i].field_infos,
                reader: &readers[i],
                live_docs: opened[i].live_docs.as_ref(),
                numeric_doc_values: &doc_values.numeric[i],
                binary_doc_values: &doc_values.binary[i],
                sorted_doc_values: &doc_values.sorted[i],
                sorted_numeric_doc_values: &doc_values.sorted_numeric[i],
                sorted_set_doc_values: &doc_values.sorted_set[i],
                norms: &per_source_norms[i],
                term_vectors: opened_term_vectors[i].as_ref(),
                postings: &per_source_postings[i],
                points: &per_source_points[i],
                vectors: per_source_vectors[i].as_ref(),
                // Both read straight off this source's own `.si`, exactly as
                // `SegmentReader.getMetaData()` builds its `LeafMetaData`.
                min_version: opened[i].min_version,
                has_blocks: opened[i].has_blocks,
                pruned_field: opened[i].recovery_source.as_ref().map(|(number, keep)| {
                    merge::PrunedField {
                        field_number: *number,
                        keep: keep.as_ref(),
                    }
                }),
                numeric_floor: opened[i]
                    .seq_no_floor
                    .as_ref()
                    .map(|(docs, fields)| merge::SentinelValues { docs, fields }),
            })
            .collect();

        self.check_sources_cover_formats(plan, &opened, &sources, &opened_postings)?;

        let per_tier_keys = merge_sort_keys(&opened, &readers, merge_sort.as_deref())?;
        let per_tier_key_slices: Vec<Vec<&[Option<i64>]>> = per_tier_keys
            .iter()
            .map(|per_source| per_source.iter().map(|k| k.as_slice()).collect())
            .collect();
        let sort_specs: Vec<merge::MergeSortKeySpec<'_>> = merge_sort
            .iter()
            .flatten()
            .zip(&per_tier_key_slices)
            .map(|(tier, keys)| merge::MergeSortKeySpec {
                sort: tier,
                per_source_keys: keys,
            })
            .collect();

        let merged = merge::merge_segments_mapped(
            dir,
            &sources,
            (!sort_specs.is_empty()).then_some(sort_specs.as_slice()),
            &self.merge_options(plan, &opened, reorder, hook_diagnostics),
            &plan.merged_name,
            plan.merged_id,
            &self.codec_name,
            self.lucene_version,
        )?;
        let merged = self.finish_merged_segment(dir, plan, merged)?;

        Ok(MergeOutcome::Merged {
            compound_decided: plan.early_compound.is_some(),
            source_live: opened.iter().map(|o| o.live_docs.clone()).collect(),
            merged: Box::new(merged),
        })
    }

    /// Open one source segment of a merge: its `.si`, the files of every
    /// format it has, its deletes (with soft-deletes retention applied) and
    /// its fields under its own numbers.
    fn open_merge_source(
        &self,
        dir: &dyn Directory,
        plan: &MergePlan,
        name: &str,
        sci: SegmentCommitInfo,
    ) -> Result<OpenedSegment> {
        // The `.si` is opened *before* anything is sized by a document
        // count, because it is the authority on that count -- Java's
        // `SegmentMerger` works from `SegmentReader.maxDoc()`, which is
        // `SegmentInfo.maxDoc()`, never from the stored-fields file. It is
        // also what says whether the segment is compound.
        let si_bytes = dir.open(&format!("{name}.si"))?;
        let si = segment_info::parse_for_commit(&si_bytes, &sci)?;

        // `SegmentCoreReaders`: a compound segment -- every segment real
        // Lucene flushes by default -- keeps its codec files inside
        // `.cfs`, and its `.si` lists only `.cfs`/`.cfe`/`.si`. Every read
        // below goes through `seg_dir` and every "does it have format X"
        // test through `seg_files`, which name the members; reading the
        // loose names instead finds no stored fields at all, and testing
        // `si.files` finds no postings, doc values or points -- which the
        // merge would then drop without a word.
        let compound = if si.is_compound_file {
            Some(CompoundReader::open(dir, name, &sci.segment_id)?)
        } else {
            None
        };
        let seg_dir: &dyn Directory = match &compound {
            Some(c) => c,
            None => dir,
        };
        let seg_files: Vec<String> = match &compound {
            Some(c) => si
                .files
                .iter()
                .filter(|f| !f.ends_with(".cfs") && !f.ends_with(".cfe"))
                .cloned()
                .chain(c.member_files())
                .collect(),
            None => si.files.clone(),
        };

        let (fdt, fdx, fdm, stored_suffix) =
            open_stored_fields_files(seg_dir, &seg_files, name, &sci, &si)?;
        let live_docs = read_live_docs(seg_dir, name, &sci, &si)?;
        let postings = open_raw_postings(seg_dir, &seg_files, name)?;
        let term_vectors = open_raw_term_vectors(seg_dir, &seg_files, name)?;

        // `IndexWriter.readFieldInfos(SegmentCommitInfo)`: the
        // generational `.fnm` when this segment has had a doc-values
        // update, the base one otherwise. Only the *location* of each
        // column is taken from it -- the merged schema stays this
        // writer's own `self.fields`, whose `doc_values_gen` is `-1`,
        // because the merge folds every generation back into one base
        // column.
        let current_infos =
            crate::field_updates::read_current_field_infos(seg_dir, &sci, &seg_files)?;
        per_field_postings::validate_routed_groups(&current_infos.fields, &seg_files)
            .map_err(|message| Error::Store(lucene_store::Error::Corrupted(message)))?;
        let own_field_infos = self.source_field_infos(&current_infos);
        let doc_values = read_source_doc_values(seg_dir, &sci, &seg_files, &current_infos)?;

        let norms = open_raw_norms(seg_dir, &seg_files, name)?;
        let retired_vectors = open_raw_retired_vectors(seg_dir, &seg_files, name)?;
        let quantized_vectors = open_raw_quantized_vectors(seg_dir, &seg_files, name)?;
        // Every current-format `PerFieldKnnVectorsFormat` instance
        // (`FieldsReader` opens one reader per suffix the `.fnm` names).
        let vectors = if retired_vectors.is_none() {
            open_raw_vector_groups(seg_dir, &seg_files, name, &sci.segment_id)?
        } else {
            Vec::new()
        };
        let points = open_raw_points(seg_dir, &seg_files, name)?;

        // `SoftDeletesRetentionMergePolicy`: a soft-deleted document
        // the retention policy no longer needs is dropped like a hard
        // delete; one it still needs is carried over, soft-deleted.
        let (live_docs, postings_live) = match &plan.retention {
            Some(retention) => apply_soft_deletes_retention(
                live_docs,
                si.doc_count,
                &current_infos,
                &doc_values,
                retention,
            )?,
            None => (live_docs, None),
        };
        let recovery_source = match &plan.retention {
            Some(retention) => {
                recovery_source_pruning(si.doc_count, &current_infos, &doc_values, retention)?
            }
            None => None,
        };
        let seq_no_floor = match &plan.retention {
            Some(retention) => {
                seq_no_sentinels(si.doc_count, &current_infos, &doc_values, retention)?
            }
            None => None,
        };

        // Computed before the push, because the struct literal moves
        // `sci` in its first field.
        let all_files = sci.files(&seg_files);
        let opened_segment = OpenedSegment {
            sci,
            fdt,
            fdx,
            fdm,
            stored_suffix,
            live_docs,
            postings_live,
            recovery_source,
            seq_no_floor,
            postings,
            term_vectors,
            doc_values,
            norms,
            vectors,
            retired_vectors,
            quantized_vectors,
            points,
            field_infos: own_field_infos,
            index_sort: si.index_sort.clone(),
            has_blocks: si.has_blocks,
            lz4_hc: segment_writer::has_lz4_hc_attribute(&si.attributes),
            min_version: si.min_version,
            // `SegmentCommitInfo::files`, not `si.files`: a generational
            // `.dvm`/`.dvd` is never listed in the `.si` (it did not
            // exist when the `.si` was written), so a gate reading the
            // `.si` alone would be blind to exactly the files an update
            // round produced.
            files: all_files,
        };
        Ok(opened_segment)
    }

    /// `mergeMiddle`: `merge.wrapForMerge(reader)` on every source, then
    /// `merge.reorder(mergedView)` -- unless an index sort is configured, or
    /// blocks cannot be identified by a parent field. Returns the reorder map
    /// and the hooks' diagnostics.
    fn apply_merge_hooks(
        &self,
        dir: &dyn Directory,
        plan: &MergePlan,
        opened: &mut [OpenedSegment],
        merge_sort: Option<&[segment_info::IndexSortField]>,
    ) -> Result<MergeHookOutput> {
        let mut reorder: Option<Vec<i32>> = None;
        let mut hook_diagnostics: Vec<(String, String)> = Vec::new();
        if let Some((hooks, infos)) = &plan.hooks {
            // A source without a parent field is reordered whole only when
            // it carries the hooks' block root field.
            let root = hooks.block_root_field();
            let blocks_without_parent = opened.iter().any(|o| o.has_blocks)
                && opened.iter().any(|o| {
                    !o.field_infos.iter().any(|f| {
                        f.parent_field
                            || root.is_some_and(|r| {
                                f.name == r
                                    && f.doc_values_type
                                        == lucene_codecs::field_infos::DocValuesType::Numeric
                            })
                    })
                });
            let may_reorder =
                self.index_sort.is_none() && merge_sort.is_none() && !blocks_without_parent;
            let prepared = hooks
                .prepare(dir, infos, may_reorder)
                .map_err(Error::MergeHook)?;
            if prepared.live_docs.len() != opened.len() {
                return Err(Error::MergeHook(format!(
                    "wrapForMerge returned {} readers for {} sources",
                    prepared.live_docs.len(),
                    opened.len()
                )));
            }
            // The hooks read the sources with their hard deletes only, so a
            // document they keep may still be one soft-deletes retention
            // drops: a document is carried over when both keep it.
            // A live-docs length other than the source's `maxDoc` is caught
            // by the merge itself (`LiveDocsLengthMismatch`).
            for (o, live) in opened.iter_mut().zip(prepared.live_docs) {
                let Some(bits) = live else {
                    continue;
                };
                if let Some(postings_live) = o.postings_live.as_mut() {
                    for doc in 0..bits.len().min(postings_live.len()) {
                        if !bits.get(doc) {
                            postings_live.clear(doc);
                        }
                    }
                }
                match o.live_docs.as_mut() {
                    Some(own) if own.len() == bits.len() => {
                        for doc in 0..bits.len() {
                            // FBS: `doc < bits.len() == own.len()`.
                            if !bits.get(doc) {
                                own.clear(doc);
                            }
                        }
                    }
                    _ => o.live_docs = Some(bits),
                }
            }
            reorder = may_reorder.then_some(prepared.new_to_old).flatten();
            hook_diagnostics = prepared.diagnostics;
        }
        Ok((reorder, hook_diagnostics))
    }

    /// One source's fields under **its own numbers**, with this writer's
    /// schema laid over the fields it declares.
    fn source_field_infos(
        &self,
        current_infos: &lucene_codecs::field_infos::FieldInfos,
    ) -> Vec<FieldInfo> {
        // `SegmentMerger` reads every source through that source's own
        // `FieldInfos`: a segment real Lucene wrote numbers its fields
        // its own way, and relabelling its stored fields, term
        // dictionary or doc values through this writer's numbers would
        // swap data between fields (or fail, where a number names
        // nothing here). So the **numbers** are always the source's.
        // The **schema** of a field this writer declares is this
        // writer's: its own flushes zero a feature a segment happened to
        // hold no data for (`fields_with_per_field_attributes`), where
        // Java would leave the field out, and those per-segment gaps must
        // not read as a schema disagreement. A field this writer does not
        // declare keeps the source's `FieldInfo` whole. Generations are
        // folded away: the merged column is a base one.
        //
        // `store_payloads` is the source's too. It is not schema but a
        // fact about one segment's bytes (`FieldInfo.setStorePayloads`,
        // set by the first payload a segment's documents give the field),
        // and it decides how that segment's `.pos` is decoded: reading a
        // segment with payloads as one without, or the reverse, decodes
        // garbage positions. The merged field ORs it across the sources
        // (`merge::reconcile_field_numbers`), as Java's does.
        current_infos
            .fields
            .iter()
            .map(|f| {
                let mut info = match self.fields.iter().find(|w| w.name == f.name) {
                    Some(declared) => FieldInfo {
                        number: f.number,
                        store_payloads: f.store_payloads,
                        // Per-field readers enumerate routed fields even when
                        // a prior merge removed their final term. Routing is
                        // source segment state, not the writer's schema.
                        attributes: f.attributes.clone(),
                        // Point dimensions belong to the source segment, including
                        // a declared shape with no surviving BKD entry.
                        point_dimension_count: f.point_dimension_count,
                        point_index_dimension_count: f.point_index_dimension_count,
                        point_num_bytes: f.point_num_bytes,
                        ..declared.clone()
                    },
                    None => f.clone(),
                };
                info.doc_values_gen = -1;
                info
            })
            .collect()
    }

    /// **The merge-completeness gate.** Every format a source's own `.si`
    /// lists files for must be a format the merge actually opened onto that
    /// source's `MergeSource` -- and every extension in that `.si` must be
    /// one some `merge::SegmentFormat` claims.
    ///
    /// A format this stage forgets to open is a format the merge drops, and
    /// the merged segment is well-formed, checksummed and `CheckIndex`-clean
    /// either way. Refusing the merge is strictly better than performing
    /// it, because the loss is otherwise unobservable.
    ///
    /// See `merge::check_format_coverage` for why a *new* format cannot
    /// slip past it: an unclaimed extension is an error, and satisfying
    /// that error means adding a `SegmentFormat` variant whose two
    /// exhaustive `match`es then force the merge to open it.
    fn check_sources_cover_formats(
        &self,
        plan: &MergePlan,
        opened: &[OpenedSegment],
        sources: &[merge::MergeSource<'_>],
        opened_postings: &[OpenedPostings<'_>],
    ) -> Result<()> {
        let source_names: Vec<&str> = plan.names.iter().map(|s| s.as_str()).collect();
        // The codec's own files are written afresh for the merged segment.
        let codec_extensions = self
            .codec
            .as_deref()
            .and_then(|c| c.segment_files_writer())
            .map_or(&[][..], |w| w.extensions());
        let format_files: Vec<Vec<String>> = opened
            .iter()
            .map(|o| {
                o.files
                    .iter()
                    .filter(|f| {
                        !f.rsplit_once('.')
                            .is_some_and(|(_, ext)| codec_extensions.contains(&ext))
                    })
                    .cloned()
                    .collect()
            })
            .collect();
        let source_files: Vec<&[String]> = format_files.iter().map(Vec::as_slice).collect();
        let postings_opened: Vec<bool> = opened_postings.iter().map(Option::is_some).collect();
        merge::check_format_coverage(&source_names, &source_files, sources, &postings_opened)?;
        Ok(())
    }

    /// The merge's options: this writer's configuration, plus what the plan
    /// and the sources add.
    fn merge_options(
        &self,
        plan: &MergePlan,
        opened: &[OpenedSegment],
        reorder: Option<Vec<i32>>,
        hook_diagnostics: Vec<(String, String)>,
    ) -> merge::MergeOptions {
        merge::MergeOptions {
            hnsw_m: self.hnsw_m,
            hnsw_beam_width: self.hnsw_beam_width,
            postings_formats: self.postings_formats.clone(),
            doc_values_formats: self.doc_values_formats.clone(),
            default_doc_values_format: self.resolved_default_doc_values_format(),
            index_format: self.index_format,
            default_postings_format: self.default_postings_format,
            stored_fields_lz4_high_compression: self.merge_lz4_high_compression
                && !(self.merge_lz4_hc_non_force
                    && matches!(
                        plan.kind,
                        merge::MergeKind::Merge { max_num_segments, .. } if max_num_segments >= 0
                    )),
            ivf_partition_clusters: self.ivf_partition_clusters,
            lz4_high_compression_sources: opened.iter().map(|o| o.lz4_hc).collect(),
            zstd_dictionary_min_bytes: self.zstd_dictionary_min_bytes,
            knn_vectors_formats: self.knn_vectors_formats.clone(),
            codec: self.codec.clone(),
            reorder,
            sorted_numeric_remap: plan.sorted_numeric_remap.clone(),
            kind: plan.kind,
            extra_diagnostics: hook_diagnostics,
            extra_attributes: plan.attributes.clone(),
        }
    }

    /// `mergeMiddle`: the compound file, then
    /// `mergedSegmentWarmer.warm(reader)`, before `commitMerge`, outside
    /// the writer's lock.
    fn finish_merged_segment(
        &self,
        dir: &dyn Directory,
        plan: &MergePlan,
        mut merged: merge::MergedSegment,
    ) -> Result<merge::MergedSegment> {
        if let Some(Some(decider)) = &plan.early_compound {
            if decider.decide(dir, &merged)? {
                before_write(
                    self.codec.as_deref(),
                    lucene_codecs::codec::WritePoint::Compound,
                )?;
                merged.files = segment_writer::convert_to_compound(
                    dir,
                    &merged.info.segment_name,
                    &merged.info.segment_id,
                )?;
            }
        }
        if let Some(warmer) = &self.merged_segment_warmer {
            warmer.warm(dir, &merged.info)?;
        }
        Ok(merged)
    }
}

/// `MultiSorter.sort`'s `IndexSorter.getComparableValues`: each sort tier's
/// key for every document of every source.
fn merge_sort_keys(
    opened: &[OpenedSegment],
    readers: &[stored_fields::StoredFieldsReader<'_>],
    merge_sort: Option<&[segment_info::IndexSortField]>,
) -> Result<Vec<Vec<Vec<Option<i64>>>>> {
    // `MultiSorter.sort`'s `IndexSorter.getComparableValues`: each sort
    // tier's key for every document of every source, read out of the very
    // NUMERIC column the merged segment will carry -- so the order the
    // merge imposes and the column `CheckIndex.testSort` re-derives it
    // from are one fact, exactly as at flush time.
    //
    // A byte-keyed tier (`STRING`, `SortedSetSortField`,
    // `BinarySortField`) is read as each document's bytes and ranked
    // across **all** sources at once -- `StringSorter`'s `OrdinalMap`
    // global ordinals, which is what makes one source's "b" compare
    // equal to another's.
    let mut per_tier_keys: Vec<Vec<Vec<Option<i64>>>> = match merge_sort {
        None => Vec::new(),
        Some(sort) => sort
            .iter()
            .map(|tier| {
                let byte_keyed = !matches!(
                    tier.kind,
                    segment_info::IndexSortKind::Numeric(_)
                        | segment_info::IndexSortKind::SortedNumeric { .. }
                );
                if byte_keyed {
                    let terms = opened
                        .iter()
                        .zip(readers.iter())
                        .map(|(o, reader)| {
                            IndexWriter::read_sort_terms(
                                &o.doc_values,
                                &o.field_infos,
                                tier,
                                reader.max_doc(),
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    return Ok(crate::index_sorter::rank_terms(&terms));
                }
                opened
                    .iter()
                    .zip(readers.iter())
                    .map(|(o, reader)| {
                        IndexWriter::read_sort_keys(
                            &o.doc_values,
                            &o.field_infos,
                            tier,
                            reader.max_doc(),
                        )
                    })
                    .collect::<Result<Vec<Vec<Option<i64>>>>>()
            })
            .collect::<Result<Vec<_>>>()?,
    };
    // `MultiSorter.sort`'s per-reader parent bit set: a source with blocks
    // sorts each document by the key of the parent closing its block
    // (`parents.nextSetBit(docID)`), so blocks move whole; one with blocks
    // and no parent field cannot be sorted (`CorruptIndexException`).
    if merge_sort.is_some() {
        for (i, (o, reader)) in opened.iter().zip(readers.iter()).enumerate() {
            if !o.has_blocks {
                continue;
            }
            let parents =
                source_parents(o.field_infos.as_slice(), &o.doc_values, reader.max_doc())?
                    .ok_or(Error::IndexSortWithBlocksAndNoParentField)?;
            for tier in &mut per_tier_keys {
                // FBS: `doc` walks the source's keys, one per document below
                // `maxDoc`, the set's length; the check keeps a mismatch
                // from panicking.
                explicit::key_of_parent(&mut tier[i], &|doc| {
                    doc < parents.len() && parents.get(doc)
                });
            }
        }
    }
    Ok(per_tier_keys)
}

/// `IndexWriter.validateIndexSort`: a merge of sources that disagree on the
/// index sort is refused.
fn check_index_sorts_agree(
    opened: &[OpenedSegment],
    merge_sort: Option<&[segment_info::IndexSortField]>,
) -> Result<()> {
    if let Some(bad) = opened
        .iter()
        .find(|o| o.index_sort.as_deref() != merge_sort)
    {
        return Err(Error::MergeSortDisagreement {
            expected: segment_info::describe_index_sort(merge_sort),
            found: segment_info::describe_index_sort(bad.index_sort.as_deref()),
            segment: bad.sci.segment_name.clone(),
        });
    }
    Ok(())
}

// Each source's postings-eligible fields, under its own numbers --
// exactly the `IndexOptions` `set_postings_field`/`add_postings_field`
// accept at write time, **including the positional ones**. Narrowing
// this to `Docs`/`DocsAndFreqs` would drop a positional field's
// postings from every merged segment while `describe_written_files`
// left its `index_options` claiming them: an indexed field with no
// registered postings producer, which reads back as having no terms.
// `merge_postings` handles positions, offsets and payloads.
fn postings_field_infos(opened: &[OpenedSegment]) -> Vec<lucene_codecs::field_infos::FieldInfos> {
    opened
        .iter()
        .map(|o| lucene_codecs::field_infos::FieldInfos {
            fields: o
                .field_infos
                .iter()
                .filter(|f| {
                    matches!(
                        f.index_options,
                        IndexOptions::Docs
                                | IndexOptions::DocsAndFreqs
                                // A custom-frequency field's terms live in
                                // the same dictionary; left out, a source
                                // holding them fails to open.
                                | IndexOptions::DocsAndCustomFreqs
                                | IndexOptions::DocsAndFreqsAndPositions
                                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
                    )
                })
                .cloned()
                .collect(),
        })
        .collect()
}

/// A multi-format source's groups, combined once
/// (`PerFieldPostingsFormat.FieldsReader`) into one dictionary over
/// concatenated `.doc`/`.pos`/`.pay` the inputs of [`open_source_postings`]
/// borrow.
fn combine_group_postings(
    opened: &[OpenedSegment],
    readers: &[stored_fields::StoredFieldsReader<'_>],
    postings_field_infos: &[lucene_codecs::field_infos::FieldInfos],
) -> Result<Vec<Option<per_field_postings::CombinedPostings>>> {
    opened
        .iter()
        .zip(readers.iter())
        .zip(postings_field_infos)
        .map(|((o, reader), postings_field_infos)| match &o.postings {
            Some(raw) if !raw.other_groups.is_empty() || is_foreign_postings_group(&raw.suffix) => {
                let groups: Vec<per_field_postings::GroupFiles<'_>> = std::iter::once(raw)
                    .chain(&raw.other_groups)
                    .map(|g| per_field_postings::GroupFiles {
                        suffix: &g.suffix,
                        dictionary: g.dictionary.iter().map(|d| &**d).collect(),
                        doc: g.doc.as_deref(),
                        pos: g.pos.as_deref(),
                        pay: g.pay.as_deref(),
                    })
                    .collect();
                Ok::<_, Error>(Some(per_field_postings::open_groups(
                    &groups,
                    postings_field_infos,
                    &o.sci.segment_id,
                    reader.max_doc(),
                )?))
            }
            _ => Ok(None),
        })
        .collect::<std::result::Result<Vec<_>, Error>>()
}

/// Each source's postings readers: the combined groups of a multi-format
/// source, or its one block-tree dictionary with its `.doc`/`.pos`/`.pay`.
fn open_source_postings<'a>(
    opened: &'a [OpenedSegment],
    readers: &[stored_fields::StoredFieldsReader<'_>],
    postings_field_infos: &[lucene_codecs::field_infos::FieldInfos],
    combined_postings: &'a [Option<per_field_postings::CombinedPostings>],
) -> Result<Vec<OpenedPostings<'a>>> {
    opened
        .iter()
        .zip(readers.iter())
        .zip(postings_field_infos)
        .zip(combined_postings)
        .map(
            |(((o, reader), postings_field_infos), combined)| match (&o.postings, combined) {
                (_, Some(c)) => Ok(Some((
                    c.fields.clone(),
                    lucene_codecs::postings::DocInput::validated(&c.doc),
                    (!c.pos.is_empty())
                        .then(|| lucene_codecs::postings::PosInput::validated(&c.pos)),
                    (!c.pay.is_empty())
                        .then(|| lucene_codecs::postings::PayInput::validated(&c.pay)),
                ))),
                (Some(raw), None) => {
                    let fields = lucene_codecs::blocktree::open(
                        &raw.dictionary[0],
                        &raw.dictionary[1],
                        &raw.dictionary[2],
                        postings_field_infos,
                        &o.sci.segment_id,
                        &raw.suffix,
                        reader.max_doc(),
                    )?;
                    let doc_in = lucene_codecs::postings::DocInput::open(
                        raw.doc.as_deref().ok_or_else(|| {
                            lucene_store::Error::Corrupted(format!(
                                "postings group {} has no .doc",
                                raw.suffix
                            ))
                        })?,
                        &o.sci.segment_id,
                        &raw.suffix,
                    )?;
                    let pos_in = raw
                        .pos
                        .as_ref()
                        .map(|pos| {
                            lucene_codecs::postings::PosInput::open(
                                pos,
                                &o.sci.segment_id,
                                &raw.suffix,
                            )
                        })
                        .transpose()?;
                    let pay_in = raw
                        .pay
                        .as_ref()
                        .map(|pay| {
                            lucene_codecs::postings::PayInput::open(
                                pay,
                                &o.sci.segment_id,
                                &raw.suffix,
                            )
                        })
                        .transpose()?;
                    Ok::<_, Error>(Some((fields, doc_in, pos_in, pay_in)))
                }
                (None, None) => Ok(None),
            },
        )
        .collect::<std::result::Result<Vec<_>, Error>>()
}

/// One `Vec<SourcePostings>` per source, holding every postings-eligible
/// field that source's own term dictionary actually has an entry for.
fn source_postings<'a>(
    plan: &MergePlan,
    opened: &'a [OpenedSegment],
    opened_postings: &'a [OpenedPostings<'a>],
    postings_field_infos: &[lucene_codecs::field_infos::FieldInfos],
) -> Vec<Vec<merge::SourcePostings<'a>>> {
    let prune_field = plan
        .retention
        .as_ref()
        .and_then(|r| r.prune_postings_field.as_deref());
    opened_postings
        .iter()
        .zip(postings_field_infos)
        .zip(opened)
        .map(|((maybe, postings_field_infos), o)| match maybe {
            Some((fields, doc_in, pos_in, pay_in)) => postings_field_infos
                .fields
                .iter()
                .filter_map(|f| {
                    fields
                        .field(&f.name)
                        .map(|field_terms| merge::SourcePostings {
                            field_number: f.number,
                            field_terms,
                            doc_in: Some(doc_in),
                            pos_in: pos_in.as_ref(),
                            pay_in: pay_in.as_ref(),
                            live: o
                                .postings_live
                                .as_ref()
                                .filter(|_| prune_field == Some(f.name.as_str())),
                        })
                })
                .collect(),
            None => Vec::new(),
        })
        .collect()
}

fn open_term_vectors(
    opened: &[OpenedSegment],
) -> Result<Vec<Option<lucene_codecs::term_vectors::TermVectorsReader<'_>>>> {
    opened
        .iter()
        .map(|o| match &o.term_vectors {
            Some((tvd, tvx, tvm, suffix)) => Ok::<_, Error>(Some(
                lucene_codecs::term_vectors::open(tvd, tvx, tvm, &o.sci.segment_id, suffix)?,
            )),
            None => Ok(None),
        })
        .collect::<std::result::Result<Vec<_>, Error>>()
}

/// Every source's doc-values entries, one list per type per source.
struct SourceDocValueLists<'a> {
    numeric: Vec<Vec<merge::SourceNumericDocValues<'a>>>,
    binary: Vec<Vec<merge::SourceBinaryDocValues<'a>>>,
    sorted: Vec<Vec<merge::SourceSortedDocValues<'a>>>,
    sorted_numeric: Vec<Vec<merge::SourceSortedNumericDocValues<'a>>>,
    sorted_set: Vec<Vec<merge::SourceSortedSetDocValues<'a>>>,
}

impl<'a> SourceDocValueLists<'a> {
    fn new(opened: &'a [OpenedSegment]) -> Self {
        // Doc values. A source's `.dvm` entries carry that source's own field
        // numbers, as its `field_infos` do; `merge_segments` maps both to the
        // merged numbers.
        // **Every** doc-values type, not just NUMERIC. `segment_stats` now
        // offers any `.dvd`-bearing segment to the merge policy, and this
        // writer can flush all five types (`collect_dense_column`), so a type
        // opened here and not merged would be silently dropped: the merge
        // writes no column, `merge::describe_written_files` then zeroes the
        // field's `DocValuesType` to keep the `.fnm` honest, and what is left
        // is a valid, `CheckIndex`-clean segment with the data gone.
        // One list per type per source. Each entry names a field and the
        // column its *current* generation lives in, so a source with a
        // doc-values update contributes its updated values rather than the
        // superseded base ones.
        macro_rules! per_source_dv {
            ($ty:ident, $bucket:ident) => {
                opened
                    .iter()
                    .map(|o| {
                        o.doc_values
                            .per_field
                            .iter()
                            .filter_map(|&(field_number, at)| {
                                let (meta, data) = &o.doc_values.columns[at];
                                meta.$bucket
                                    .iter()
                                    .find(|e| e.field_number == field_number)
                                    .map(|entry| merge::$ty {
                                        data: data.as_slice(),
                                        entry: entry.clone(),
                                    })
                            })
                            .collect::<Vec<merge::$ty>>()
                    })
                    .collect::<Vec<_>>()
            };
        }
        let lists = SourceDocValueLists {
            numeric: per_source_dv!(SourceNumericDocValues, numeric),
            binary: per_source_dv!(SourceBinaryDocValues, binary),
            sorted: per_source_dv!(SourceSortedDocValues, sorted),
            sorted_numeric: per_source_dv!(SourceSortedNumericDocValues, sorted_numeric),
            sorted_set: per_source_dv!(SourceSortedSetDocValues, sorted_set),
        };
        // Stated as an assertion rather than left to review: the five lists
        // above are built by one macro over five named `DocValuesMeta`
        // buckets, and a sixth doc-values type would need a sixth line that
        // nothing else would miss.
        debug_assert!(
            opened.iter().all(|o| {
                let total: usize = o
                    .doc_values
                    .columns
                    .iter()
                    .map(|(m, _)| {
                        // ARITH: five in-memory `Vec` lengths; their sum is
                        // bounded by the entries this process holds.
                        #[allow(clippy::arithmetic_side_effects)]
                        {
                            m.numeric.len()
                                + m.binary.len()
                                + m.sorted.len()
                                + m.sorted_numeric.len()
                                + m.sorted_set.len()
                        }
                    })
                    .sum();
                // Every entry the opened columns declare is either reached by
                // one of the five lists or belongs to a field this segment's
                // current schema no longer points at that column for (a
                // superseded base entry for an updated field).
                total >= o.doc_values.per_field.len()
            }),
            "every doc-values entry a source's .dvm declares must reach the merge"
        );
        lists
    }
}

fn open_norm_metas(opened: &[OpenedSegment]) -> Result<Vec<Option<norms::Norms>>> {
    opened
        .iter()
        .map(|o| match &o.norms {
            Some((nvm, _)) => {
                let (_v, meta) = norms::parse_meta(nvm, &o.sci.segment_id, "")?;
                Ok::<_, Error>(Some(meta))
            }
            None => Ok(None),
        })
        .collect::<std::result::Result<Vec<_>, Error>>()
}

fn source_norms<'a>(
    opened: &'a [OpenedSegment],
    opened_norms: &'a [Option<norms::Norms>],
) -> Vec<Vec<merge::SourceNorms<'a>>> {
    opened
        .iter()
        .zip(opened_norms)
        .map(|(o, meta)| match (&o.norms, meta) {
            (Some((_, nvd)), Some(meta)) => meta
                .entries
                .iter()
                .map(|entry| merge::SourceNorms {
                    data: nvd,
                    entry: *entry,
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect()
}

/// Every source's vector readers. The flat store and the graph are opened
/// separately because a segment may legitimately have the first and not the
/// second.
struct OpenedVectors<'a> {
    groups: Vec<Vec<OpenedVectorGroup<'a>>>,
    retired: Vec<Option<RetiredHnswVectorsReader<'a>>>,
    quantized: Vec<Vec<QuantizedVectorsReader<'a>>>,
}

impl<'a> OpenedVectors<'a> {
    fn open(opened: &'a [OpenedSegment]) -> Result<Self> {
        let opened_vector_groups = opened
            .iter()
            .map(|o| {
                o.vectors
                    .iter()
                    .map(|g| {
                        let mut flat = vectors::FlatVectorsReader::open(
                            &g.vemf,
                            &g.vec,
                            &o.sci.segment_id,
                            &g.suffix,
                        )?;
                        flat.resolve_similarities_from(&o.field_infos)?;
                        let graph = match &g.graph {
                            Some((vem, vex)) => Some(hnsw_vectors::HnswVectorsReader::open(
                                vem,
                                vex,
                                &o.sci.segment_id,
                                &g.suffix,
                            )?),
                            None => None,
                        };
                        // Only the graph-less quantized format's centroid is
                        // visible to a merge (`getCentroid`).
                        let quantized = match (&g.quantized, g.flat_quantized) {
                            (Some((vemq, veq)), true) => Some(
                                lucene_codecs::scalar_quantized_vectors::ScalarQuantizedVectorsReader::open(
                                    vemq,
                                    veq,
                                    &o.sci.segment_id,
                                    &g.suffix,
                                )?,
                            ),
                            _ => None,
                        };
                        Ok::<_, Error>((flat, graph, quantized, g.ivf.as_ref()))
                    })
                    .collect::<std::result::Result<Vec<_>, Error>>()
            })
            .collect::<std::result::Result<Vec<_>, Error>>()?;
        // A retired-format source contributes its vectors only: 10.5.0's
        // `IncrementalHnswGraphMerger` reuses a source graph only from an
        // `HnswGraphProvider`, which the 9.x readers are not, so the merged
        // graph is built from scratch over them.
        let opened_retired_vectors: Vec<Option<RetiredHnswVectorsReader>> = opened
            .iter()
            .map(|o| match &o.retired_vectors {
                Some((format, suffix, vem, vec_bytes, vex)) => {
                    Ok::<_, Error>(Some(RetiredHnswVectorsReader::open(
                        *format,
                        vem,
                        vec_bytes,
                        vex,
                        &o.sci.segment_id,
                        suffix,
                    )?))
                }
                None => Ok(None),
            })
            .collect::<std::result::Result<Vec<_>, Error>>()?;
        // A retired quantized group contributes its raw vectors (the
        // `.vec`/`.vemf` beside the codes) and its graph, which is a
        // `Lucene99HnswVectorsFormat` one: Java's merger may start from it as
        // it would from any `HnswGraphProvider`'s.
        let opened_quantized_vectors: Vec<Vec<QuantizedVectorsReader>> = opened
            .iter()
            .map(|o| {
                let infos = lucene_codecs::field_infos::FieldInfos::new(o.field_infos.clone())?;
                o.quantized_vectors
                    .iter()
                    .map(|q| {
                        Ok::<_, Error>(QuantizedVectorsReader::open(
                            q.format,
                            QuantizedFiles {
                                vemf: &q.vemf,
                                vec: &q.vec,
                                quantized_meta: &q.quantized_meta,
                                quantized_data: &q.quantized_data,
                                graph: q.graph.as_ref().map(|(m, x)| (&m[..], &x[..])),
                            },
                            &infos,
                            &o.sci.segment_id,
                            &q.suffix,
                        )?)
                    })
                    .collect::<std::result::Result<Vec<_>, Error>>()
            })
            .collect::<std::result::Result<Vec<_>, Error>>()?;
        Ok(OpenedVectors {
            groups: opened_vector_groups,
            retired: opened_retired_vectors,
            quantized: opened_quantized_vectors,
        })
    }

    fn sources(&'a self) -> Vec<Option<merge::SourceVectors<'a>>> {
        let (opened_vector_groups, opened_retired_vectors, opened_quantized_vectors) =
            (&self.groups, &self.retired, &self.quantized);
        opened_vector_groups
            .iter()
            .zip(opened_retired_vectors)
            .zip(opened_quantized_vectors)
            .map(|((current, retired), quantized)| {
                let mut groups: Vec<merge::VectorGroup> = Vec::new();
                match retired {
                    Some(r) => groups.push(merge::VectorGroup {
                        flat: r.flat(),
                        graph: None,
                        flat_quantized: None,
                        ivf: None,
                    }),
                    None => groups.extend(current.iter().map(|(flat, graph, q, ivf)| {
                        merge::VectorGroup {
                            flat,
                            graph: graph.as_ref(),
                            flat_quantized: q.as_ref(),
                            ivf: *ivf,
                        }
                    })),
                }
                groups.extend(quantized.iter().map(|q| merge::VectorGroup {
                    flat: q.flat(),
                    graph: q.graph(),
                    flat_quantized: None,
                    ivf: None,
                }));
                (!groups.is_empty()).then_some(merge::SourceVectors { groups })
            })
            .collect()
    }
}

fn open_points(
    opened: &[OpenedSegment],
) -> Result<Vec<Option<lucene_codecs::points::PointsReader<'_>>>> {
    opened
        .iter()
        .map(|o| match &o.points {
            Some((kdm, kdi, kdd)) => Ok::<_, Error>(Some(lucene_codecs::points::open(
                kdm,
                kdi,
                kdd,
                &o.sci.segment_id,
                "",
            )?)),
            None => Ok(None),
        })
        .collect::<std::result::Result<Vec<_>, Error>>()
}

/// A `SourcePoints` for every field a source's FieldInfos declares,
/// including legitimate null BKD entries.
fn source_points<'a>(
    opened: &'a [OpenedSegment],
    opened_points: &'a [Option<lucene_codecs::points::PointsReader<'a>>],
) -> Vec<Vec<merge::SourcePoints<'a>>> {
    opened_points
        .iter()
        .zip(opened)
        .map(|(reader, source)| match reader {
            Some(reader) => source
                .field_infos
                .iter()
                .filter(|field| field.point_dimension_count > 0)
                .map(|field| merge::SourcePoints {
                    field_number: field.number,
                    reader,
                })
                .collect(),
            None => Vec::new(),
        })
        .collect()
}

fn open_stored_fields_files(
    seg_dir: &dyn Directory,
    seg_files: &[String],
    name: &str,
    sci: &SegmentCommitInfo,
    si: &segment_info::SegmentInfo,
) -> Result<(Input, Input, Input, String)> {
    let stored_suffix = format_suffix(seg_files, name, "fdt").unwrap_or_default();
    let stored_file = |ext: &str| segment_file_name(name, &stored_suffix, ext);
    let fdt = seg_dir.open(&stored_file("fdt"))?;
    let fdx = seg_dir.open(&stored_file("fdx"))?;
    let fdm = seg_dir.open(&stored_file("fdm"))?;

    // `stored_fields::open` checks only that its own `maxDoc` is
    // non-negative, and `merge_segments` sizes this source's live-id
    // list and doc-id map from it. Left uncrossed-checked, a `.fdm`
    // claiming `i32::MAX` turns a merge into two ~8.6 GB allocations
    // -- an abort, and `open_segment_for_deletes` two thousand lines
    // below already takes the count from the `.si` for exactly this
    // reason. Disagreement is corruption whichever file is right.
    let stored_fields_max_doc =
        stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, &stored_suffix)?.max_doc();
    if stored_fields_max_doc != si.doc_count {
        return Err(Error::SegmentDocCountMismatch {
            segment: name.to_string(),
            si_doc_count: si.doc_count,
            stored_fields_max_doc,
        });
    }
    Ok((fdt, fdx, fdm, stored_suffix))
}

fn read_live_docs(
    seg_dir: &dyn Directory,
    name: &str,
    sci: &SegmentCommitInfo,
    si: &segment_info::SegmentInfo,
) -> Result<Option<FixedBitSet>> {
    Ok(if sci.del_gen >= 0 {
        let liv = seg_dir.open(&deletes::liv_file_name(name, sci.del_gen))?;
        Some(lucene_codecs::live_docs::parse(
            &liv,
            &sci.segment_id,
            sci.del_gen,
            // `usize::try_from` cannot fail: `si.doc_count` is
            // non-negative (`segment_info::parse` rejects otherwise).
            usize::try_from(si.doc_count).unwrap_or(0),
            sci.del_count as usize,
        )?)
    } else {
        None
    })
}

/// The postings files of a source: one group, or one per routed postings
/// format when its fields were routed to several.
fn open_raw_postings(
    seg_dir: &dyn Directory,
    seg_files: &[String],
    name: &str,
) -> Result<RawPostingsFiles> {
    Ok(if per_field_postings::has_postings_files(seg_files) {
        let read_group = |seg: &str, suffix: String| -> Result<RawPostings> {
            let read_optional = |ext: &str| -> Result<Option<Input>> {
                let file = format!("{seg}.{ext}");
                Ok(if seg_files.contains(&file) {
                    Some(seg_dir.open(&file)?)
                } else {
                    None
                })
            };
            let format = per_field_postings::format_of_suffix(&suffix);
            let checksummed = !per_field_postings::is_unchecksummed_dictionary_file(&format!(
                "{seg}.{}",
                per_field_postings::marker_extension(format)
            ));
            let dictionary = per_field_postings::dictionary_extensions(format)
                .iter()
                .map(|ext| {
                    let name = format!("{seg}.{ext}");
                    if !seg_files.contains(&name)
                        && !per_field_postings::dictionary_file_required(format, ext)
                    {
                        return Ok(Input::Owned(Vec::new()));
                    }
                    Ok(seg_dir.open(&name)?)
                })
                .collect::<Result<Vec<_>>>()?;
            let raw = RawPostings {
                dictionary,
                doc: read_optional("doc")?,
                pos: read_optional("pos")?,
                pay: read_optional("pay")?,
                suffix,
                other_groups: Vec::new(),
            };
            // Java: FieldsConsumer.merge checks each source producer's
            // integrity before consuming postings. Metadata is checked
            // on acquisition by Java's reader; native open only checks
            // footer structure, so verify every original group here.
            let metadata = read_optional("psm")?;
            // An empty dictionary entry is a file the format left
            // unwritten (`dictionary_file_required`).
            // A footerless dictionary (`RAMOnly`'s token) is not.
            let written = raw
                .dictionary
                .iter()
                .filter(|d| !d.is_empty() && checksummed);
            for bytes in written.map(|d| &**d).chain(
                [
                    raw.doc.as_deref(),
                    raw.pos.as_deref(),
                    raw.pay.as_deref(),
                    metadata.as_deref(),
                ]
                .into_iter()
                .flatten(),
            ) {
                let footer = bytes
                    .len()
                    .saturating_sub(lucene_store::codec_util::FOOTER_LENGTH);
                lucene_store::codec_util::check_whole_file_footer(bytes, footer)?;
            }
            Ok(raw)
        };
        let suffixes = per_field_postings::group_suffixes(seg_files, name);
        if suffixes.len() > 1 || suffixes.iter().any(|s| is_foreign_postings_group(s)) {
            let mut groups = suffixes
                .into_iter()
                .map(|suffix| read_group(&format!("{name}_{suffix}"), suffix))
                .collect::<Result<Vec<_>>>()?;
            let mut first = groups.remove(0);
            first.other_groups = groups;
            Some(first)
        } else {
            let (seg, suffix) = postings_file_base(seg_files, name);
            Some(read_group(&seg, suffix)?)
        }
    } else {
        None
    })
}

fn open_raw_term_vectors(
    seg_dir: &dyn Directory,
    seg_files: &[String],
    name: &str,
) -> Result<RawTermVectorFiles> {
    Ok(match format_suffix(seg_files, name, "tvd") {
        Some(suffix) => {
            let file = |ext: &str| segment_file_name(name, &suffix, ext);
            let tvd = seg_dir.open(&file("tvd"))?;
            let tvx = seg_dir.open(&file("tvx"))?;
            let tvm = seg_dir.open(&file("tvm"))?;
            Some((tvd, tvx, tvm, suffix))
        }
        None => None,
    })
}

/// Each distinct `.dvm`/`.dvd` column pair a source's current generation
/// points at, read once.
fn read_source_doc_values(
    seg_dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    seg_files: &[String],
    current_infos: &lucene_codecs::field_infos::FieldInfos,
) -> Result<SourceDocValueColumns> {
    let mut columns: Vec<(doc_values::DocValuesMeta, Vec<u8>)> = Vec::new();
    let mut per_field: Vec<(i32, usize)> = Vec::new();
    // Distinct column locations, so a segment's base pair is read
    // once rather than once per field that lives in it.
    let mut seen: Vec<(i64, String, usize)> = Vec::new();
    for (index, field) in current_infos.fields.iter().enumerate() {
        if field.doc_values_type == DocValuesType::None {
            continue;
        }
        let per_field_component = crate::field_updates::per_field_component(
            field,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        );
        let key = (field.doc_values_gen, per_field_component.clone());
        if let Some(&(_, _, at)) = seen
            .iter()
            .find(|(gen, comp, _)| *gen == key.0 && *comp == key.1)
        {
            per_field.push((field.number, at));
            continue;
        }
        let Some((meta, data)) = crate::field_updates::read_current_column(
            seg_dir,
            sci,
            seg_files,
            current_infos,
            index,
            &per_field_component,
        )?
        else {
            // The `.fnm` declares a type but the segment never wrote
            // a column: legitimate, and `merge_segments` writes an
            // all-missing merged column for it the same way
            // `Lucene90DocValuesConsumer.writeValues` does.
            continue;
        };
        columns.push((meta, data.to_vec()));
        let at = columns.len().saturating_sub(1);
        seen.push((key.0, key.1, at));
        per_field.push((field.number, at));
    }
    Ok(SourceDocValueColumns { columns, per_field })
}

fn open_raw_norms(
    seg_dir: &dyn Directory,
    seg_files: &[String],
    name: &str,
) -> Result<RawNormsFiles> {
    Ok(if seg_files.iter().any(|f| f.ends_with(".nvd")) {
        let nvm = seg_dir.open(&format!("{name}.nvm"))?;
        let nvd = seg_dir.open(&format!("{name}.nvd"))?;
        Some((nvm, nvd))
    } else {
        None
    })
}

/// A segment Lucene 9.0-9.8 wrote keeps its vectors in a retired format's
/// `.vem`/`.vec`/`.vex` triple and has no `.vemf`.
fn open_raw_retired_vectors(
    seg_dir: &dyn Directory,
    seg_files: &[String],
    name: &str,
) -> Result<RawRetiredVectorFiles> {
    Ok(match retired_vector_files(seg_files, name) {
        Some((format, suffix)) => {
            let seg = format!("{name}_{suffix}");
            Some((
                format,
                suffix,
                seg_dir.open(&format!("{seg}.vem"))?,
                seg_dir.open(&format!("{seg}.vec"))?,
                seg_dir.open(&format!("{seg}.vex"))?,
            ))
        }
        None => None,
    })
}

/// Every per-field group a retired quantized format wrote, each under its
/// own suffix -- beside each other and beside the current format's group in
/// the same segment.
fn open_raw_quantized_vectors(
    seg_dir: &dyn Directory,
    seg_files: &[String],
    name: &str,
) -> Result<Vec<RawQuantizedVectorFiles>> {
    let mut quantized_vectors = Vec::new();
    for (format, suffix) in quantized_vector_files(seg_files, name) {
        let seg = format!("{name}_{suffix}");
        let (meta_ext, data_ext) = format.quantized_extensions();
        let graph = if format.has_graph() {
            Some((
                seg_dir.open(&format!("{seg}.vem"))?,
                seg_dir.open(&format!("{seg}.vex"))?,
            ))
        } else {
            None
        };
        quantized_vectors.push(RawQuantizedVectorFiles {
            format,
            suffix,
            vemf: seg_dir.open(&format!("{seg}.vemf"))?,
            vec: seg_dir.open(&format!("{seg}.vec"))?,
            quantized_meta: seg_dir.open(&format!("{seg}.{meta_ext}"))?,
            quantized_data: seg_dir.open(&format!("{seg}.{data_ext}"))?,
            graph,
        });
    }
    Ok(quantized_vectors)
}

fn open_raw_vector_groups(
    seg_dir: &dyn Directory,
    seg_files: &[String],
    name: &str,
    segment_id: &[u8; ID_LENGTH],
) -> Result<RawVectorFiles> {
    let mut vectors: RawVectorFiles = Vec::new();
    for suffix in current_vector_suffixes(seg_files, name) {
        let seg = if suffix.is_empty() {
            name.to_string()
        } else {
            format!("{name}_{suffix}")
        };
        let listed = |ext: &str| seg_files.contains(&format!("{seg}.{ext}"));
        let pair = |a: &str, b: &str| -> Result<Option<(Input, Input)>> {
            Ok(if listed(a) {
                Some((
                    seg_dir.open(&format!("{seg}.{a}"))?,
                    seg_dir.open(&format!("{seg}.{b}"))?,
                ))
            } else {
                None
            })
        };
        // A segment can legitimately have the flat pair and no
        // graph files at all (a graph-less format); this writer's
        // HNSW formats always write `.vem`/`.vex` (with
        // `numLevels = 0` below `HNSW_GRAPH_THRESHOLD`).
        vectors.push(RawVectorGroup {
            flat_quantized: suffix
                .rsplit_once('_')
                .and_then(|(f, _)| per_field_knn_vectors::for_name(f))
                .is_some_and(|f| {
                    matches!(
                        f.files_format(),
                        per_field_knn_vectors::KnnVectorsFormat::ScalarQuantized { .. }
                    )
                }),
            vec: seg_dir.open(&format!("{seg}.vec"))?,
            vemf: seg_dir.open(&format!("{seg}.vemf"))?,
            graph: pair("vem", "vex")?,
            quantized: pair("vemq", "veq")?,
            ivf: if listed("ivm") {
                Some(crate::ivf_vectors::IvfInputs {
                    meta: seg_dir.open(&format!("{seg}.ivm"))?,
                    postings: std::sync::Arc::new(seg_dir.open(&format!("{seg}.ivp"))?),
                    vectors: std::sync::Arc::new(seg_dir.open(&format!("{seg}.ivv"))?),
                    segment_id: *segment_id,
                    suffix: suffix.clone(),
                })
            } else {
                None
            },
            suffix,
        });
    }
    Ok(vectors)
}

fn open_raw_points(
    seg_dir: &dyn Directory,
    seg_files: &[String],
    name: &str,
) -> Result<Option<(Input, Input, Input)>> {
    Ok(if seg_files.iter().any(|f| f.ends_with(".kdd")) {
        Some((
            seg_dir.open(&format!("{name}.kdm"))?,
            seg_dir.open(&format!("{name}.kdi"))?,
            seg_dir.open(&format!("{name}.kdd"))?,
        ))
    } else {
        None
    })
}
