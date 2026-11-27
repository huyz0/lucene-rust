use super::*;

/// Everything resolving a buffered delete against one segment needs, owned:
/// this port's slice of Java's `ReadersAndUpdates`. [`OpenedDeleteSegment::view`]
/// turns it into the borrowed form the resolvers take, which is the only way to
/// express "a `DocInput` borrowing the `.doc` bytes next to it" without a
/// self-referential struct.
pub(super) struct OpenedDeleteSegment {
    max_doc: usize,
    live_docs: Option<FixedBitSet>,
    /// The term dictionary and the `.doc` file's bytes -- the latter held as
    /// the `Input` the directory handed over rather than copied into a `Vec`
    /// (on an `MmapDirectory` that `Input` *is* the mapping, so a `to_vec()`
    /// would heap-copy the whole postings file on every buffered-delete
    /// round). Pooled across rounds ([`crate::reader_pool`]).
    postings: std::sync::Arc<crate::reader_pool::PooledPostings>,
    segment_id: [u8; ID_LENGTH],
    suffix: String,
}

impl IndexingConfig {
    /// `FrozenBufferedUpdates.apply(SegmentState[])` for one segment: resolve
    /// every applicable packet's term deletes, query deletes and doc-values
    /// updates against it, then write the results out as one `.liv` generation
    /// and one doc-values-update generation.
    ///
    /// Java resolves and writes per packet through a pooled
    /// `ReadersAndUpdates`; this port has no reader pool, so it opens the
    /// segment once, resolves *all* applicable packets against that one open,
    /// and writes once. Same outcome (deletes are a set union; doc-values
    /// updates are applied in ascending generation order so the newest still
    /// wins), one open and one generation bump instead of N.
    /// Returns whether this apply left the segment 100% hard-deleted, which is
    /// `closeSegmentStates`' `rld.isFullyDeleted()` -- see
    /// [`IndexWriter::drop_fully_deleted_segments`].
    pub(crate) fn apply_packets_to_segment(
        &self,
        dir: &dyn Directory,
        packets: &[FrozenBufferedUpdates],
        sci: &mut SegmentCommitInfo,
        pending_sort_map: Option<&(String, Vec<usize>)>,
    ) -> Result<bool> {
        let seg_gen = sci.buffered_deletes_gen;
        self.apply_packets_at_gen(dir, packets, sci, seg_gen, pending_sort_map)
    }

    /// `DocumentsWriterPerThread.flush`'s `applyDeletes` half: a segment just
    /// built, not yet published, has its own buffer's deletes resolved against
    /// it -- each limited to the documents added before it (in `sort_map`'s
    /// pre-sort order, when the flush sorted) -- by the thread that built it,
    /// with no writer lock held. Java does this during the flush too, from the
    /// in-RAM postings. The segment is then published with no private packet.
    /// Returns whether it left every document deleted.
    pub(crate) fn apply_flush_private_updates(
        &self,
        dir: &dyn Directory,
        private: &crate::buffered_updates::BufferedUpdates,
        sci: &mut SegmentCommitInfo,
        sort_map: Option<&(String, Vec<usize>)>,
    ) -> Result<bool> {
        // The packet and the segment share generation 0 for this one apply:
        // equal generations are what makes the per-entry limits apply.
        let packet = FrozenBufferedUpdates::new(private, Some(sci.segment_name.clone()))
            .with_flush_generation();
        self.apply_packets_at_gen(dir, std::slice::from_ref(&packet), sci, 0, sort_map)
    }

    pub(super) fn apply_packets_at_gen(
        &self,
        dir: &dyn Directory,
        packets: &[FrozenBufferedUpdates],
        sci: &mut SegmentCommitInfo,
        seg_gen: i64,
        pending_sort_map: Option<&(String, Vec<usize>)>,
    ) -> Result<bool> {
        let applicable: Vec<&FrozenBufferedUpdates> = packets
            .iter()
            .filter(|p| p.applies_to(seg_gen))
            .filter(|p| {
                // A segment-private packet belongs to exactly one segment;
                // `applies_to` alone would also let it reach an older segment
                // that happens to share the generation boundary.
                p.private_segment
                    .as_deref()
                    .is_none_or(|name| name == sci.segment_name)
            })
            .collect();
        if applicable.is_empty() {
            // Not a candidate for the fully-deleted drop: Java's
            // `openSegmentStates` never opened a state for it.
            return Ok(false);
        }

        // `ReadersAndUpdates.sortMap` for this segment, if this writer's
        // flush just sorted it. Every `docIDUpto` limit below is a position
        // in the **pre-sort** buffer, so a doc id from the sorted segment has
        // to be mapped back through `newToOld` before it is compared -- Java
        // does the same in `FrozenBufferedUpdates`' two `sortMap.newToOld(doc)
        // < limit` branches.
        let sort_map: Option<&Vec<usize>> = pending_sort_map
            .filter(|(name, _)| name == &sci.segment_name)
            .map(|(_, map)| map);
        // "Is this doc id below the packet's pre-sort limit?" -- the one
        // predicate every filter below uses, so the mapping cannot be applied
        // in one place and forgotten in another.
        let owned = self.open_segment_for_deletes(dir, sci)?;
        let opened = owned.view()?;
        let max_doc = opened.max_doc;
        // A doc id the postings decoder produced out of range is the
        // `DocOutOfRange` corruption `deletes::mark_deleted` reports; checking
        // it here, before `below_limit` indexes the sort map or `deleted` sets
        // a bit, keeps it an error rather than an index panic.
        let in_range = |doc_id: i32| -> std::result::Result<usize, deletes::Error> {
            usize::try_from(doc_id)
                .ok()
                .filter(|idx| *idx < max_doc)
                .ok_or(deletes::Error::DocOutOfRange { doc_id, max_doc })
        };
        let below_limit = |doc: usize, limit: i32| -> bool {
            match &sort_map {
                None => i64::try_from(doc).is_ok_and(|d| d < i64::from(limit)),
                Some(map) => map
                    .get(doc)
                    .is_some_and(|&old| i64::try_from(old).is_ok_and(|o| o < i64::from(limit))),
            }
        };
        // `below_limit` indexes the map by doc id. The invariant is
        // maintained by construction (the map is the permutation of the
        // buffer that became this segment, and the name filter above picks
        // the segment it names), but it is maintained by three separate
        // assignments rather than by a type, so pin it here.
        debug_assert!(
            sort_map.as_ref().is_none_or(|m| m.len() == opened.max_doc),
            "the sort map must have one entry per document of the segment it names"
        );
        // A set, not a list: `applyDocValuesUpdates` has to ask whether a doc
        // is already dead, and the same doc can be named by several packets.
        // A bitset over the segment's docs: one bit test per lookup, and it
        // comes out in doc id order for `apply_deletes`.
        let mut deleted = FixedBitSet::new(max_doc);
        let mut any_deleted = false;
        // field number -> updates in ascending packet-generation order, so the
        // last write for a doc wins (`DocValuesFieldUpdates.mergedIterator`).
        let mut numeric_updates: Vec<PerFieldNumericUpdates> = Vec::new();
        let mut binary_updates: Vec<PerFieldBinaryUpdates> = Vec::new();

        for packet in &applicable {
            for (term, entry_limit) in &packet.delete_terms {
                let limit = packet.limit_for(seg_gen, *entry_limit);
                let docs = term_delete::resolve_term_doc_ids(
                    opened.fields,
                    opened.doc_in.as_ref(),
                    opened.live_docs,
                    &term.field,
                    &term.bytes,
                )?;
                for d in docs {
                    let d = in_range(d)?;
                    if below_limit(d, limit) {
                        // FBS: `deleted` is `FixedBitSet::new(max_doc)` and
                        // `in_range` returned `d < max_doc`.
                        deleted.set(d);
                        any_deleted = true;
                    }
                }
            }
            for (query, entry_limit) in &packet.delete_queries {
                let limit = packet.limit_for(seg_gen, *entry_limit);
                // A sorted segment scatters the documents below the limit
                // across the whole doc id range, so the scan cannot stop
                // early -- Java's own sorted branch likewise drops the
                // `docID < limit` loop bound and filters instead.
                let bound = if sort_map.is_some() {
                    opened.max_doc
                } else {
                    query_bound(limit)
                };
                let docs = IndexWriter::resolve_delete_query(query, &opened, bound)?;
                for d in docs {
                    let d = in_range(d)?;
                    if below_limit(d, limit) {
                        // FBS: `deleted` is `FixedBitSet::new(max_doc)` and
                        // `in_range` returned `d < max_doc`.
                        deleted.set(d);
                        any_deleted = true;
                    }
                }
            }
            for (field, buffer) in &packet.field_updates {
                let Some(info) = self.field_by_name(field) else {
                    // Java's `verifyOrCreateDvOnlyField` already rejected an
                    // unknown field at buffer time; reaching here means the
                    // update was buffered before this writer's field list was
                    // in play, and there is nothing to write it against.
                    continue;
                };
                for update in &buffer.updates {
                    let limit = packet.limit_for(seg_gen, update.doc_id_upto);
                    let docs = term_delete::resolve_term_doc_ids(
                        opened.fields,
                        opened.doc_in.as_ref(),
                        opened.live_docs,
                        &update.term.field,
                        &update.term.bytes,
                    )?;
                    // `applyDocValuesUpdates` reads
                    // `segState.rld.getLiveDocs()`, which by then already
                    // reflects *this packet's* term and query deletes (Java
                    // runs the three in that order inside one
                    // `FrozenBufferedUpdates.apply`). So a document this packet
                    // just killed takes no update. `resolve_term_doc_ids` has
                    // already filtered by the segment's pre-pass live docs;
                    // `deleted` is the rest of the answer.
                    let mut matched: Vec<i32> = Vec::with_capacity(docs.len());
                    for d in docs {
                        let idx = in_range(d)?;
                        // FBS: `deleted` is `FixedBitSet::new(max_doc)` and
                        // `in_range` returned `idx < max_doc`.
                        if below_limit(idx, limit) && !deleted.get(idx) {
                            matched.push(d);
                        }
                    }
                    if matched.is_empty() {
                        continue;
                    }
                    if buffer.is_numeric {
                        let value = match &update.value {
                            UpdateValue::Numeric(v) => Some(*v),
                            // `hasValue == false` -> `reset(doc)`.
                            _ => None,
                        };
                        let slot = entry_for(&mut numeric_updates, info.number);
                        slot.extend(matched.into_iter().map(|d| (d, value)));
                    } else {
                        let value = match &update.value {
                            UpdateValue::Binary(v) => Some(v.clone()),
                            _ => None,
                        };
                        let slot = entry_for(&mut binary_updates, info.number);
                        slot.extend(matched.into_iter().map(|d| (d, value.clone())));
                    }
                }
            }
        }

        if any_deleted {
            // Ascending doc id order, straight off the bitset; every set bit
            // is below `max_doc`, which fits an `i32` (it is a `.si` count).
            let deleted_ids = std::iter::successors(deleted.next_set_bit(0), |&d| {
                d.checked_add(1).and_then(|n| deleted.next_set_bit(n))
            })
            .filter_map(|d| i32::try_from(d).ok());
            let updated = deletes::apply_deletes_with_codec(
                dir,
                sci,
                opened.live_docs,
                opened.max_doc,
                deleted_ids,
                self.codec.as_deref(),
            );
            if updated.is_err() {
                // `PendingDeletes.writeLiveDocs`' failure path:
                // `info.advanceNextWriteDelGen()`, so a retry never asks for
                // the name a failed attempt may have left behind.
                let next = sci.next_write_del_gen();
                sci.set_next_write_del_gen(next.saturating_add(1));
            }
            *sci = updated?;
        }

        if !numeric_updates.is_empty() || !binary_updates.is_empty() {
            Self::write_doc_values_update_generation(
                dir,
                sci,
                &numeric_updates,
                &binary_updates,
                &self.fields,
                self.codec.as_deref(),
            )?;
        }
        // `PendingDeletes.isFullyDeleted`: `getDelCount() == info.info.maxDoc()`.
        Ok(i64::from(sci.del_count) == opened.max_doc as i64)
    }

    /// Writes one doc-values-update generation for `sci` --
    /// `ReadersAndUpdates.writeFieldUpdates`, delegated to
    /// [`crate::field_updates`], which owns the file naming, the
    /// `FieldInfos` generation and the `SegmentCommitInfo` bookkeeping.
    ///
    /// The bytes are **real Lucene's**: each updated field's whole column is
    /// rewritten through the `Lucene90DocValuesFormat` writer into a
    /// generation-suffixed `.dvm`/`.dvd`/`.dvs` triple, and a new `.fnm`
    /// generation records the field's `FieldInfo.docValuesGen`.
    pub(super) fn write_doc_values_update_generation(
        dir: &dyn Directory,
        sci: &mut SegmentCommitInfo,
        numeric: &[PerFieldNumericUpdates],
        binary: &[PerFieldBinaryUpdates],
        schema: &[FieldInfo],
        codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
    ) -> Result<()> {
        crate::field_updates::write_field_updates_with_codec(
            dir,
            sci,
            numeric,
            binary,
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            schema,
            codec,
        )?;
        Ok(())
    }

    /// Opens exactly what resolving a buffered delete against one segment
    /// needs: its `max_doc` (from the `.si`, not from a stored-fields open --
    /// the doc count is already recorded there), its current live docs, and
    /// its term dictionary + `.doc` postings when it has any.
    ///
    /// This is the slice of Java's `ReaderPool`/`ReadersAndUpdates` that
    /// buffered deletes actually use. A segment with no postings files opens
    /// with an empty [`lucene_codecs::blocktree::BlockTreeFields`], which
    /// resolves every term to zero documents -- the same outcome as Java's
    /// `TermDocsIterator.nextTerm` returning null, and the reason a delete
    /// against a stored-fields-only segment is a legitimate no-op rather than
    /// an error.
    pub(super) fn open_segment_for_deletes(
        &self,
        dir: &dyn Directory,
        sci: &SegmentCommitInfo,
    ) -> Result<OpenedDeleteSegment> {
        let si = self.reader_pool.segment_info(dir, sci)?;
        let max_doc = si.doc_count as usize;

        let live_docs = if sci.del_gen >= 0 {
            let liv = dir.open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))?;
            Some(lucene_codecs::live_docs::parse(
                &liv,
                &sci.segment_id,
                sci.del_gen,
                max_doc,
                sci.del_count as usize,
            )?)
        } else {
            None
        };

        let postings = self.reader_pool.get_or_open(sci, || {
            Self::open_postings_for_deletes(dir, sci, &si, max_doc)
        })?;
        let suffix = postings.suffix.clone();
        Ok(OpenedDeleteSegment {
            max_doc,
            live_docs,
            postings,
            segment_id: sci.segment_id,
            suffix,
        })
    }

    /// The part of [`Self::open_segment_for_deletes`] that does not change
    /// between delete rounds, and so is pooled: the segment's term dictionary
    /// and `.doc` input, from `.cfs` for a compound segment.
    pub(super) fn open_postings_for_deletes(
        dir: &dyn Directory,
        sci: &SegmentCommitInfo,
        si: &segment_info::SegmentInfo,
        max_doc: usize,
    ) -> Result<crate::reader_pool::PooledPostings> {
        // A compound segment -- one real Lucene flushed -- keeps its term
        // dictionary inside `.cfs`, and its `.si` lists no `.tim` at all:
        // testing `si.files` alone would report "no postings" and every
        // delete against it would silently match nothing.
        let compound = if si.is_compound_file {
            Some(CompoundReader::open(
                dir,
                &sci.segment_name,
                &sci.segment_id,
            )?)
        } else {
            None
        };
        let seg_dir: &dyn Directory = match &compound {
            Some(c) => c,
            None => dir,
        };
        let seg_files: Vec<String> = match &compound {
            Some(c) => c.member_files(),
            None => si.files.clone(),
        };

        // The segment's own postings format: an older Lucene's segment names
        // its files (and frames their headers) after it, not after this
        // writer's.
        let (seg, suffix) = postings_file_base(&seg_files, &sci.segment_name);
        if !per_field_postings::has_postings_files(&seg_files) {
            return Ok(crate::reader_pool::PooledPostings {
                fields: lucene_codecs::blocktree::BlockTreeFields::empty(),
                doc_input: None,
                suffix,
                combined: false,
            });
        }

        // The segment's **own** field infos, every indexed field of them:
        // `blocktree::open` resolves the `.tmd`'s field numbers through this
        // list, and those numbers are the segment's -- a segment another
        // writer (real Lucene) produced numbers its fields its own way, which
        // this writer's schema need not match. Java's `SegmentReader` reads
        // the same `.fnm` for the same reason.
        let own = crate::field_updates::read_current_field_infos(seg_dir, sci, &seg_files)?;
        let field_infos = lucene_codecs::field_infos::FieldInfos {
            fields: own
                .fields
                .into_iter()
                .filter(|f| f.index_options != IndexOptions::None)
                .collect(),
        };
        // `PerFieldPostingsFormat` with more than one format: every group,
        // combined into one dictionary over one concatenated `.doc`.
        let suffixes = per_field_postings::group_suffixes(&seg_files, &sci.segment_name);
        if suffixes.len() > 1 || suffixes.iter().any(|s| is_foreign_postings_group(s)) {
            let mut opened = Vec::with_capacity(suffixes.len());
            for suffix in &suffixes {
                let seg = format!("{}_{suffix}", sci.segment_name);
                let open = |ext: &str| seg_dir.open(&format!("{seg}.{ext}"));
                let format = per_field_postings::format_of_suffix(suffix);
                let dictionary = per_field_postings::dictionary_extensions(format)
                    .iter()
                    .map(|ext| {
                        if !seg_files.contains(&format!("{seg}.{ext}"))
                            && !per_field_postings::dictionary_file_required(format, ext)
                        {
                            return Ok(lucene_store::directory::Input::Owned(Vec::new()));
                        }
                        open(ext)
                    })
                    .collect::<lucene_store::Result<Vec<_>>>()?;
                let doc = if seg_files.contains(&format!("{seg}.doc")) {
                    Some(open("doc")?)
                } else {
                    None
                };
                opened.push((suffix, dictionary, doc));
            }
            let groups: Vec<per_field_postings::GroupFiles<'_>> = opened
                .iter()
                .map(|(suffix, dictionary, doc)| per_field_postings::GroupFiles {
                    suffix,
                    dictionary: dictionary.iter().map(|d| &**d).collect(),
                    doc: doc.as_deref(),
                    pos: None,
                    pay: None,
                })
                .collect();
            let combined = per_field_postings::open_groups(
                &groups,
                &field_infos,
                &sci.segment_id,
                max_doc as i32,
            )?;
            return Ok(crate::reader_pool::PooledPostings {
                fields: combined.fields,
                doc_input: Some(lucene_store::directory::Input::Owned(combined.doc)),
                suffix,
                combined: true,
            });
        }
        // Borrowed from the directory's `Input` (a mapping, under
        // `MmapDirectory`), not copied: `blocktree::open` builds its own
        // structures from these and does not retain the slices, and the `.doc`
        // `Input` is kept alive in the returned struct for `DocInput` to
        // borrow.
        let tim = seg_dir.open(&format!("{seg}.tim"))?;
        let tip = seg_dir.open(&format!("{seg}.tip"))?;
        let tmd = seg_dir.open(&format!("{seg}.tmd"))?;
        let doc_input = seg_dir.open(&format!("{seg}.doc"))?;
        let fields = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &suffix,
            max_doc as i32,
        )?;
        Ok(crate::reader_pool::PooledPostings {
            fields,
            doc_input: Some(doc_input),
            suffix,
            combined: false,
        })
    }
}

impl OpenedDeleteSegment {
    pub(super) fn view(&self) -> Result<DeleteSegmentView<'_>> {
        let doc_in = match &self.postings.doc_input {
            Some(bytes) if self.postings.combined => {
                Some(lucene_codecs::postings::DocInput::validated(bytes))
            }
            Some(bytes) => Some(lucene_codecs::postings::DocInput::open(
                bytes,
                &self.segment_id,
                &self.suffix,
            )?),
            None => None,
        };
        Ok(DeleteSegmentView {
            max_doc: self.max_doc,
            live_docs: self.live_docs.as_ref(),
            fields: &self.postings.fields,
            doc_in,
        })
    }
}

/// The borrowed view of an [`OpenedDeleteSegment`] the delete resolvers take.
pub(super) struct DeleteSegmentView<'a> {
    pub(super) max_doc: usize,
    pub(super) live_docs: Option<&'a FixedBitSet>,
    pub(super) fields: &'a lucene_codecs::blocktree::BlockTreeFields,
    pub(super) doc_in: Option<lucene_codecs::postings::DocInput<'a>>,
}

/// One field's resolved NUMERIC doc-values updates for one generation:
/// `(field number, (doc id, value-or-reset) pairs)`.
pub(super) type PerFieldNumericUpdates = (i32, Vec<(i32, Option<i64>)>);

/// One field's resolved BINARY doc-values updates for one generation.
pub(super) type PerFieldBinaryUpdates = (i32, Vec<(i32, Option<Vec<u8>>)>);

/// What [`IndexWriter::resolve_term_span`]'s classifier decides for one term.
pub(super) enum TermSpan {
    /// The term is in the query's span: take its documents.
    Take,
    /// Below the span; keep walking.
    Skip,
    /// Past the span; the dictionary is sorted, so nothing after it can match.
    Stop,
}
