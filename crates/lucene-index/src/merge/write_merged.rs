//! Writing a merged segment: [`merge_segments_mapped_inner`] checks the
//! sources, decides the merged document order, then writes each format's
//! files (stored fields, norms, postings, doc values, term vectors, points,
//! vectors), the field infos and the segment info, in the order Java's
//! `SegmentMerger` does. The order is part of the behaviour: it decides which
//! files exist when a stage fails.

use super::*;
use rayon::prelude::*;

/// What every writing stage reads: the sources, where the merged segment
/// goes, and the merged fields and document order the stages share.
struct SegmentWriter<'a> {
    dir: &'a dyn Directory,
    sources: &'a [MergeSource<'a>],
    options: &'a MergeOptions,
    segment_name: &'a str,
    segment_id: [u8; ID_LENGTH],
    merged_fields: Vec<FieldInfo>,
    per_source_maps: &'a [HashMap<i32, i32>],
    reverse_maps: &'a [HashMap<i32, i32>],
    per_source_live_ids: &'a [Vec<i32>],
    doc_order: &'a [(usize, i32)],
    doc_id_maps: &'a [Vec<i32>],
    doc_count: i32,
}

/// How the merged segment's stored fields and term vectors are written.
struct StoredFormat {
    /// A `CompressingCodec`'s stored fields and term vectors take its
    /// format's name, settings and segment suffix.
    compressing: Option<&'static lucene_codecs::compression::CompressingFormat>,
    mode: stored_fields::Mode,
    /// The high-compression match finder only changes LZ4 modes' bytes.
    lz4_hc: bool,
    suffix: &'static str,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn merge_segments_mapped_inner(
    dir: &dyn Directory,
    sources: &[MergeSource],
    sort_fields: Option<&[MergeSortKeySpec<'_>]>,
    options: &MergeOptions,
    merged_segment_name: &str,
    merged_segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
) -> Result<MergedSegment> {
    check_sort_key_tables(sources, sort_fields)?;

    let sources_fields: Vec<&[FieldInfo]> = sources.iter().map(|s| s.field_infos).collect();
    let (merged_fields, per_source_maps) = reconcile_field_numbers(&sources_fields)?;

    let per_source_live_ids = live_doc_ids(sources)?;
    check_norms_integrity(sources, &per_source_live_ids)?;

    let per_source_max_doc: Vec<i32> = sources.iter().map(|s| s.reader.max_doc()).collect();
    let doc_order = plan_doc_order(
        sources,
        sort_fields,
        options,
        &per_source_live_ids,
        &per_source_max_doc,
    )?;
    let doc_count = doc_order.len() as i32;
    let doc_id_maps = build_doc_id_maps(&per_source_max_doc, &doc_order);
    let partition_starts = match sort_fields.and_then(|f| f.first()) {
        Some(tier) if options.ivf_partition_clusters.is_some() => {
            Some(partition_starts(&doc_order, tier))
        }
        _ => None,
    };
    let partitions = partition_starts
        .as_deref()
        .zip(options.ivf_partition_clusters)
        .map(|(starts, min_vectors)| crate::ivf_vectors::Partitions {
            starts,
            min_vectors,
        });

    // Doc values, norms and term vectors, all resolved through the one
    // `doc_order` -- the merged segment's physical order, whichever way it
    // was produced.
    //
    // The five doc-values merges, the norms merge and the points merge read
    // only the sources and write nothing, so they run in parallel up front.
    // Each result is unwrapped at the stage that consumes it, so the files
    // written before a failing stage, and which error wins, do not depend on
    // the parallelism.
    let reverse_maps = invert_field_number_maps(&per_source_maps);
    let norms_result = merge_norms(
        sources,
        &merged_fields,
        &reverse_maps,
        &per_source_live_ids,
        &doc_order,
    );
    let doc_values_field_numbers =
        merged_doc_values_field_numbers(sources, &per_source_maps, &per_source_live_ids);

    let writer = SegmentWriter {
        dir,
        sources,
        options,
        segment_name: merged_segment_name,
        segment_id: merged_segment_id,
        merged_fields,
        per_source_maps: &per_source_maps,
        reverse_maps: &reverse_maps,
        per_source_live_ids: &per_source_live_ids,
        doc_order: &doc_order,
        doc_id_maps: &doc_id_maps,
        doc_count,
    };
    let mut files: Vec<String> = Vec::new();
    let stored = writer.stored_format();
    let norms_factory = options
        .codec
        .as_deref()
        .and_then(|codec| codec.norms_writer_factory());
    let points_factory = options
        .codec
        .as_deref()
        .and_then(|codec| codec.points_writer_factory());
    let postings_factory = options
        .codec
        .as_deref()
        .and_then(|codec| codec.postings_writer_factory());
    let term_vectors_factory = options
        .codec
        .as_deref()
        .and_then(|codec| codec.term_vectors_writer_factory());
    let installed_segment_writer = norms_factory.is_some()
        || points_factory.is_some()
        || postings_factory.is_some()
        || term_vectors_factory.is_some();
    // Java: SegmentMerger.merge writes stored fields before norms, and closes
    // norms before selecting postings/doc-values/vector formats. Source columns
    // here remain Rust-owned decoded inputs; no source NormsProducer is routed.
    if installed_segment_writer {
        writer.write_stored(&stored, &mut files)?;
    }

    let merged_norms = norms_result?;
    // Missing values use one only for postings impacts. Sparse wire columns
    // omit them, and a present zero remains zero (see `MergedNorms`).
    let norms_fields: Vec<norms::NormsField<'_>> = merged_norms
        .iter()
        .map(MergedNorms::as_norms_field)
        .collect();
    if let Some(factory) = norms_factory {
        if !norms_fields.is_empty() {
            writer.write_installed_norms(factory, &norms_fields, &mut files)?;
        }
    }
    if installed_segment_writer && norms_factory.is_none() && !merged_norms.is_empty() {
        writer.write_native_norms(&norms_fields, &mut files)?;
    }

    let tv_files = if term_vectors_factory.is_none() {
        write_merged_term_vectors(
            stored.compressing,
            sources,
            &per_source_maps,
            &doc_order,
            &merged_segment_id,
        )?
    } else {
        None
    };
    let wrote_term_vectors = sources.iter().any(|source| source.term_vectors.is_some());
    let postings_plans = plan_postings(
        sources,
        &per_source_maps,
        &per_source_live_ids,
        &doc_order,
        &writer.merged_fields,
    )?;
    // What the merged `.fnm` is allowed to claim, captured before the values
    // below are consumed by their writers (see `describe_written_files`).
    let merged_field_by_number: HashMap<i32, &FieldInfo> =
        writer.merged_fields.iter().map(|f| (f.number, f)).collect();
    let field_name = |number: i32| {
        merged_field_by_number
            .get(&number)
            .map_or_else(|| number.to_string(), |m| m.name.clone())
    };
    let postings_field_names: Vec<(i32, String)> = postings_plans
        .iter()
        .map(|f| (f.field_number, field_name(f.field_number)))
        .collect();
    let format_for = |name: &str| options.postings_format_for(name);
    let postings_suffixes = per_field_postings::field_suffixes(&postings_field_names, &format_for);
    // `PerFieldDocValuesFormat.FieldsWriter.merge`: the merged `FieldInfos`
    // in field-number order, each field to the instance of the format it is
    // routed to now (`getInstance(fi, true)` ignores the sources' routing).
    let doc_values_groups = {
        let mut fields: Vec<(i32, String)> = doc_values_field_numbers
            .iter()
            .map(|&n| (n, field_name(n)))
            .collect();
        fields.sort_by_key(|(number, _)| *number);
        let fields: Vec<(i32, &str)> = fields.iter().map(|(n, s)| (*n, s.as_str())).collect();
        let format_for = |name: &str| options.doc_values_format_for(name);
        per_field_doc_values::group_fields(&fields, &format_for)
    };
    let doc_values_suffixes: Vec<(i32, &'static str, u32)> = doc_values_groups
        .iter()
        .flat_map(|g| {
            g.field_numbers
                .iter()
                .map(move |&n| (n, g.format.name(), g.suffix))
        })
        .collect();
    if !installed_segment_writer {
        writer.write_stored(&stored, &mut files)?;
    }

    if postings_factory.is_some() {
        writer.write_postings(
            &postings_plans,
            &postings_field_names,
            &merged_norms,
            postings_factory,
            &mut files,
        )?;
    }

    writer.write_doc_values(&doc_values_groups, &mut files)?;

    if !installed_segment_writer && norms_factory.is_none() && !merged_norms.is_empty() {
        writer.write_native_norms(&norms_fields, &mut files)?;
    }

    if let Some((tvd, tvx, tvm)) = tv_files {
        for (ext, bytes) in [("tvd", &tvd), ("tvx", &tvx), ("tvm", &tvm)] {
            let name = writer.format_file(&stored, ext);
            write_file(dir, &name, bytes)?;
            files.push(name);
        }
    }

    if postings_factory.is_none() {
        writer.write_postings(
            &postings_plans,
            &postings_field_names,
            &merged_norms,
            postings_factory,
            &mut files,
        )?;
    }

    writer.write_points(points_factory, &mut files)?;

    let merged_vectors = merge_vectors(
        dir,
        merged_segment_name,
        sources,
        &per_source_maps,
        &doc_id_maps,
        &writer.merged_fields,
        doc_count,
        options,
        &merged_segment_id,
        partitions,
    )?;

    // Java: SegmentMerger.mergeTermVectors follows postings, doc values,
    // points and KNN merging. Source checks precede the vectors factory.
    // The default native path above retains its compressed-chunk fast path.
    if let Some(factory) = term_vectors_factory {
        if wrote_term_vectors {
            writer.write_installed_term_vectors(factory, &stored, &mut files)?;
        }
    }

    let mut writer = writer;
    writer.write_field_infos(
        &postings_suffixes,
        &doc_values_suffixes,
        wrote_term_vectors,
        merged_vectors.as_ref(),
        &mut files,
    )?;
    if let Some(vector_files) = merged_vectors {
        writer.write_vector_files(&vector_files, &mut files)?;
    }

    let extra = crate::index_writer::write_codec_segment_files(
        options.codec.as_deref(),
        lucene_codecs::codec::SegmentWrite {
            directory: dir,
            segment: merged_segment_name,
            id: &merged_segment_id,
            max_doc: doc_count,
            field_infos: &writer.merged_fields,
            files: &files,
        },
    )?;
    files.extend(extra);
    let info =
        writer.write_segment_info(&mut files, &stored, sort_fields, codec_name, lucene_version)?;
    Ok(MergedSegment {
        info,
        doc_id_maps,
        files,
    })
}

impl SegmentWriter<'_> {
    /// How the merged segment's stored fields are written, from the codec.
    fn stored_format(&self) -> StoredFormat {
        let options = self.options;
        let compressing = options
            .codec
            .as_deref()
            .and_then(|codec| codec.compressing_format());
        let stored_mode = options
            .codec
            .as_deref()
            .map_or(stored_fields::Mode::BestSpeed, |codec| {
                codec.stored_fields_mode()
            });
        // The high-compression match finder only changes LZ4 modes' bytes.
        let lz4_hc = options.stored_fields_lz4_high_compression
            && matches!(stored_mode, stored_fields::Mode::BestSpeed)
            && options
                .codec
                .as_deref()
                .and_then(|codec| codec.stored_fields_writer_factory())
                .is_none();
        let suffix = stored_mode.segment_suffix();
        StoredFormat {
            compressing,
            mode: stored_mode,
            lz4_hc,
            suffix,
        }
    }

    /// The name of a file in the merged segment's stored-fields format.
    fn format_file(&self, stored: &StoredFormat, ext: &str) -> String {
        crate::index_writer::segment_file_name(self.segment_name, stored.suffix, ext)
    }

    fn write_stored(&self, stored: &StoredFormat, files: &mut Vec<String>) -> Result<()> {
        let SegmentWriter {
            dir,
            sources,
            options,
            segment_name: merged_segment_name,
            segment_id: merged_segment_id,
            per_source_maps,
            doc_order,
            ..
        } = *self;
        let (stored_mode, lz4_hc, format_suffix) = (stored.mode, stored.lz4_hc, stored.suffix);
        let format_file = |ext| self.format_file(stored, ext);
        let fdt_name = format_file("fdt");
        let fdx_name = format_file("fdx");
        let fdm_name = format_file("fdm");
        if let Some(factory) = options
            .codec
            .as_deref()
            .and_then(|codec| codec.stored_fields_writer_factory())
        {
            write_merged_stored_fields_with_factory(
                factory,
                lucene_codecs::stored_fields_writer::StoredFieldsWriteState {
                    directory: dir,
                    segment: merged_segment_name,
                    id: &merged_segment_id,
                    suffix: format_suffix,
                    mode: stored_mode,
                },
                sources,
                per_source_maps,
                doc_order,
            )?;
            files.extend([fdt_name, fdx_name, fdm_name]);
        } else {
            // The `.fdt` streams to its file as chunks are copied: a merge
            // holds a few chunks of stored fields, not the merged segment's.
            let (fdt_out, fdx, fdm) = write_merged_stored_fields(
                stored_mode,
                lz4_hc,
                &options.lz4_high_compression_sources,
                options.zstd_dictionary_min_bytes,
                sources,
                per_source_maps,
                doc_order,
                &merged_segment_id,
                dir.create_output(&fdt_name)?,
            )?;
            fdt_out.close()?;
            files.push(fdt_name.clone());
            for (name, bytes) in [(&fdx_name, &fdx), (&fdm_name, &fdm)] {
                write_file(dir, name, bytes)?;
                files.push(name.clone());
            }
        }

        Ok(())
    }

    fn write_installed_norms(
        &self,
        factory: &dyn lucene_codecs::norms_writer::NormsWriterFactory,
        norms_fields: &[norms::NormsField<'_>],
        files: &mut Vec<String>,
    ) -> Result<()> {
        files.extend(crate::norms_writer::write_installed(
            factory,
            lucene_codecs::norms_writer::NormsWriteState {
                directory: self.dir,
                segment: self.segment_name,
                id: &self.segment_id,
                suffix: "",
                max_doc: self.doc_count,
            },
            &self.merged_fields,
            norms_fields,
        )?);
        Ok(())
    }

    /// The norms in this crate's own `.nvm`/`.nvd` format.
    fn write_native_norms(
        &self,
        norms_fields: &[norms::NormsField<'_>],
        files: &mut Vec<String>,
    ) -> Result<()> {
        let (nvm, nvd) = norms::write_fields(norms_fields, self.doc_count, &self.segment_id, "")?;
        for (ext, bytes) in [("nvm", &nvm), ("nvd", &nvd)] {
            let name = format!("{}.{ext}", self.segment_name);
            write_file(self.dir, &name, bytes)?;
            files.push(name);
        }
        Ok(())
    }

    fn write_installed_term_vectors(
        &self,
        factory: &dyn lucene_codecs::term_vectors_writer::TermVectorsWriterFactory,
        stored: &StoredFormat,
        files: &mut Vec<String>,
    ) -> Result<()> {
        files.extend(write_installed_term_vectors(
            factory,
            lucene_codecs::term_vectors_writer::TermVectorsWriteState {
                directory: self.dir,
                segment: self.segment_name,
                id: &self.segment_id,
                suffix: stored.suffix,
                max_doc: self.doc_count,
                format: stored.compressing,
            },
            self.sources,
            self.per_source_maps,
            self.doc_order,
            &self.merged_fields,
        )?);
        Ok(())
    }

    /// The postings of every merged field, through the codec's factory when
    /// it installed one, streamed to the files otherwise.
    fn write_postings(
        &self,
        postings_plans: &[PostingsFieldPlan<'_, '_>],
        postings_field_names: &[(i32, String)],
        merged_norms: &[MergedNorms],
        postings_factory: Option<&dyn lucene_codecs::postings_consumer::PostingsWriterFactory>,
        files: &mut Vec<String>,
    ) -> Result<()> {
        let SegmentWriter {
            dir,
            options,
            segment_name: merged_segment_name,
            segment_id: merged_segment_id,
            doc_id_maps,
            doc_count,
            ..
        } = *self;
        let merged_fields = &self.merged_fields;
        let format_for = |name: &str| options.postings_format_for(name);
        if postings_plans.is_empty() {
            return Ok(());
        }
        let names: HashMap<i32, &String> = postings_field_names
            .iter()
            .map(|(number, name)| (*number, name))
            .collect();
        let name_of = |n: i32| names.get(&n).map(|name| (*name).clone());
        let norms_for_impacts: Vec<postings_writer::FieldNorms<'_>> = merged_norms
            .iter()
            .map(|norms| postings_writer::FieldNorms {
                field_number: norms.field_number,
                values: &norms.values,
            })
            .collect();
        // The very columns the `.nvd` above was written from, handed to the
        // postings writer so its level-0/level-1 impacts are real
        // `(freq, norm)` pairs rather than `(maxFreq, 1)` -- item 18.
        // `Lucene104PostingsWriter` gets them the same way, from the
        // `NormsProducer` its `FieldsConsumer.write` call is given. Borrowed,
        // not rebuilt: `MergedNorms::values`, missing norms as one.
        //
        // The block-tree format streams: each window of merged term ranges
        // is encoded and dropped before the next is decoded, so the merge
        // never holds the merged segment's decoded postings.
        let streamed = if postings_factory.is_none() {
            let specs: Vec<per_field_postings::StreamedFieldSpec> = postings_plans
                .iter()
                .map(|plan| per_field_postings::StreamedFieldSpec {
                    field_number: plan.field_number,
                    index_options: plan.index_options,
                    has_payloads: plan.shape.has_payloads,
                })
                .collect();
            // `.doc`/`.pos`/`.pay` stream to their files as windows are
            // encoded, so the merge holds neither the decoded postings nor the
            // encoded files whole.
            per_field_postings::write_streaming::<Error>(
                Some((dir, merged_segment_name)),
                &specs,
                merged_fields,
                &name_of,
                &format_for,
                &merged_segment_id,
                &mut |i, writer| {
                    merge_field_runs(&postings_plans[i], doc_id_maps, &mut |run| {
                        let views: Vec<MergedTermView<'_>> = run.views().collect();
                        writer.add_terms(&views, &norms_for_impacts)?;
                        Ok(())
                    })
                },
            )?
        } else {
            None
        };
        let merged_postings_fields = if streamed.is_some() {
            Vec::new()
        } else {
            materialize_postings(postings_plans, doc_id_maps)?
        };
        if let Some(groups) = streamed {
            for group in &groups {
                files.extend(group.streamed_files.iter().cloned());
                for (ext, bytes) in group.files() {
                    let name = group.file_name(merged_segment_name, ext);
                    write_file(dir, &name, bytes)?;
                    files.push(name);
                }
            }
        } else {
            // One small view per term over its range's shared buffers.
            let views: Vec<Vec<MergedTermView<'_>>> = merged_postings_fields
                .iter()
                .map(|f| f.runs.iter().flat_map(MergedRun::views).collect())
                .collect();
            let inputs: Vec<FieldPostingsInput<'_, MergedTermView<'_>>> = merged_postings_fields
                .iter()
                .zip(&views)
                .map(|(f, terms)| FieldPostingsInput {
                    field_number: f.field_number,
                    index_options: f.index_options,
                    doc_count: f.doc_count,
                    has_payloads: f.has_payloads,
                    terms,
                })
                .collect();
            let groups = if let Some(factory) = postings_factory {
                crate::postings_writer::write_installed(
                    factory,
                    lucene_codecs::postings_consumer::PostingsWriteState {
                        directory: dir,
                        segment: merged_segment_name,
                        id: &merged_segment_id,
                        suffix: "",
                        max_doc: doc_count,
                        field_infos: merged_fields,
                        format: per_field_postings::Lucene104PostingsFormat::default(),
                    },
                    &inputs,
                    &norms_for_impacts,
                    &name_of,
                    &format_for,
                    true,
                )?
            } else {
                per_field_postings::write(
                    &inputs,
                    merged_fields,
                    &norms_for_impacts,
                    &name_of,
                    &format_for,
                    &merged_segment_id,
                )?
            };
            for group in &groups {
                if !group.written_files.is_empty() {
                    files.extend(group.written_files.iter().cloned());
                    continue;
                }
                for (ext, bytes) in group.files() {
                    let name = group.file_name(merged_segment_name, ext);
                    write_file(dir, &name, bytes)?;
                    files.push(name);
                }
            }
        }
        Ok(())
    }

    /// Every doc-values field of the merged segment shares one
    /// `.dvm`/`.dvd`/`.dvs` triple per format group, which is what a real
    /// multi-field `Lucene90DocValuesFormat` segment looks like.
    fn write_doc_values(
        &self,
        doc_values_groups: &[per_field_doc_values::Group],
        files: &mut Vec<String>,
    ) -> Result<()> {
        let SegmentWriter {
            dir,
            options,
            segment_name: merged_segment_name,
            segment_id: merged_segment_id,
            doc_count,
            ..
        } = *self;
        let merged_fields = &self.merged_fields;
        let merge_doc_values_field = |field_number: i32| self.merge_doc_values_field(field_number);
        for group in doc_values_groups {
            let in_group: HashSet<i32> = group.field_numbers.iter().copied().collect();
            // `addNumericField` & co. write a skip index for every field whose
            // merged `FieldInfo` asks for one.
            let skip_indexes: Vec<i32> = merged_fields
                .iter()
                .filter(|f| {
                    in_group.contains(&f.number)
                        && f.doc_values_skip_index_type
                            != lucene_codecs::field_infos::DocValuesSkipIndexType::None
                })
                .map(|f| f.number)
                .collect();
            let codec_suffix = group.codec_suffix();
            crate::index_writer::with_doc_values_faults(
                options.codec.as_deref(),
                group.field_numbers.len(),
                || -> Result<()> {
                    let mut writer = doc_values::FieldsWriter::new(
                        &skip_indexes,
                        doc_count,
                        &merged_segment_id,
                        &codec_suffix,
                        group.format,
                    );
                    for window in group.field_numbers.chunks(DOC_VALUES_MERGE_WINDOW) {
                        let columns: Vec<Vec<MergedDocValuesField>> = window
                            .par_iter()
                            .map(|&n| merge_doc_values_field(n))
                            .collect::<Result<_>>()?;
                        for column in columns.iter().flatten() {
                            writer.write_field(&column.as_dense_field())?;
                        }
                    }
                    let doc_values::DocValuesFiles { dvm, dvd, dvs, dvp } =
                        writer.finish_files()?;
                    let field_infos = field_infos::FieldInfos {
                        fields: merged_fields
                            .iter()
                            .filter(|f| in_group.contains(&f.number))
                            .cloned()
                            .collect(),
                    };
                    crate::index_writer::check_doc_values(
                        options.codec.as_deref(),
                        &lucene_codecs::codec::DocValuesWrite {
                            max_doc: doc_count,
                            segment_id: &merged_segment_id,
                            segment_suffix: &codec_suffix,
                            format_name: group.format.name(),
                            field_infos: &field_infos,
                            dvm: &dvm,
                            dvd: &dvd,
                            dvs: &dvs,
                        },
                    )?;
                    let dvp_file = dvp.as_ref().map(|dvp| ("dvp", dvp));
                    for (ext, bytes) in [("dvm", &dvm), ("dvd", &dvd), ("dvs", &dvs)]
                        .into_iter()
                        .chain(dvp_file)
                    {
                        let name = format!("{merged_segment_name}_{codec_suffix}.{ext}");
                        write_file(dir, &name, bytes)?;
                        files.push(name);
                    }
                    Ok(())
                },
            )?;
        }
        Ok(())
    }

    /// One field's doc-values columns, merged. The columns are built a few
    /// fields at a time as the files are written, so a merge holds a handful
    /// of columns, not the segment's whole doc-values data.
    fn merge_doc_values_field(&self, field_number: i32) -> Result<Vec<MergedDocValuesField>> {
        let SegmentWriter {
            sources,
            options,
            per_source_maps,
            reverse_maps,
            per_source_live_ids,
            doc_order,
            ..
        } = *self;
        let merged_fields = &self.merged_fields;
        let sorted_numeric_remap = options.sorted_numeric_remap.as_deref();
        let only = Some(field_number);
        let mut merged = merge_numeric_doc_values(
            sources,
            per_source_maps,
            reverse_maps,
            per_source_live_ids,
            doc_order,
            only,
        )?;
        merged.extend(merge_binary_doc_values(
            sources,
            per_source_maps,
            reverse_maps,
            per_source_live_ids,
            doc_order,
            only,
        )?);
        merged.extend(merge_sorted_doc_values(
            sources,
            per_source_maps,
            reverse_maps,
            per_source_live_ids,
            doc_order,
            only,
        )?);
        merged.extend(merge_sorted_numeric_doc_values(
            sources,
            per_source_maps,
            reverse_maps,
            per_source_live_ids,
            doc_order,
            merged_fields,
            sorted_numeric_remap,
            only,
        )?);
        merged.extend(merge_sorted_set_doc_values(
            sources,
            per_source_maps,
            reverse_maps,
            per_source_live_ids,
            doc_order,
            only,
        )?);
        Ok(merged)
    }

    /// The points of every merged field. Java opens a `PointsWriter` from
    /// the merged `FieldInfos` even when every point was deleted; `finish()`
    /// then emits a header/footer-only triple, while the field's dimensions
    /// remain in the merged `.fnm`.
    fn write_points(
        &self,
        points_factory: Option<&dyn lucene_codecs::points_writer::PointsWriterFactory>,
        files: &mut Vec<String>,
    ) -> Result<()> {
        let SegmentWriter {
            dir,
            sources,
            segment_name: merged_segment_name,
            segment_id: merged_segment_id,
            per_source_maps,
            per_source_live_ids,
            doc_id_maps,
            doc_count,
            ..
        } = *self;
        let merged_fields = &self.merged_fields;
        if merged_fields.iter().any(|f| f.point_dimension_count > 0) {
            // Lucene90PointsWriter's instanceof dispatch: a point-free source
            // has no PointsReader and selects generic merging, including for 1D.
            //
            // Java also falls back to generic merging when `needsIndexSort`: the
            // native 1D merge emits equal values in each source's mapped doc
            // order, which is only ascending when every source's doc map is.
            let native_merge = sources.iter().all(|source| {
                source
                    .field_infos
                    .iter()
                    .any(|f| f.point_dimension_count > 0)
            }) && doc_id_maps.iter().all(|map| doc_map_is_monotonic(map));
            if let Some(factory) = points_factory {
                files.extend(crate::points_writer::merge(
                    factory,
                    lucene_codecs::points_writer::PointsWriteState {
                        directory: dir,
                        segment: merged_segment_name,
                        id: &merged_segment_id,
                        suffix: "",
                        max_doc: doc_count,
                    },
                    merge_points(
                        sources,
                        per_source_maps,
                        per_source_live_ids,
                        doc_id_maps,
                        merged_fields,
                        None,
                    )?,
                    native_merge,
                )?);
            } else {
                // One field's points are gathered at a time, as its tree is
                // written.
                let fields = points_candidates(merged_fields).into_iter().flat_map(|n| {
                    match merge_points(
                        sources,
                        per_source_maps,
                        per_source_live_ids,
                        doc_id_maps,
                        merged_fields,
                        Some(n),
                    ) {
                        Ok(found) => found.into_iter().map(Ok).collect::<Vec<_>>(),
                        Err(e) => vec![Err(e)],
                    }
                });
                let (kdm, kdi, kdd) = points::write_merged_streaming(
                    fields,
                    points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
                    doc_count,
                    dir,
                    merged_segment_name,
                    &merged_segment_id,
                    "",
                    native_merge,
                )?;
                for (ext, bytes) in [("kdm", &kdm), ("kdi", &kdi), ("kdd", &kdd)] {
                    let name = format!("{merged_segment_name}.{ext}");
                    write_file(dir, &name, bytes)?;
                    files.push(name);
                }
            }
        }
        Ok(())
    }

    /// The merged `.fnm`, once every format has said what it wrote
    /// (`describe_written_files`).
    fn write_field_infos(
        &mut self,
        postings_suffixes: &[(i32, &'static str, u32)],
        doc_values_suffixes: &[(i32, &'static str, u32)],
        wrote_term_vectors: bool,
        merged_vectors: Option<&MergedVectorFiles>,
        files: &mut Vec<String>,
    ) -> Result<()> {
        let vector_field_numbers: Vec<(i32, Option<(&'static str, u32)>)> = merged_vectors
            .map(|v| v.written.clone())
            .unwrap_or_default();
        let points_field_numbers: Vec<i32> = self
            .merged_fields
            .iter()
            .filter(|f| f.point_dimension_count > 0)
            .map(|f| f.number)
            .collect();

        let fnm_name = format!("{}.fnm", self.segment_name);
        describe_written_files(
            &mut self.merged_fields,
            postings_suffixes,
            doc_values_suffixes,
            wrote_term_vectors,
            &vector_field_numbers,
            &points_field_numbers,
        );
        crate::index_writer::before_write(
            self.options.codec.as_deref(),
            lucene_codecs::codec::WritePoint::FieldInfos,
        )?;
        let fnm = field_infos::write(&self.merged_fields, &self.segment_id, "");
        write_file(self.dir, &fnm_name, &fnm)?;
        files.push(fnm_name);
        Ok(())
    }

    fn write_vector_files(
        &self,
        vector_files: &MergedVectorFiles,
        files: &mut Vec<String>,
    ) -> Result<()> {
        files.extend(vector_files.written_early.iter().cloned());
        for (suffix, group_files) in &vector_files.groups {
            for (ext, bytes) in group_files {
                let name = crate::index_writer::segment_file_name(self.segment_name, suffix, ext);
                write_file(self.dir, &name, bytes)?;
                files.push(name);
            }
        }
        Ok(())
    }

    /// The `.si`, last: it lists every file of the segment, itself included.
    fn write_segment_info(
        &self,
        files: &mut Vec<String>,
        stored: &StoredFormat,
        sort_fields: Option<&[MergeSortKeySpec<'_>]>,
        codec_name: &str,
        lucene_version: LuceneVersion,
    ) -> Result<SegmentCommitInfo> {
        let SegmentWriter {
            dir,
            sources,
            options,
            segment_name: merged_segment_name,
            segment_id: merged_segment_id,
            doc_count,
            ..
        } = *self;
        let (stored_mode, lz4_hc) = (stored.mode, stored.lz4_hc);
        // `Lucene99SegmentInfoFormat.write` does `si.addFile(fileName)` *before*
        // writing, so a real Lucene `.si` lists itself -- and `IndexFileDeleter`
        // reference-counts from exactly this set, so a merged `.si` absent from
        // its own file list is a file nothing holds a reference to (and a
        // `CheckIndex` failure: `si.files_lists_itself`). The flush path
        // (`segment_writer`) already did this; the merge did not.
        let si_name = format!("{merged_segment_name}.si");
        files.push(si_name.clone());

        let si = SegmentInfo {
            id: merged_segment_id,
            version: lucene_version,
            // `SegmentMerger`'s constructor, not this writer's own version: see
            // [`merged_min_version`].
            min_version: merged_min_version(sources, lucene_version),
            doc_count,
            is_compound_file: false,
            // `IndexWriter.mergeMiddle`: the merged segment holds blocks iff any
            // source did -- see [`MergeSource::has_blocks`].
            has_blocks: sources.iter().any(|s| s.has_blocks),
            diagnostics: {
                let mut diagnostics = options.kind.diagnostics(sources.len(), lucene_version);
                for (key, value) in &options.extra_diagnostics {
                    // `SegmentInfo.addDiagnostics`: a later value replaces an earlier one.
                    match diagnostics.iter_mut().find(|(k, _)| k == key) {
                        Some(slot) => slot.1.clone_from(value),
                        None => diagnostics.push((key.clone(), value.clone())),
                    }
                }
                diagnostics
            },
            files: files.clone(),
            attributes: {
                let mut attributes =
                    crate::segment_writer::stored_fields_mode_attribute(stored_mode);
                if lz4_hc {
                    attributes.push((
                        crate::segment_writer::LZ4_HC_ATTRIBUTE.to_string(),
                        crate::segment_writer::LZ4_HC_VALUE.to_string(),
                    ));
                }
                for (key, value) in &options.extra_attributes {
                    crate::index_writer::set_attribute(&mut attributes, key, value);
                }
                attributes
            },
            // The merged segment claims the sort exactly when this merge produced
            // it. A concatenating merge must not claim one (it would describe an
            // order the bytes do not have); the k-way merge genuinely preserves
            // the shared sort of its inputs, so it must.
            index_sort: sort_fields
                .map(|sort_fields| sort_fields.iter().map(|spec| spec.sort.clone()).collect()),
        };
        crate::index_writer::before_write(
            options.codec.as_deref(),
            lucene_codecs::codec::WritePoint::SegmentInfo,
        )?;
        let si_bytes = segment_info::write(&si, "");
        write_file(dir, &si_name, &si_bytes)?;
        // Not synced: like a flushed segment, a merged one becomes durable when
        // the first commit that names it syncs it (`IndexWriter.startCommit`).

        let info = SegmentCommitInfo {
            segment_name: merged_segment_name.to_string(),
            segment_id: merged_segment_id,
            codec_name: codec_name.to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            // `IndexWriter.mergeMiddle` line 5086: a merged segment gets its own
            // `StringHelper.randomId()` too, same as a flushed one.
            sci_id: Some(crate::segment_writer::derive_sci_id(&merged_segment_id)),
            field_infos_files: vec![],
            dv_update_files: vec![],
            ..Default::default()
        };
        Ok(info)
    }
}

/// Rejects a mis-shaped key table with a named error rather than a panic.
fn check_sort_key_tables(
    sources: &[MergeSource],
    sort_fields: Option<&[MergeSortKeySpec<'_>]>,
) -> Result<()> {
    // Reported, not asserted: this is a `pub` entry point, and a caller that
    // hands it a mis-shaped key table gets a named error rather than a panic
    // (the same call this module's `read`-side errors already make).
    if let Some(sort_fields) = sort_fields {
        if sort_fields.is_empty() {
            return Err(Error::EmptySortFields);
        }
        for spec in sort_fields {
            if spec.per_source_keys.len() != sources.len() {
                return Err(Error::SortKeysWrongLength {
                    field: spec.sort.field.clone(),
                    source_index: None,
                    expected: sources.len(),
                    found: spec.per_source_keys.len(),
                });
            }
            for (source_index, (source, keys)) in
                sources.iter().zip(spec.per_source_keys.iter()).enumerate()
            {
                if keys.len() != source.reader.max_doc().max(0) as usize {
                    return Err(Error::SortKeysWrongLength {
                        field: spec.sort.field.clone(),
                        source_index: Some(source_index),
                        expected: source.reader.max_doc().max(0) as usize,
                        found: keys.len(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Per-source live (pre-merge) doc ids, ascending.
fn live_doc_ids(sources: &[MergeSource]) -> Result<Vec<Vec<i32>>> {
    let mut per_source_live_ids: Vec<Vec<i32>> = Vec::with_capacity(sources.len());
    for (source_index, source) in sources.iter().enumerate() {
        let max_doc = source.reader.max_doc();
        // `docs/arithmetic-gate.md`'s crate rule: *never index a
        // `FixedBitSet` with an index bounded against anything other than
        // that bitset's own `len()`.* The doc id below is bounded by
        // `max_doc`, which comes off the source's `.fdm`, and the bitset
        // comes off its `.liv` -- two independent files. `FixedBitSet::get`
        // does `words[index >> 6]` behind a bare `debug_assert`, so a `.liv`
        // one word short of `max_doc` reads a **ghost bit** in release (a
        // silently wrong live/dead answer, i.e. a document merged or dropped
        // that should not have been), and one 64 or more bits short is an
        // index panic in both profiles.
        //
        // Checked once per source rather than per document: it is both the
        // bound the loop needs and cheaper than the per-document `min` that
        // would otherwise hide the disagreement.
        if let Some(bits) = source.live_docs {
            if bits.len() != usize::try_from(max_doc).unwrap_or(0) {
                return Err(Error::LiveDocsLengthMismatch {
                    source_index,
                    max_doc,
                    live_docs_len: bits.len(),
                });
            }
        }
        let mut live_ids = Vec::new();
        for doc_id in 0..max_doc {
            // ARITH-adjacent: `doc_id < max_doc == bits.len()` by the check
            // above, so this index is bounded by the bitset's own length.
            let is_live = source
                .live_docs
                .map(|bits| bits.get(doc_id as usize))
                .unwrap_or(true);
            if is_live {
                live_ids.push(doc_id);
            }
        }
        per_source_live_ids.push(live_ids);
    }
    Ok(per_source_live_ids)
}

/// Every source that still contributes has its norms checksummed.
fn check_norms_integrity(sources: &[MergeSource], per_source_live_ids: &[Vec<i32>]) -> Result<()> {
    // Java: NormsConsumer.merge calls each source NormsProducer.checkIntegrity;
    // Lucene90NormsProducer checks the entire .nvd checksum before consuming it.
    // Preserve this API's fully-deleted-source exemption, but validate all bytes
    // (including deleted documents) of every source that still contributes.
    // Several fields borrow one file: pointer and length identify that borrowed
    // file without hashing its contents once per field. No semantic DISI scan is
    // implied by a valid checksum, just as in Java's checkIntegrity.
    //
    // Sources are checked in parallel; the first failure in source order is
    // the one reported, as the serial loop did.

    in_source_order(sources.par_iter().zip(per_source_live_ids).map(
        |(source, live_ids)| -> Result<()> {
            if live_ids.is_empty() {
                return Ok(());
            }
            let mut checked = HashSet::new();
            for norms in source.norms {
                if checked.insert((norms.data.as_ptr(), norms.data.len())) {
                    let footer = norms
                        .data
                        .len()
                        .saturating_sub(lucene_store::codec_util::FOOTER_LENGTH);
                    lucene_store::codec_util::check_whole_file_footer(norms.data, footer)?;
                }
            }
            Ok(())
        },
    ))?;
    Ok(())
}

/// The merged segment's physical order: concatenated, sorted, or reordered.
fn plan_doc_order(
    sources: &[MergeSource],
    sort_fields: Option<&[MergeSortKeySpec<'_>]>,
    options: &MergeOptions,
    per_source_live_ids: &[Vec<i32>],
    per_source_max_doc: &[i32],
) -> Result<Vec<(usize, i32)>> {
    Ok(match (sort_fields, &options.reorder) {
        (None, None) => concat_doc_order(per_source_live_ids),
        (Some(sort_fields), None) => sorted_doc_order(sort_fields, per_source_live_ids),
        (None, Some(new_to_old)) => reordered_doc_order(new_to_old, per_source_max_doc, sources)?,
        (Some(_), Some(_)) => {
            return Err(Error::InvalidReorder(
                "a merge of index-sorted segments cannot be reordered".to_string(),
            ))
        }
    })
}
