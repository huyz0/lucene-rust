//! Opening one segment: [`SegmentReader::open`] reads the `.si`, then opens
//! the files of every format the segment has (checked against the
//! segment's own file list, loose or inside its compound file), one stage
//! per format, and assembles the reader. A format with only some of its
//! files is a damaged segment and fails the open.

use super::*;

/// Where a segment's files are read from, and what names them.
struct SegmentFiles<'a> {
    dir: &'a dyn Directory,
    /// The `.cfs`/`.cfe` archive of a compound segment.
    compound: Option<CompoundArchive>,
    si: &'a SegmentInfo,
    segment_name: &'a str,
    segment_id: [u8; ID_LENGTH],
}

/// A segment's norms metadata and data, when it has norms.
type NormsFiles = (Option<Arc<Norms>>, Option<Arc<Input>>);

/// The doc-values files a segment opened.
struct OpenedDocValues {
    meta: Option<Arc<DocValuesMeta>>,
    data: Option<Arc<Input>>,
    dvs: Vec<(Arc<Input>, String, Arc<doc_values::DocValuesMeta>)>,
    generations: Vec<DocValuesGeneration>,
}

impl SegmentReader {
    /// Opens one segment: reads its `.si`, then whichever of `.fnm`/
    /// `.tim`/`.tip`/`.tmd`/`.doc`/`.pos`/`.pay`/`.liv` it actually has
    /// (checked via `SegmentInfo.files` -- a segment missing postings, e.g.
    /// this port's stored-fields-only fixtures, legitimately has none of
    /// `.tim`/`.tip`/`.tmd`/`.doc`/`.pos`/`.pay`, and that is not an error).
    pub(super) fn open(
        dir: &dyn Directory,
        commit: &segment_infos::SegmentCommitInfo,
        doc_base: i32,
    ) -> Result<Self> {
        if commit.codec_name == "SimpleText" {
            return Self::open_simpletext(dir, commit, doc_base);
        }
        let segment_name = commit.segment_name.clone();
        let segment_id = commit.segment_id;

        let si_bytes = dir.open(&format!("{segment_name}.si"))?;
        // `Codec.forName(codecName).segmentInfoFormat().read(..)`: a segment
        // written by a `Lucene90`..`Lucene95` codec has no `hasBlocks` byte.
        let si: SegmentInfo =
            segment_info::parse_for_codec(&si_bytes, &segment_id, &commit.codec_name)?;

        // Every file the open below reads, fetched at once from a directory
        // that reads by range. Only a hint: each open reports its own error.
        let _ = dir.prefetch_files(&commit.files(&si.files));

        // Compound-file segments (`.cfs`/`.cfe`) pack every other codec file
        // into one archive -- see `open_segment_file`'s doc comment for how
        // every sub-file lookup below transparently reads from it instead of
        // `dir.open`-ing a loose file that wouldn't exist on disk.
        // A page-aligned `.fdt` (paged zstd stored fields) is opened with its
        // page size as the alignment hint, as is a `.cfs` holding one: the
        // compound writer starts such an entry on a page boundary.
        let stored_alignment = lucene_codecs::stored_fields::page_alignment(&si.attributes);
        let compound = if si.is_compound_file {
            Some(CompoundArchive::open(
                dir,
                &segment_name,
                &segment_id,
                stored_alignment,
            )?)
        } else {
            None
        };
        let files = SegmentFiles {
            dir,
            compound,
            si: &si,
            segment_name: &segment_name,
            segment_id,
        };

        let field_infos = read_field_infos(&files, commit)?;
        let ((fields, segment_suffix, doc_buf, pos_buf, pay_buf, postings_files), multi_format) =
            open_postings(&files, &field_infos)?;
        let live_docs = read_live_docs(&files, commit)?;
        let stored = open_stored_files(&files, stored_alignment)?;
        let kdm_buf = open_segment_file(dir, files.compound.as_ref(), &si.files, ".kdm")?;
        let kdi_buf = open_segment_file(dir, files.compound.as_ref(), &si.files, ".kdi")?;
        let kdd_buf = open_segment_file(dir, files.compound.as_ref(), &si.files, ".kdd")?;
        let OpenedDocValues {
            meta: dv_meta,
            data: dv_data,
            dvs,
            generations: dv_generations,
        } = open_doc_values(&files, &field_infos)?;
        let (norms_meta, norms_data) = open_norms(&files, &field_infos)?;
        let term_vectors = open_term_vectors(&files)?;
        let vectors = open_vector_groups(&files)?;
        let completion = open_completion(&files)?;
        check_metadata_footers(&files, &stored, &term_vectors, &vectors, &kdm_buf)?;
        let codec_files = open_codec_extra_files(&files, commit)?;

        Ok(SegmentReader {
            stored_serial: next_stored_serial(),
            codec_files: Arc::new(codec_files),
            attachments: Arc::default(),
            simpletext: None,
            num_docs: crate::reader::num_docs_of(si.doc_count, live_docs.as_deref()),
            term_vectors,
            term_vectors_meta: Arc::default(),
            vectors,
            completion,
            core_cache: crate::reader::CacheHelper::new(),
            reader_cache: crate::reader::CacheHelper::new(),
            query_cache: Arc::default(),
            segment_name,
            index_sort: si.index_sort.clone().map(Arc::new),
            has_blocks: si.has_blocks,
            diagnostics: Arc::new(si.diagnostics.clone()),
            derived_source: crate::derived_source::DerivedFields::from_attributes(
                &si.attributes,
                &field_infos,
            )
            .map(Arc::new),
            dvs,
            skip_indexes: Arc::default(),
            block_parents: Arc::default(),
            docs_with_field: Arc::default(),
            max_doc: si.doc_count,
            doc_base,
            segment_id,
            del_gen: commit.del_gen,
            field_infos_gen: commit.field_infos_gen,
            doc_values_gen: commit.doc_values_gen,
            segment_suffix,
            fields,
            doc_buf,
            postings_files,
            pos_buf,
            pay_buf,
            live_docs,
            field_infos,
            dv_data,
            kdm_buf,
            kdi_buf,
            kdd_buf,
            stored,
            points_meta: Arc::default(),
            // A multi-format segment's postings were validated format by
            // format as they were opened and concatenated.
            postings_validated: Arc::new(std::sync::atomic::AtomicBool::new(multi_format)),
            postings_stats: None,
            dv_meta,
            dv_generations,
            lazy_doc_values: None,
            norms_meta,
            norms_data,
            hidden: None,
            masked: None,
        })
    }
}

// `IndexWriter.readFieldInfos`: a segment that has taken doc-values
// updates has a *generational* `.fnm` (always outside any compound
// file), and it is the only one recording the updated fields'
// `FieldInfo.docValuesGen`. Reading the base one instead reports every
// field at generation -1 and serves superseded values.
fn read_field_infos(
    files: &SegmentFiles<'_>,
    commit: &segment_infos::SegmentCommitInfo,
) -> Result<Arc<FieldInfos>> {
    let (dir, compound, si) = (files.dir, &files.compound, files.si);
    let (segment_name, segment_id) = (files.segment_name, files.segment_id);
    let field_infos = if commit.field_infos_gen != -1 {
        let name = field_updates::field_infos_gen_file_name(segment_name, commit.field_infos_gen);
        let bytes = dir.open(&name)?;
        let suffix = lucene_util::base36::to_base36(commit.field_infos_gen);
        Arc::new(field_infos::parse(&bytes, &segment_id, &suffix)?)
    } else {
        let fnm_bytes =
            open_segment_file(dir, compound.as_ref(), &si.files, ".fnm")?.ok_or_else(|| {
                Error::Store(lucene_store::Error::Corrupted(format!(
                    "segment {segment_name} has no .fnm file"
                )))
            })?;
        Arc::new(field_infos::parse(&fnm_bytes, &segment_id, "")?)
    };
    Ok(field_infos)
}

/// The postings files: one group, or one per routed postings format when the
/// segment's fields were routed to several. Also whether it is the latter.
fn open_postings(
    files: &SegmentFiles<'_>,
    field_infos: &Arc<FieldInfos>,
) -> Result<(PostingsFiles, bool)> {
    let (dir, compound, si) = (files.dir, &files.compound, files.si);
    let (segment_name, segment_id) = (files.segment_name, files.segment_id);
    let postings_names = compound.as_ref().map_or_else(
        || si.files.clone(),
        |c| c.entries.names().map(str::to_string).collect(),
    );
    per_field_postings::validate_routed_groups(&field_infos.fields, &postings_names)
        .map_err(|message| Error::Store(lucene_store::Error::Corrupted(message)))?;
    let tim_bytes = open_segment_file(dir, compound.as_ref(), &si.files, ".tim")?;
    let tip_bytes = open_segment_file(dir, compound.as_ref(), &si.files, ".tip")?;
    let tmd_bytes = open_segment_file(dir, compound.as_ref(), &si.files, ".tmd")?;
    let found = [&tim_bytes, &tip_bytes, &tmd_bytes]
        .iter()
        .filter(|f| f.is_some())
        .count();
    // `PerFieldPostingsFormat`: a segment whose fields were routed to
    // more than one postings format has one `.tim`/`.tip`/`.tmd` (and
    // `.doc`/`.pos`/`.pay`) per format and suffix.
    let postings_suffixes = per_field_postings::group_suffixes(&postings_names, segment_name);
    // A group with another terms dictionary is read through the same
    // path: `open_groups` rewrites its dictionary as a block tree.
    let multi_format = postings_suffixes.len() > 1
        || postings_suffixes.iter().any(|s| {
            per_field_postings::is_foreign_format(per_field_postings::format_of_suffix(s))
        });

    let (fields, segment_suffix, doc_buf, pos_buf, pay_buf, postings_files) = if multi_format {
        open_per_field_postings(
            dir,
            compound.as_ref(),
            &si.files,
            &postings_suffixes,
            field_infos,
            &segment_id,
            si.doc_count,
        )?
    } else if found == 3 {
        // Suffix is embedded in the sub-file's own name: strip the
        // `<segment_name>_` prefix (loose files, e.g.
        // `_0_Lucene104_0.tim` -> `Lucene104_0`) or the leading `_`
        // (compound entries, already stripped of the segment-name prefix
        // by `IndexFileNames.stripSegmentName` when packed -- e.g.
        // `_Lucene104_0.tim` -> `Lucene104_0`), then the `.tim`
        // extension.
        let tim_file_name = find_segment_file_name(&si.files, compound.as_ref(), ".tim")
            .expect("found == 3 implies a .tim entry exists");
        // No-codec-suffix case first (this port's own writer, e.g. loose
        // `_0.tim`): the generic strip-and-derive logic below would
        // otherwise misparse the segment name's own trailing digit as a
        // bogus suffix (`_0.tim` -> strip leading `_` -> `0.tim` ->
        // suffix `"0"`, wrong) -- same fix as the `.dvm` case below.
        let segment_suffix = codec_suffix_of(&tim_file_name, segment_name, ".tim");

        // `open_shared`, not `open`: `open` copies the whole
        // `.tim`/`.tip` into fresh allocations because the returned
        // `BlockTreeFields` navigates them for its whole life. On the
        // benchmark corpus that copy is 4.7 MB and 199 us -- 34% of a
        // 579 us `DirectoryReader::open`, and the single largest thing
        // in it. `open_shared`
        // takes a [`blocktree::SharedBytes`] (a type-erased shared
        // owner), and `open_segment_file` already hands back an
        // `Arc<Input>` -- for a `MmapDirectory` that *is* the mapping, so
        // the bytes are never copied at all. Measured in
        // `benches/directory_reader_open.rs`: 199 us -> 0.5 us for the
        // dictionary open, and **579 us -> 121 us (4.8x)** for the whole
        // reader -- more than the copy alone, because the copy also
        // first-touched 4.7 MB of the mapping that the lazy dictionary now
        // never faults in.
        let fields = blocktree::open_shared(
            tim_bytes.clone().unwrap() as blocktree::SharedBytes,
            tip_bytes.clone().unwrap() as blocktree::SharedBytes,
            tmd_bytes.as_ref().unwrap(),
            field_infos,
            &segment_id,
            &segment_suffix,
            si.doc_count,
        )?;

        let doc_buf = open_segment_file(dir, compound.as_ref(), &si.files, ".doc")?;
        let pos_buf = open_segment_file(dir, compound.as_ref(), &si.files, ".pos")?;
        let pay_buf = open_segment_file(dir, compound.as_ref(), &si.files, ".pay")?;

        let postings_files = [
            tim_bytes.clone(),
            tip_bytes.clone(),
            doc_buf.clone(),
            pos_buf.clone(),
            pay_buf.clone(),
        ]
        .into_iter()
        .flatten()
        .collect();
        (
            Arc::new(fields),
            segment_suffix,
            doc_buf,
            pos_buf,
            pay_buf,
            postings_files,
        )
    } else if found == 0 {
        (
            Arc::new(BlockTreeFields::empty()),
            String::new(),
            None,
            None,
            None,
            Vec::new(),
        )
    } else {
        return Err(Error::PartialBlockTreeFiles {
            segment: segment_name.to_string(),
            found,
        });
    };
    Ok((
        (
            fields,
            segment_suffix,
            doc_buf,
            pos_buf,
            pay_buf,
            postings_files,
        ),
        multi_format,
    ))
}

/// `.liv` exists only when the segment has deletions.
fn read_live_docs(
    files: &SegmentFiles<'_>,
    commit: &segment_infos::SegmentCommitInfo,
) -> Result<Option<Arc<FixedBitSet>>> {
    let (dir, si) = (files.dir, files.si);
    let (segment_name, segment_id) = (files.segment_name, files.segment_id);
    let live_docs = if commit.del_gen != -1 {
        let liv_name = liv_file_name(segment_name, commit.del_gen);
        let liv_bytes = dir.open(&liv_name)?;
        Some(Arc::new(live_docs::parse(
            &liv_bytes,
            &segment_id,
            commit.del_gen,
            si.doc_count as usize,
            commit.del_count as usize,
        )?))
    } else {
        None
    };
    Ok(live_docs)
}

fn open_stored_files(
    files: &SegmentFiles<'_>,
    stored_alignment: Option<usize>,
) -> Result<Option<Arc<StoredFiles>>> {
    let (dir, compound, si) = (files.dir, &files.compound, files.si);
    let (segment_name, _segment_id) = (files.segment_name, files.segment_id);
    let stored = match (
        open_segment_file_advised(dir, compound.as_ref(), &si.files, ".fdt", stored_alignment)?,
        open_segment_file(dir, compound.as_ref(), &si.files, ".fdx")?,
        open_segment_file(dir, compound.as_ref(), &si.files, ".fdm")?,
    ) {
        (Some((fdt, fdt_advice)), Some(fdx), Some(fdm)) => {
            let name = find_segment_file_name(&si.files, compound.as_ref(), ".fdt")
                .expect("an opened .fdt has an entry");
            Some(Arc::new(StoredFiles {
                fdt,
                fdx,
                fdm,
                suffix: codec_suffix_of(&name, segment_name, ".fdt"),
                will_need_chunks: fdt_advice == lucene_store::read_advice::ReadAdvice::Random,
                meta: OnceLock::new(),
            }))
        }
        (None, None, None) => None,
        // A partial set is a damaged segment, not one without stored
        // fields: answering "no fields" would hand the fetch phase a
        // document with no `_source` and no `_id`.
        (fdt, fdx, fdm) => {
            return Err(Error::PartialStoredFieldsFiles {
                segment: segment_name.to_string(),
                found: [fdt.is_some(), fdx.is_some(), fdm.is_some()]
                    .iter()
                    .filter(|p| **p)
                    .count(),
            })
        }
    };
    Ok(stored)
}

// `.dvm`/`.dvd` (the skipper's `.dvs` is opened beside them and
// checksummed by `check_integrity`): present together or not at all,
// exactly like `.tim`/`.tip`/`.tmd` above -- a segment with no
// doc-values field opted in has neither (see `segment_writer.rs`'s "no
// pending values -> no files" contract). The codec suffix is derived
// the same way `.tim`'s is above -- this port's own writer uses `""`
// (matching the sparse/dense unit tests in `index_writer.rs`), but a
// real Lucene-written segment (e.g. `fixtures/data/compound_index/`)
// gives doc values their own codec suffix (`Lucene90_<n>`), independent
// of the postings suffix.
fn open_doc_values(
    files: &SegmentFiles<'_>,
    field_infos: &Arc<FieldInfos>,
) -> Result<OpenedDocValues> {
    let (dir, compound, si) = (files.dir, &files.compound, files.si);
    let (segment_name, segment_id) = (files.segment_name, files.segment_id);
    // The first doc-values format's pair (by suffix); a `.dvd` found on its
    // own by extension could belong to another format's `.dvm`.
    let dv_suffixes = segment_codec_suffixes(&si.files, compound.as_ref(), segment_name, ".dvm");
    let dv_ext = |ext: &str| match dv_suffixes.first() {
        Some(suffix) if !suffix.is_empty() => format!("_{suffix}{ext}"),
        _ => ext.to_string(),
    };
    let dvm_bytes = open_segment_file(dir, compound.as_ref(), &si.files, &dv_ext(".dvm"))?;
    let dvd_bytes = open_segment_file(dir, compound.as_ref(), &si.files, &dv_ext(".dvd"))?;
    let mut dvs = Vec::new();
    let mut base_dv_suffix = None;
    let (dv_meta, dv_data) = match (dvm_bytes, dvd_bytes) {
        (Some(dvm), Some(dvd)) => {
            let dvm_file_name =
                find_segment_file_name(&si.files, compound.as_ref(), &dv_ext(".dvm"))
                    .expect("dvm_bytes.is_some() implies a .dvm entry exists");
            // No-codec-suffix case first (this port's own writer, e.g.
            // loose `_0.dvm`): the generic strip-and-derive logic below
            // would otherwise misparse the segment name's own trailing
            // digit as a bogus suffix (`_0.dvm` -> strip leading `_` ->
            // `0.dvm` -> suffix `"0"`, wrong).
            let dv_suffix = codec_suffix_of(&dvm_file_name, segment_name, ".dvm");
            let (version, mut meta) =
                doc_values::parse_meta(&dvm, &segment_id, &dv_suffix, field_infos)?;
            if doc_values::has_disi_file(version) {
                let dvp = open_segment_file(dir, compound.as_ref(), &si.files, &dv_ext(".dvp"))?
                    .ok_or_else(|| missing_dvp(segment_name))?;
                meta.attach_disi(doc_values::DisiBytes::new(dvp));
            }
            let meta = Arc::new(meta);
            if let Some(bytes) =
                open_segment_file(dir, compound.as_ref(), &si.files, &dv_ext(".dvs"))?
            {
                dvs.push((bytes, dv_suffix.clone(), Arc::clone(&meta)));
            }
            base_dv_suffix = Some(dv_suffix);
            (Some(meta), Some(dvd))
        }
        _ => (None, None),
    };

    // `SegmentDocValuesProducer`: one producer per distinct
    // `FieldInfo.docValuesGen`, over a one-field `FieldInfos` -- the
    // generation's `.dvm` describes exactly that field, so handing it the
    // whole list would accept a `.dvm` naming a field it is not for.
    let mut dv_generations: Vec<DocValuesGeneration> = Vec::new();
    // `PerFieldDocValuesFormat.FieldsReader`: a segment whose doc-values
    // fields were routed to more than one format has a `.dvm`/`.dvd` per
    // format and suffix. The first is the base pair above; every field of
    // another format is served from that format's own pair, through the
    // same per-field routing a doc-values update uses.
    for suffix in &dv_suffixes {
        if Some(suffix) == base_dv_suffix.as_ref() {
            continue;
        }
        let open = |ext: &str| {
            open_segment_file(
                dir,
                compound.as_ref(),
                &si.files,
                &format!("_{suffix}{ext}"),
            )
        };
        let (Some(meta_bytes), Some(data)) = (open(".dvm")?, open(".dvd")?) else {
            continue;
        };
        let (version, mut meta) =
            doc_values::parse_meta(&meta_bytes, &segment_id, suffix, field_infos)?;
        if doc_values::has_disi_file(version) {
            let dvp = open(".dvp")?.ok_or_else(|| missing_dvp(segment_name))?;
            meta.attach_disi(doc_values::DisiBytes::new(dvp));
        }
        let meta = Arc::new(meta);
        if let Some(bytes) = open(".dvs")? {
            dvs.push((bytes, suffix.clone(), Arc::clone(&meta)));
        }
        for field in &field_infos.fields {
            if field.doc_values_gen == -1 && dv_per_field_suffix(field).as_ref() == Some(suffix) {
                dv_generations.push(DocValuesGeneration {
                    field_number: field.number,
                    meta: Arc::clone(&meta),
                    data: Arc::clone(&data),
                });
            }
        }
    }
    for field in &field_infos.fields {
        if field.doc_values_gen == -1 {
            continue;
        }
        let gen = field.doc_values_gen;
        // No format attributes means no producer is registered for the
        // field at all -- `PerFieldDocValuesFormat.FieldsReader` skips it
        // ("null formatName means the field is in fieldInfos, but has no
        // docvalues"), and so does this.
        let Some(per_field) = dv_per_field_suffix(field) else {
            continue;
        };
        let suffix = field_updates::generation_segment_suffix(gen, &per_field);
        let meta_name = field_updates::generation_file_name(segment_name, gen, &per_field, "dvm");
        let data_name = field_updates::generation_file_name(segment_name, gen, &per_field, "dvd");
        let meta_bytes = dir.open(&meta_name)?;
        let data_bytes = dir.open(&data_name)?;
        let only = FieldInfos {
            fields: vec![field.clone()],
        };
        let (version, mut meta) = doc_values::parse_meta(&meta_bytes, &segment_id, &suffix, &only)?;
        if doc_values::has_disi_file(version) {
            let disi_name =
                field_updates::generation_file_name(segment_name, gen, &per_field, "dvp");
            meta.attach_disi(doc_values::DisiBytes::new(Arc::new(dir.open(&disi_name)?)));
        }
        dv_generations.push(DocValuesGeneration {
            field_number: field.number,
            meta: Arc::new(meta),
            data: Arc::new(data_bytes),
        });
    }
    Ok(OpenedDocValues {
        meta: dv_meta,
        data: dv_data,
        dvs,
        generations: dv_generations,
    })
}

/// A format-3 `.dvm` without its `.dvp`: the segment lost a file.
fn missing_dvp(segment_name: &str) -> super::Error {
    lucene_store::Error::Corrupted(format!(
        "segment {segment_name}: doc values are format 3 but the .dvp file is missing"
    ))
    .into()
}

// `.nvm`/`.nvd`: present together or not at all, like `.tim`/`.tip`/
// `.tmd` and `.dvm`/`.dvd` above. The codec suffix is derived the same
// way theirs is -- this port's own writer uses `""`, a real
// Lucene-written segment gives norms their own `Lucene90_<n>` suffix.
fn open_norms(files: &SegmentFiles<'_>, field_infos: &Arc<FieldInfos>) -> Result<NormsFiles> {
    let (dir, compound, si) = (files.dir, &files.compound, files.si);
    let (segment_name, segment_id) = (files.segment_name, files.segment_id);
    let nvm_bytes = open_segment_file(dir, compound.as_ref(), &si.files, ".nvm")?;
    let nvd_bytes = open_segment_file(dir, compound.as_ref(), &si.files, ".nvd")?;
    let (norms_meta, norms_data) = match (nvm_bytes, nvd_bytes) {
        (Some(nvm), Some(nvd)) => {
            let nvm_file_name = find_segment_file_name(&si.files, compound.as_ref(), ".nvm")
                .expect("nvm_bytes.is_some() implies a .nvm entry exists");
            let norms_suffix = codec_suffix_of(&nvm_file_name, segment_name, ".nvm");
            let (_, meta) = norms::parse_meta(&nvm, &segment_id, &norms_suffix)?;
            // `Lucene90NormsProducer.readFields` validates every entry
            // against `FieldInfos` as it reads them and fails the segment
            // open. This port's `norms::parse_meta` takes no
            // `FieldInfos` (see `norms::validate_fields`), so the check
            // is the next line rather than part of the parse -- but it
            // happens at the same moment, on the same path, with the same
            // consequence.
            norms::validate_fields(&meta, field_infos)?;
            norms::check_data_header_footer(&nvd, &segment_id, &norms_suffix)?;
            (Some(Arc::new(meta)), Some(nvd))
        }
        (None, None) => (None, None),
        // Unlike `.dvm`/`.dvd`, a half-present pair here is not degraded to
        // "no norms": scoring without norms is silently wrong (every
        // document at `UNNORMED_FIELD_LENGTH`), not merely absent, so a
        // corrupt pair must surface the way `.tim`/`.tip`/`.tmd`'s does.
        _ => {
            return Err(Error::PartialNormsFiles {
                segment: segment_name.to_string(),
            })
        }
    };
    Ok((norms_meta, norms_data))
}

/// `.tvd`/`.tvx`/`.tvm`: all three or none, like stored fields.
fn open_term_vectors(files: &SegmentFiles<'_>) -> Result<Option<Arc<CodecFiles>>> {
    let (dir, compound, si) = (files.dir, &files.compound, files.si);
    let (segment_name, _segment_id) = (files.segment_name, files.segment_id);
    let term_vectors = match (
        open_segment_file(dir, compound.as_ref(), &si.files, ".tvd")?,
        open_segment_file(dir, compound.as_ref(), &si.files, ".tvx")?,
        open_segment_file(dir, compound.as_ref(), &si.files, ".tvm")?,
    ) {
        (Some(data), Some(index), Some(meta)) => {
            let name = find_segment_file_name(&si.files, compound.as_ref(), ".tvd")
                .expect("an opened .tvd has an entry");
            Some(Arc::new(CodecFiles {
                data,
                index: Some(index),
                meta,
                graph_meta: None,
                ivf: None,
                scalar_quantized: None,
                hnsw_codes: None,
                disk_hnsw: None,
                flat_reader: OnceLock::new(),
                graph_reader: OnceLock::new(),
                suffix: codec_suffix_of(&name, segment_name, ".tvd"),
            }))
        }
        (None, None, None) => None,
        (d, x, m) => {
            return Err(Error::PartialTermVectorsFiles {
                segment: segment_name.to_string(),
                found: [d.is_some(), x.is_some(), m.is_some()]
                    .iter()
                    .filter(|p| **p)
                    .count(),
            })
        }
    };
    Ok(term_vectors)
}

// `.vec`/`.vemf`: the flat vectors every HNSW field stores beside its
// graph (`.vex`/`.vem`, read by `crate::vector_query`'s own callers).
// One pair per `PerFieldKnnVectorsFormat` instance
// (`FieldsReader` opens a reader per suffix the fields name). A
// `.vec` without `.vemf` is a pre-`Lucene99` vectors format (`.vec` +
// `.vem` only), read through the backward codecs, not this reader.
// `Lucene99HnswVectorsReader` opens its graph's `.vem` through a
// `ChecksumIndexInput` (checked with the other metadata below) and
// checksums `.vex` in `checkIntegrity`, so both are opened here even
// though this reader searches the flat vectors (`.vex` rides as the
// pair's `index`).
fn open_vector_groups(files: &SegmentFiles<'_>) -> Result<Vec<Arc<CodecFiles>>> {
    let (dir, compound, si) = (files.dir, &files.compound, files.si);
    let segment_name = files.segment_name;
    let mut vectors = Vec::new();

    for suffix in segment_codec_suffixes(&si.files, compound.as_ref(), segment_name, ".vemf") {
        // No suffix: a codec whose `knnVectorsFormat()` is the format
        // itself, not `PerFieldKnnVectorsFormat` (`ConfigurableMCodec`).
        let open = |ext: &str| {
            open_segment_file(
                dir,
                compound.as_ref(),
                &si.files,
                &suffixed_ext(&suffix, ext),
            )
        };
        if let (Some(data), Some(meta)) = (open(".vec")?, open(".vemf")?) {
            let ivf = match (open(".ivm")?, open(".ivp")?, open(".ivv")?) {
                (Some(meta), Some(postings), Some(vectors)) => Some(IvfFiles {
                    meta,
                    postings,
                    vectors,
                    reader: OnceLock::new(),
                }),
                (None, None, None) => None,
                _ => {
                    return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                        "segment {segment_name}: an IVF index without all of its \
                             .ivm, .ivp and .ivv files"
                    ))))
                }
            };
            let scalar_quantized = match (open(".vemq")?, open(".veq")?) {
                (Some(meta), Some(data))
                    if lucene_codecs::scalar_quantized_vectors::is_meta_file(&meta) =>
                {
                    Some(ScalarQuantizedFiles {
                        meta,
                        data,
                        reader: OnceLock::new(),
                    })
                }
                _ => None,
            };
            let hnsw_codes = match (open(".vqm")?, open(".vqd")?) {
                (Some(meta), Some(data)) => Some(HnswCodesFiles {
                    meta,
                    data,
                    reader: OnceLock::new(),
                }),
                (None, None) => None,
                _ => {
                    return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                        "segment {segment_name}: quantized HNSW codes without both of \
                             their .vqm and .vqd files"
                    ))))
                }
            };
            let disk_hnsw = match (open(".dhm")?, open(".dhc")?, open(".dhn")?) {
                (Some(meta), Some(codes), Some(nodes)) => Some(DiskHnswFiles {
                    meta,
                    codes,
                    nodes,
                    reader: OnceLock::new(),
                }),
                (None, None, None) => None,
                _ => {
                    return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                        "segment {segment_name}: a disk HNSW layout without all of its \
                             .dhm, .dhc and .dhn files"
                    ))))
                }
            };
            vectors.push(Arc::new(CodecFiles {
                data,
                index: open(".vex")?,
                meta,
                graph_meta: open(".vem")?,
                ivf,
                scalar_quantized,
                hnsw_codes,
                disk_hnsw,
                flat_reader: OnceLock::new(),
                graph_reader: OnceLock::new(),
                suffix,
            }));
        }
    }
    Ok(vectors)
}

// `CompletionFieldsProducer`: one per `CompletionXX` instance, under
// the same suffix as its delegate's postings files. The suffix names
// the format, whose name is the files' codec name.
fn open_completion(
    files: &SegmentFiles<'_>,
) -> Result<Vec<Arc<lucene_codecs::completion::CompletionFields>>> {
    let (dir, compound, si) = (files.dir, &files.compound, files.si);
    let (segment_name, segment_id) = (files.segment_name, files.segment_id);
    let mut completion = Vec::new();
    for suffix in segment_codec_suffixes(&si.files, compound.as_ref(), segment_name, ".cmp") {
        let format = per_field_postings::completion_format_of(&suffix).ok_or_else(|| {
            Error::Store(lucene_store::Error::IllegalArgument(format!(
                "segment {segment_name}: _{suffix}.cmp is not a completion postings format \
                     Lucene 10.5.0 reads"
            )))
        })?;
        let open = |ext: &str| {
            open_segment_file(
                dir,
                compound.as_ref(),
                &si.files,
                &format!("_{suffix}{ext}"),
            )
        };
        // Java's producer opens both files and throws on either missing:
        // a listed `.cmp` without its `.lkp` is a corrupt segment.
        let required = |ext: &str| {
            open(ext)?.ok_or_else(|| {
                Error::Store(lucene_store::Error::Corrupted(format!(
                    "segment {segment_name} has no _{suffix}{ext} file"
                )))
            })
        };
        let cmp = required(".cmp")?;
        let lkp = required(".lkp")?;
        completion.push(Arc::new(lucene_codecs::completion::CompletionFields::open(
            format,
            &segment_id,
            &suffix,
            &cmp,
            lkp.to_vec(),
        )?));
    }
    Ok(completion)
}

// Java reads each format's metadata through a `ChecksumIndexInput`
// when the segment opens and `checkFooter`s it there. This reader
// parses `.fdm`, `.tvm`, `.vemf` and `.kdm` on first use and never
// reads `.psm` (its impact maxima matter to `CheckIndex` only) or the
// graph's `.vem` (its searches are exhaustive), so
// their checksums are verified here: otherwise a damaged one would
// pass both the open and `check_integrity`.
fn check_metadata_footers(
    files: &SegmentFiles<'_>,
    stored: &Option<Arc<StoredFiles>>,
    term_vectors: &Option<Arc<CodecFiles>>,
    vectors: &[Arc<CodecFiles>],
    kdm_buf: &Option<Arc<Input>>,
) -> Result<()> {
    let (dir, compound, si) = (files.dir, &files.compound, files.si);
    let segment_name = files.segment_name;
    let psm = segment_codec_suffixes(&si.files, compound.as_ref(), segment_name, ".psm")
        .into_iter()
        .map(|suffix| {
            let ext = if suffix.is_empty() {
                ".psm".to_string()
            } else {
                format!("_{suffix}.psm")
            };
            open_segment_file(dir, compound.as_ref(), &si.files, &ext)
        })
        .collect::<Result<Vec<_>>>()?;
    for meta in stored
        .iter()
        .map(|s| &s.fdm)
        .chain(term_vectors.iter().chain(vectors).map(|c| &c.meta))
        .chain(psm.iter().flatten())
        .chain(vectors.iter().filter_map(|c| c.graph_meta.as_ref()))
        .chain(kdm_buf.iter())
    {
        let end = meta
            .len()
            .saturating_sub(lucene_store::codec_util::FOOTER_LENGTH);
        lucene_store::codec_util::check_whole_file_footer(meta, end)?;
    }
    Ok(())
}

fn open_codec_extra_files(
    files: &SegmentFiles<'_>,
    commit: &segment_infos::SegmentCommitInfo,
) -> Result<Vec<(&'static str, Arc<Input>)>> {
    let (dir, compound, si) = (files.dir, &files.compound, files.si);
    let mut codec_files = Vec::new();
    if let Some(codec) = lucene_codecs::codecs::for_name(&commit.codec_name) {
        for ext in codec.extra_files {
            if let Some(input) =
                open_segment_file(dir, compound.as_ref(), &si.files, &format!(".{ext}"))?
            {
                codec_files.push((*ext, input));
            }
        }
    }
    Ok(codec_files)
}
