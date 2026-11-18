use super::*;

/// Live documents of `sci` with a value in the soft-deletes field `soft`. A
/// compound segment's base files are read through its archive; its `.liv`
/// and update generations live beside it.
///
/// With `retention`, only the soft-deleted documents it no longer keeps are
/// counted -- what a merge would reclaim, the soft half of Java's
/// `MergePolicy.numDeletesToMerge` under `SoftDeletesRetentionMergePolicy`.
pub(super) fn count_soft_deletes(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    soft: &str,
    retention: Option<&SoftDeletesRetention>,
) -> Result<Option<i32>> {
    let si = segment_info::parse_for_commit(&dir.open(&format!("{}.si", sci.segment_name))?, sci)?;
    let compound = if si.is_compound_file {
        Some(CompoundReader::open(
            dir,
            &sci.segment_name,
            &sci.segment_id,
        )?)
    } else {
        None
    };
    let (dir, files): (&dyn Directory, Vec<String>) = match &compound {
        Some(c) => (
            c,
            si.files
                .iter()
                .filter(|f| !f.ends_with(".cfs") && !f.ends_with(".cfe"))
                .cloned()
                .chain(c.member_files())
                .collect(),
        ),
        None => (dir, si.files.clone()),
    };
    let infos = crate::field_updates::read_current_field_infos(dir, sci, &files)?;
    let column = |name: &str| -> Result<
        Option<(
            doc_values::DocValuesMeta,
            lucene_store::directory::Input,
            i32,
        )>,
    > {
        let Some(index) = infos
            .fields
            .iter()
            .position(|f| f.name == name && f.doc_values_type == DocValuesType::Numeric)
        else {
            return Ok(None);
        };
        let per_field = crate::field_updates::per_field_component(
            &infos.fields[index],
            &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        );
        Ok(
            crate::field_updates::read_current_column(dir, sci, &files, &infos, index, &per_field)?
                .map(|(meta, data)| (meta, data, infos.fields[index].number)),
        )
    };
    let Some((meta, data, number)) = column(soft)? else {
        return Ok(Some(0));
    };
    let Some(entry) = meta.numeric_entry(number) else {
        return Ok(Some(0));
    };
    let seq_column = match retention {
        Some(r) => column(&r.seq_no_field)?,
        None => None,
    };
    let mut seq_values = seq_column.as_ref().and_then(|(meta, data, number)| {
        Some(doc_values::NumericReader::new(
            data,
            meta.numeric_entry(*number)?,
        ))
    });
    let max_doc = usize::try_from(si.doc_count).unwrap_or(0);
    let live = if sci.del_gen >= 0 {
        Some(lucene_codecs::live_docs::parse(
            &dir.open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))?,
            &sci.segment_id,
            sci.del_gen,
            max_doc,
            usize::try_from(sci.del_count).unwrap_or(0),
        )?)
    } else {
        None
    };
    let column_err = Error::DocValuesRead;
    let mut values = doc_values::NumericReader::new(&data, entry);
    let mut count = 0i32;
    for doc in 0..si.doc_count {
        if live.as_ref().is_some_and(|l| !l.get_doc(doc)) {
            continue;
        }
        if values.value(doc).map_err(column_err)?.is_none() {
            continue;
        }
        if let Some(retention) = retention {
            let seq_no = match seq_values.as_mut() {
                Some(reader) => reader.value(doc).map_err(column_err)?,
                None => None,
            };
            if seq_no.is_some_and(|s| s >= retention.min_retained_seq_no) {
                continue;
            }
        }
        count = count.saturating_add(1);
    }
    Ok(Some(count))
}
