use super::*;

/// `PerFieldPostingsFormat.FieldsReader` for a segment with more than one
/// postings format ([`per_field_postings::open_groups`]): every format's
/// dictionary as one, and every format's `.doc`/`.pos`/`.pay` concatenated
/// -- so the one `DocInput` every consumer of a segment holds serves every
/// field. The copy is paid only by multi-format segments.
pub(super) fn open_per_field_postings(
    dir: &dyn Directory,
    compound: Option<&CompoundArchive>,
    files: &[String],
    suffixes: &[String],
    field_infos: &FieldInfos,
    segment_id: &[u8; ID_LENGTH],
    doc_count: i32,
) -> Result<PostingsFiles> {
    let mut opened = Vec::with_capacity(suffixes.len());
    for suffix in suffixes {
        let open = |ext: &str| open_segment_file(dir, compound, files, &format!("_{suffix}{ext}"));
        let format = per_field_postings::format_of_suffix(suffix);
        let extensions = per_field_postings::dictionary_extensions(format);
        let mut dictionary = Vec::with_capacity(extensions.len());
        for ext in extensions {
            match open(&format!(".{ext}"))? {
                Some(bytes) => dictionary.push(bytes),
                None if !per_field_postings::dictionary_file_required(format, ext) => {
                    dictionary.push(std::sync::Arc::new(lucene_store::Input::Owned(Vec::new())));
                }
                None => {
                    return Err(Error::PartialBlockTreeFiles {
                        segment: format!("{suffix} of a per-field segment"),
                        found: dictionary.len(),
                    })
                }
            }
        }
        opened.push((
            suffix,
            dictionary,
            open(".doc")?,
            open(".pos")?,
            open(".pay")?,
        ));
    }
    let groups: Vec<per_field_postings::GroupFiles<'_>> = opened
        .iter()
        .map(
            |(suffix, dictionary, doc, pos, pay)| per_field_postings::GroupFiles {
                suffix,
                dictionary: dictionary.iter().map(|d| &***d).collect(),
                doc: doc.as_deref().map(|d| &**d),
                pos: pos.as_deref().map(|d| &**d),
                pay: pay.as_deref().map(|d| &**d),
            },
        )
        .collect();
    // Every group's `.tim`/`.tip` as opened (a mapping, or a compound
    // member's slice), so the combined dictionary navigates them in place.
    let shared: Vec<Option<[lucene_codecs::blocktree::SharedBytes; 2]>> = opened
        .iter()
        .map(|(_, dictionary, ..)| match &dictionary[..] {
            [tim, tip, ..] => Some([tim.clone() as _, tip.clone() as _]),
            _ => None,
        })
        .collect();
    let combined = per_field_postings::open_groups_shared(
        &groups,
        &shared,
        field_infos,
        segment_id,
        doc_count,
    )?;
    let owned = |bytes: Vec<u8>| (!bytes.is_empty()).then(|| Arc::new(Input::Owned(bytes)));
    let postings_files = opened
        .iter()
        .flat_map(|(suffix, dictionary, doc, pos, pay)| {
            // A footerless dictionary (`RAMOnly`'s token) has nothing to check.
            let checksummed = per_field_postings::dictionary_extensions(
                per_field_postings::format_of_suffix(suffix),
            )
            .iter()
            .all(|ext| {
                !per_field_postings::is_unchecksummed_dictionary_file(&format!("_{suffix}.{ext}"))
            });
            dictionary
                .iter()
                .filter(move |_| checksummed)
                .cloned()
                .chain(
                    [doc.clone(), pos.clone(), pay.clone()]
                        .into_iter()
                        .flatten(),
                )
        })
        .collect();
    Ok((
        Arc::new(combined.fields),
        suffixes[0].clone(),
        owned(combined.doc),
        owned(combined.pos),
        owned(combined.pay),
        postings_files,
    ))
}
