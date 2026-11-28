//! A merged vector group: one format instance's flat vectors, quantized codes
//! and index files, streamed to the directory as they are built.

use super::*;

/// Writes `vectors` (`dimension` floats each) to `name`, each vector
/// unit-normalized as [`lucene_ivf::prepare_vectors`] does for a cosine
/// field, a few thousand at a time.
fn write_normalized_vectors(
    dir: &dyn Directory,
    name: &str,
    dimension: usize,
    vectors: &[f32],
) -> Result<()> {
    const ROWS: usize = 4096;
    let mut out = dir.create_output(name)?;
    let mut rows: Vec<f32> = Vec::new();
    let mut bytes: Vec<u8> = Vec::new();
    for chunk in vectors.chunks(dimension.max(1).saturating_mul(ROWS)) {
        rows.clear();
        rows.extend_from_slice(chunk);
        lucene_ivf::prepare_vectors(lucene_ivf::Similarity::Cosine, dimension, &mut rows);
        bytes.clear();
        for x in &rows {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        out.write_bytes(&bytes);
    }
    out.close()?;
    Ok(())
}

/// One `PerFieldKnnVectorsFormat` instance's merge
/// (`Lucene99HnswVectorsWriter.mergeOneField` or a flat format's
/// `mergeOneFlatVectorField`): the raw vectors, the quantized codes when the
/// format quantizes (`Lucene104ScalarQuantizedVectorsWriter
/// .mergeOneFlatVectorField`, centroids from `mergeAndRecalculateCentroids`),
/// then -- for a format with a graph -- the graphs, over the vectors just
/// written, scored with the quantized scorer for a quantized `FLOAT32` field
/// (`getRandomVectorScorerSupplierForMerge`) and the raw one otherwise.
/// Returns the instance's files by extension.
#[allow(clippy::too_many_arguments)]
pub(super) fn merge_vector_group(
    dir: &dyn Directory,
    merged_segment_name: &str,
    sources: &[MergeSource],
    doc_id_maps: &[Vec<i32>],
    plan: &[&(&FieldInfo, Vec<FieldSource>)],
    format: per_field_knn_vectors::KnnVectorsFormat,
    merged_max_doc: i32,
    merged_segment_id: &[u8; ID_LENGTH],
    suffix: &str,
    partitions: Option<crate::ivf_vectors::Partitions<'_>>,
    ivf_models: &[(i32, std::sync::Arc<lucene_ivf::SharedModel>)],
) -> Result<(GroupFiles, Vec<String>)> {
    let mut flat_writer =
        vectors::FlatVectorsWriter::new(merged_max_doc, merged_segment_id, suffix);
    // The merged vectors go to their file as they are copied: the merge
    // never holds the file whole.
    let vec_name = crate::index_writer::segment_file_name(merged_segment_name, suffix, "vec");
    flat_writer.stream_to(dir.create_output(&vec_name)?);
    for (merged_field, field_sources) in plan.iter().map(|p| (&p.0, &p.1)) {
        let encoding = merged_field.vector_encoding;
        let mut flat_sources: Vec<vectors::FlatVectorMergeSource<'_>> =
            Vec::with_capacity(field_sources.len());
        for fs in field_sources {
            let source_vectors = sources[fs.source_index]
                .vectors
                .expect("only sources with a flat reader are planned");
            let group = source_vectors.group_of(fs.original_field_number);
            let values = match encoding {
                VectorEncoding::Float32 => vectors::MergeSourceValues::Float32(
                    group.flat.float_vector_values(fs.original_field_number)?,
                ),
                VectorEncoding::Byte => vectors::MergeSourceValues::Byte(
                    group.flat.byte_vector_values(fs.original_field_number)?,
                ),
            };
            flat_sources.push(vectors::FlatVectorMergeSource {
                values,
                doc_map: &doc_id_maps[fs.source_index],
            });
        }
        flat_writer.merge_one_flat_vector_field(&vectors::MergedFlatVectorField {
            field_number: merged_field.number,
            encoding,
            similarity: merged_field.vector_similarity_function,
            dimension: merged_field.vector_dimension,
            sources: &flat_sources,
        })?;
    }
    let (vec_out, vemf_bytes) = flat_writer.finish_streamed();
    // Everything built over the vectors reads their file: the merge holds
    // no in-memory copy of every vector.
    if let Some(out) = vec_out {
        out.close()?;
    }
    let vec_input = dir.open(&vec_name)?;
    let vec_bytes: &[u8] = &vec_input;

    // Reopen exactly the bytes just written and build everything else over
    // *those*, so the graph's ordinals, the codes' ordinals and the flat
    // store's ordinals are the same fact rather than three derivations of it.
    let mut flat =
        vectors::FlatVectorsReader::open(&vemf_bytes, vec_bytes, merged_segment_id, suffix)?;
    flat.resolve_similarities(|number| {
        plan.iter()
            .find(|p| p.0.number == number)
            .map(|p| p.0.vector_similarity_function)
    })?;
    let mut files: Vec<(&'static str, Vec<u8>)> = Vec::new();
    let mut ivv_name: Option<String> = None;

    if let Some(encoding) = format.quantization() {
        let mut writer = scalar_quantized_vectors::ScalarQuantizedVectorsWriter::new(
            encoding,
            merged_max_doc,
            merged_segment_id,
            suffix,
        );
        for (merged_field, field_sources) in plan.iter().map(|p| (&p.0, &p.1)) {
            if merged_field.vector_encoding != VectorEncoding::Float32 {
                continue;
            }
            // Every source's vectors for the field, deleted ones included,
            // in its ordinal order -- what `calculateCentroid` iterates.
            let mut owned: Vec<OwnedQuantizedSource<'_>> = Vec::new();
            for fs in field_sources {
                let source = &sources[fs.source_index];
                let group = source
                    .vectors
                    .expect("only sources with a flat reader are planned")
                    .group_of(fs.original_field_number);
                let all = group
                    .flat
                    .float_vector_values(fs.original_field_number)?
                    .all_vectors()?;
                let live = match source.live_docs {
                    Some(live_docs) => {
                        let values = group.flat.float_vector_values(fs.original_field_number)?;
                        let mut flags =
                            Vec::with_capacity(usize::try_from(values.size()).unwrap_or(0));
                        for ord in 0..values.size() {
                            flags.push(live_docs.get_doc(values.ord_to_doc(ord)?));
                        }
                        Some(flags)
                    }
                    None => None,
                };
                owned.push((
                    group
                        .flat_quantized
                        .and_then(|q| q.centroid(fs.original_field_number)),
                    all,
                    source.live_docs.is_some(),
                    live,
                ));
            }
            let merge_sources: Vec<scalar_quantized_vectors::QuantizedMergeSource<'_>> = owned
                .iter()
                .map(|(centroid, vectors, has_deletions, live)| {
                    scalar_quantized_vectors::QuantizedMergeSource {
                        centroid: *centroid,
                        vectors,
                        has_deletions: *has_deletions,
                        live: live.as_deref(),
                    }
                })
                .collect();
            let values = flat.float_vector_values(merged_field.number)?;
            let mut docs = Vec::with_capacity(usize::try_from(values.size()).unwrap_or(0));
            for ord in 0..values.size() {
                docs.push(values.ord_to_doc(ord)?);
            }
            let merged = values.all_vectors()?;
            writer.merge_field(
                merged_field.number,
                merged_field.vector_similarity_function,
                merged_field.vector_dimension,
                &merge_sources,
                &docs,
                &merged,
            )?;
        }
        let (veq, vemq) = writer.finish();
        files.push(("veq", veq));
        files.push(("vemq", vemq));
    }

    if let Some(mut ivf) =
        crate::ivf_vectors::IvfGroupWriter::new(format, merged_segment_id, suffix)?.map(|ivf| {
            ivf_models.iter().fold(ivf, |ivf, (number, model)| {
                ivf.with_model(*number, model.clone())
            })
        })
    {
        // The `.ivv` goes to its file as it is built rather than being held
        // whole beside the vectors it is built from.
        if plan
            .iter()
            .any(|p| p.0.vector_encoding == VectorEncoding::Float32)
        {
            let name = crate::index_writer::segment_file_name(
                merged_segment_name,
                suffix,
                crate::ivf_vectors::EXTENSIONS[2],
            );
            ivf.stream_vectors_to(dir.create_output(&name)?);
            ivv_name = Some(name);
        }
        // Each source group's IVF index, opened once for every field it
        // serves.
        let mut readers: Vec<(*const crate::ivf_vectors::IvfInputs, lucene_ivf::IvfReader)> =
            Vec::new();
        for (merged_field, field_sources) in plan.iter().map(|p| (&p.0, &p.1)) {
            if merged_field.vector_encoding != VectorEncoding::Float32 {
                continue;
            }
            // The IVF reader keys fields by the source's own numbers; a
            // source that numbered the field differently lends nothing.
            let mut lent: Vec<(usize, *const crate::ivf_vectors::IvfInputs)> = Vec::new();
            for fs in field_sources {
                if fs.original_field_number != merged_field.number {
                    continue;
                }
                let group = sources[fs.source_index]
                    .vectors
                    .expect("only sources with a flat reader are planned")
                    .group_of(fs.original_field_number);
                if let Some(inputs) = group.ivf {
                    let key: *const crate::ivf_vectors::IvfInputs = inputs;
                    if !readers.iter().any(|(k, _)| *k == key) {
                        readers.push((key, inputs.open()?));
                    }
                    lent.push((fs.source_index, key));
                }
            }
            let ivf_sources: Vec<crate::ivf_vectors::IvfSource<'_>> = lent
                .iter()
                .filter_map(|&(source, key)| {
                    let reader = &readers.iter().find(|(k, _)| *k == key)?.1;
                    Some(crate::ivf_vectors::IvfSource {
                        reader,
                        doc_map: &doc_id_maps[source],
                    })
                })
                .collect();
            let values = flat.float_vector_values(merged_field.number)?;
            let mut docs = Vec::with_capacity(usize::try_from(values.size()).unwrap_or(0));
            for ord in 0..values.size() {
                docs.push(values.ord_to_doc(ord)?);
            }
            let merged = values.all_vectors()?;
            let dimension = usize::try_from(merged_field.vector_dimension).unwrap_or(0);
            let similarity = merged_field.vector_similarity_function;
            if similarity != field_infos::VectorSimilarityFunction::Cosine {
                ivf.add(
                    &crate::ivf_vectors::IvfField {
                        number: merged_field.number,
                        similarity,
                        dimension,
                        docs: &docs,
                        vectors: &merged,
                        partitions,
                    },
                    &ivf_sources,
                )?;
            } else {
                // A cosine field is written unit-normalized: the normalized
                // vectors go to a scratch file the writer reads mapped, so the
                // merge holds neither them nor the source's vectors on the heap.
                let fits_half = ivf.fits_half(&merged);
                let scratch =
                    crate::index_writer::segment_file_name(merged_segment_name, suffix, "ivn");
                write_normalized_vectors(dir, &scratch, dimension, &merged)?;
                drop(merged);
                let input = dir.open(&scratch)?;
                match lucene_util::vector_util::le_f32s(&input) {
                    Some(normalized) => ivf.add_prepared(
                        &crate::ivf_vectors::IvfField {
                            number: merged_field.number,
                            similarity,
                            dimension,
                            docs: &docs,
                            vectors: normalized,
                            partitions,
                        },
                        &ivf_sources,
                        fits_half,
                    )?,
                    None => {
                        return Err(lucene_store::Error::Io(std::io::Error::other(
                            "the normalized vectors are not aligned floats",
                        ))
                        .into())
                    }
                }
                drop(input);
                dir.delete_file(&scratch)?;
            }
        }
        match ivf.finish_streamed() {
            Some((written, out)) => {
                files.extend(written);
                match out {
                    Some(out) => {
                        out.close()?;
                    }
                    None => ivv_name = None,
                }
            }
            None => {
                if let Some(name) = ivv_name.take() {
                    dir.delete_file(&name)?;
                }
            }
        }
    }

    if format.hnsw_quantizer().is_some() {
        // The merged vectors are encoded afresh, against their own mean.
        let mut merged_fields = Vec::new();
        for merged_field in plan.iter().map(|p| &p.0) {
            if merged_field.vector_encoding != VectorEncoding::Float32 {
                continue;
            }
            let values = flat.float_vector_values(merged_field.number)?;
            merged_fields.push((merged_field, values.all_vectors()?));
        }
        let fields: Vec<crate::hnsw_quant_vectors::CodesField<'_>> = merged_fields
            .iter()
            .map(|(field, vectors)| crate::hnsw_quant_vectors::CodesField {
                number: field.number,
                similarity: field.vector_similarity_function,
                dimension: usize::try_from(field.vector_dimension).unwrap_or(0),
                vectors,
            })
            .collect();
        if let Some(codes) =
            crate::hnsw_quant_vectors::write(format, &fields, merged_segment_id, suffix)?
        {
            files.extend(codes);
        }
    }

    if let Some((m, beam_width)) = format.graph() {
        let quantized = match files.iter().find(|(e, _)| *e == "vemq") {
            Some((_, vemq)) => {
                let veq = &files
                    .iter()
                    .find(|(e, _)| *e == "veq")
                    .expect("written with the .vemq")
                    .1;
                Some(
                    scalar_quantized_vectors::ScalarQuantizedVectorsReader::open(
                        vemq,
                        veq,
                        merged_segment_id,
                        suffix,
                    )?,
                )
            }
            None => None,
        };
        let mut hnsw_fields: Vec<hnsw_vectors::HnswVectorsField<'_>> = Vec::new();
        let mut graphs: Vec<Option<hnsw::OnHeapHnswGraph>> = Vec::with_capacity(plan.len());
        let mut counts: Vec<i32> = Vec::with_capacity(plan.len());
        // Each field's graph is merged on its own (a fixed seed, no shared
        // state), so the fields run in parallel; collected in field order,
        // the first failing field is the one reported, as the serial loop did.
        let merge_field_graph = |merged_field: &&FieldInfo,
                                 field_sources: &Vec<FieldSource>|
         -> Result<(i32, Option<hnsw::OnHeapHnswGraph>)> {
            // Every source's own graph for this field, in the same order the
            // flat merge consumed them. A source with the flat pair and no
            // graph still contributes vectors -- they are simply inserted.
            let mut source_graphs: Vec<Option<hnsw_vectors::OffHeapHnswGraph<'_>>> =
                Vec::with_capacity(field_sources.len());
            for fs in field_sources {
                let source_vectors = sources[fs.source_index]
                    .vectors
                    .expect("only sources with a flat reader are planned");
                let graph = match source_vectors.group_of(fs.original_field_number).graph {
                    Some(reader) if reader.field(fs.original_field_number).is_some() => {
                        reader.graph(fs.original_field_number)?
                    }
                    _ => None,
                };
                source_graphs.push(graph);
            }
            let graph_sources: Vec<
                hnsw_vectors::GraphMergeSource<'_, hnsw_vectors::OffHeapHnswGraph<'_>>,
            > = field_sources
                .iter()
                .zip(&source_graphs)
                .map(|(fs, graph)| hnsw_vectors::GraphMergeSource {
                    graph: graph.as_ref(),
                    ord_to_doc: &fs.ord_to_doc,
                    doc_map: &doc_id_maps[fs.source_index],
                })
                .collect();

            let (count, graph) = match merged_field.vector_encoding {
                VectorEncoding::Float32 => {
                    let values = flat.float_vector_values(merged_field.number)?;
                    let merged_ord_to_doc: Vec<i32> = (0..values.size())
                        .map(|ord| values.ord_to_doc(ord))
                        .collect::<std::result::Result<Vec<i32>, _>>()?;
                    let graph = match &quantized {
                        Some(reader) if values.size() > 0 => {
                            let qv = reader.quantized_vector_values(merged_field.number)?;
                            let similarity = merged_field.vector_similarity_function;
                            let query_side = if qv.encoding().is_asymmetric() {
                                let raw = values.all_vectors()?;
                                Some(scalar_quantized_vectors::binarized_query_data(
                                    &qv, similarity, &raw,
                                )?)
                            } else {
                                None
                            };
                            let scorer = scalar_quantized_vectors::merge_scorer_supplier(
                                &qv,
                                similarity,
                                query_side.as_deref(),
                            )?;
                            hnsw_vectors::merge_one_field(
                                scorer,
                                m,
                                beam_width,
                                hnsw::DEFAULT_RAND_SEED,
                                &merged_ord_to_doc,
                                &graph_sources,
                            )?
                        }
                        _ => hnsw_vectors::merge_one_field(
                            values.ord_scorer(),
                            m,
                            beam_width,
                            hnsw::DEFAULT_RAND_SEED,
                            &merged_ord_to_doc,
                            &graph_sources,
                        )?,
                    };
                    (values.size(), graph)
                }
                VectorEncoding::Byte => {
                    let values = flat.byte_vector_values(merged_field.number)?;
                    let merged_ord_to_doc: Vec<i32> = (0..values.size())
                        .map(|ord| values.ord_to_doc(ord))
                        .collect::<std::result::Result<Vec<i32>, _>>()?;
                    let graph = hnsw_vectors::merge_one_field(
                        values.ord_scorer(),
                        m,
                        beam_width,
                        hnsw::DEFAULT_RAND_SEED,
                        &merged_ord_to_doc,
                        &graph_sources,
                    )?;
                    (values.size(), graph)
                }
            };
            Ok((count, graph))
        };
        let merged_graphs: Vec<Result<(i32, Option<hnsw::OnHeapHnswGraph>)>> = {
            use rayon::prelude::*;
            plan.par_iter()
                .map(|p| merge_field_graph(&p.0, &p.1))
                .collect()
        };
        for merged in merged_graphs {
            let (count, graph) = merged?;
            counts.push(count);
            graphs.push(graph);
        }
        for ((merged_field, _), (graph, count)) in plan
            .iter()
            .map(|p| (&p.0, &p.1))
            .zip(graphs.iter().zip(&counts))
        {
            hnsw_fields.push(hnsw_vectors::HnswVectorsField {
                field_number: merged_field.number,
                encoding: merged_field.vector_encoding,
                similarity: merged_field.vector_similarity_function,
                dimension: merged_field.vector_dimension,
                count: *count,
                graph: graph.as_ref(),
                m,
            });
        }
        let (vex_bytes, vem_bytes) =
            hnsw_vectors::write_hnsw_vectors(&hnsw_fields, merged_segment_id, suffix)?;
        files.push(("vex", vex_bytes));
        files.push(("vem", vem_bytes));
        if format.hnsw_disk_quantizer().is_some() {
            // The merged graph and vectors are copied into node records
            // afresh.
            let mut merged_fields = Vec::new();
            for (merged_field, graph) in plan.iter().map(|p| &p.0).zip(&graphs) {
                let Some(graph) = graph else { continue };
                if merged_field.vector_encoding != VectorEncoding::Float32 {
                    continue;
                }
                let values = flat.float_vector_values(merged_field.number)?;
                let docs = (0..values.size())
                    .map(|ord| values.ord_to_doc(ord))
                    .collect::<std::result::Result<Vec<i32>, _>>()?;
                merged_fields.push((merged_field, docs, values.all_vectors()?, graph));
            }
            let disk: Vec<crate::disk_hnsw_vectors::DiskField<'_>> = merged_fields
                .iter()
                .map(
                    |(field, docs, vectors, graph)| crate::disk_hnsw_vectors::DiskField {
                        number: field.number,
                        similarity: field.vector_similarity_function,
                        dimension: usize::try_from(field.vector_dimension).unwrap_or(0),
                        docs,
                        vectors,
                        graph,
                    },
                )
                .collect();
            if let Some(written) =
                crate::disk_hnsw_vectors::write_merged(format, &disk, merged_segment_id, suffix)?
            {
                files.extend(written);
            }
        }
    }
    files.insert(0, ("vemf", vemf_bytes));
    let mut early = vec![vec_name];
    early.extend(ivv_name);
    Ok((files, early))
}

impl MergeOptions {
    /// Whether the codec's `knnVectorsFormat()` is a format itself rather
    /// than per-field routing.
    pub(super) fn knn_vectors_direct(&self) -> bool {
        self.codec
            .as_ref()
            .is_some_and(|c| c.knn_vectors_format_is_direct())
    }

    /// The KNN vectors format `field` is routed to.
    pub(super) fn knn_vectors_format_for(
        &self,
        field: &str,
    ) -> per_field_knn_vectors::KnnVectorsFormat {
        self.knn_vectors_formats
            .iter()
            .find(|(name, _)| name == field)
            .map(|(_, format)| *format)
            .or_else(|| {
                self.codec
                    .as_ref()
                    .map(|c| c.knn_vectors_format_for_field(field))
            })
            .unwrap_or(per_field_knn_vectors::KnnVectorsFormat::Hnsw {
                max_conn: self.hnsw_m,
                beam_width: self.hnsw_beam_width,
            })
    }
}
