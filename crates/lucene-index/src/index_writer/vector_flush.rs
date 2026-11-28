//! The vector files a flush writes: [`IndexWriter::build_vectors_output`]
//! builds each format instance's files, streaming the large ones to the
//! directory as they are encoded.

use super::*;

impl IndexWriter<'_> {
    /// Builds this flush's `.vec`/`.vemf`/`.vem`/`.vex` bytes for every field
    /// opted in through [`IndexWriter::set_vector_field`] --
    /// `Lucene99HnswVectorsWriter.flush`, whose two halves are
    /// `Lucene99FlatVectorsWriter.flush` (the flat store) and
    /// `writeField`/`writeGraph` (the graph) on top of it.
    ///
    /// The order is Java's and is load-bearing: the flat store is written and
    /// **reopened** first, and the graph is then built over the reopened
    /// values. Java does the same (`Lucene99FlatVectorsWriter.flush` returns a
    /// `FlatCloseableRandomVectorScorerSupplier` over the bytes it just
    /// wrote), because the graph's neighbour ordinals must address the same
    /// ordinal space the `.vec` file defines. Building the graph over the
    /// in-memory `Vec<f32>` instead would be equivalent only as long as the
    /// two agree -- and an `alignOutput` or ordinal-assignment mistake is
    /// exactly the kind of thing that makes them not.
    ///
    /// A field no pending document carried a vector for is **skipped
    /// entirely**, and [`IndexingConfig::fields_with_per_field_attributes`] then zeroes
    /// its `.fnm` `vector_dimension`. Java reaches the same state from the
    /// other side: `IndexingChain` only creates a field's
    /// `KnnFieldVectorsWriter` (and only sets `FieldInfo.vectorDimension`)
    /// when a document actually carries the field, so a segment's `.fnm`
    /// never claims vectors its files do not hold.
    ///
    /// Returns `None` when no field has any vector, so no files are written.
    ///
    /// Without `indexes` (a realtime chunk of a buffer, read by searches
    /// until the buffer flushes) no graph or IVF clusters are built: every
    /// `.vem` records `numLevels = 0`, Lucene's tiny-segment case, which
    /// both readers search exhaustively with exact scores.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_vectors_output(
        dir: &dyn Directory,
        segment_name: &str,
        partitions: Option<crate::ivf_vectors::Partitions<'_>>,
        pending_vectors: &[Vec<DocumentVector>],
        configs: &[VectorFieldConfig],
        max_doc: i32,
        format_for: &dyn Fn(&str) -> per_field_knn_vectors::KnnVectorsFormat,
        direct: bool,
        indexes: bool,
        graphs: &[BufferGraphField],
        old_to_new: Option<&[usize]>,
        clusters: &crate::ivf_vectors::ClusterSink,
        ivf_models: &[(String, String)],
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<Option<VectorsOutput>> {
        // Empty when no buffered document carries a vector (a buffer of
        // explicit documents); otherwise one entry per document.
        debug_assert!(
            pending_vectors.is_empty() || pending_vectors.len() as i32 == max_doc,
            "pending_vectors must stay aligned 1:1 with pending_docs"
        );
        // `IndexingChain` calls `KnnVectorsWriter.addField` the first time a
        // document of the flush carries a field, and
        // `PerFieldKnnVectorsFormat.FieldsWriter.getInstance` numbers the
        // format instances -- and every writer lists its fields -- in that
        // order.
        let mut first_seen: Vec<&str> = Vec::new();
        for doc_vectors in pending_vectors {
            for v in doc_vectors {
                if !first_seen.contains(&v.field_name.as_str()) {
                    first_seen.push(v.field_name.as_str());
                }
            }
        }
        let mut ordered: Vec<&VectorFieldConfig> = configs.iter().collect();
        ordered.sort_by_key(|c| {
            first_seen
                .iter()
                .position(|n| *n == c.name)
                .unwrap_or(usize::MAX)
        });

        let mut flat_fields: Vec<FlatVectorsField> = Vec::new();
        let mut names: Vec<(i32, String)> = Vec::new();
        for config in ordered {
            let mut docs: Vec<i32> = Vec::new();
            let mut floats: Vec<f32> = Vec::new();
            let mut bytes: Vec<u8> = Vec::new();
            // Sized once: growing by doubling would keep up to twice the
            // field's bytes allocated for the rest of the flush.
            let (mut float_len, mut byte_len) = (0usize, 0usize);
            for doc_vectors in pending_vectors {
                if let Some(entry) = doc_vectors.iter().find(|v| v.field_name == config.name) {
                    match &entry.value {
                        VectorValue::Float32(v) => float_len = float_len.saturating_add(v.len()),
                        VectorValue::Byte(v) => byte_len = byte_len.saturating_add(v.len()),
                    }
                }
            }
            floats.reserve_exact(float_len);
            bytes.reserve_exact(byte_len);
            for (doc_id, doc_vectors) in pending_vectors.iter().enumerate() {
                let Some(entry) = doc_vectors.iter().find(|v| v.field_name == config.name) else {
                    continue;
                };
                docs.push(doc_id as i32);
                match &entry.value {
                    VectorValue::Float32(v) => floats.extend_from_slice(v),
                    VectorValue::Byte(v) => bytes.extend_from_slice(v),
                }
            }
            if docs.is_empty() {
                continue;
            }
            // Only one of the two accumulators can be non-empty:
            // `add_document_with_vectors` rejects a value whose variant
            // disagrees with the field's declared encoding, so the other is
            // always empty. Asserted rather than assumed, because picking the
            // wrong one here would silently drop every vector of the field.
            debug_assert!(
                floats.is_empty() || bytes.is_empty(),
                "field {:?} accumulated both float and byte components",
                config.name
            );
            let values = match config.encoding {
                VectorEncoding::Float32 => FieldVectorData::Float32(floats),
                VectorEncoding::Byte => FieldVectorData::Byte(bytes),
            };
            names.push((config.field_number, config.name.clone()));
            flat_fields.push(FlatVectorsField {
                field_number: config.field_number,
                similarity: config.similarity,
                dimension: config.dimension,
                docs,
                values,
            });
        }
        if flat_fields.is_empty() {
            return Ok(None);
        }
        let name_refs: Vec<(i32, &str)> = names.iter().map(|(n, s)| (*n, s.as_str())).collect();
        let groups = match name_refs.first() {
            Some(&(_, first)) if direct => {
                per_field_knn_vectors::direct_group(&name_refs, format_for(first))
            }
            _ => per_field_knn_vectors::group_fields(&name_refs, format_for),
        };

        let mut output = VectorsOutput {
            groups: Vec::with_capacity(groups.len()),
            written: Vec::with_capacity(names.len()),
        };
        // Each field is in exactly one group, so the groups take their
        // fields by move: a clone copied every vector a second time.
        let mut flat_fields: Vec<Option<FlatVectorsField>> =
            flat_fields.into_iter().map(Some).collect();
        for group in &groups {
            let suffix = group.codec_suffix();
            let mut fields: Vec<FlatVectorsField> = group
                .field_numbers
                .iter()
                .filter_map(|n| {
                    flat_fields
                        .iter_mut()
                        .find(|f| f.as_ref().is_some_and(|f| f.field_number == *n))
                        .and_then(Option::take)
                })
                .collect();
            let (files, early) = Self::build_vector_group(
                dir,
                segment_name,
                &mut fields,
                group.format,
                max_doc,
                indexes,
                graphs,
                old_to_new,
                clusters,
                &crate::ivf_vectors::models_for(
                    ivf_models,
                    names.iter().map(|(n, s)| (*n, s.as_str())),
                ),
                segment_id,
                &suffix,
                partitions,
            )?;
            for (number, name) in &names {
                if group.field_numbers.contains(number) {
                    output.written.push(WrittenVectorField {
                        name: name.clone(),
                        format: group.format.name(),
                        suffix: group.suffix,
                        direct: group.direct,
                    });
                }
            }
            output.groups.push(VectorGroupFiles {
                suffix,
                files,
                early,
            });
        }
        Ok(Some(output))
    }

    /// The graph the buffer kept over `field`'s vectors as they arrived,
    /// when it is the one this flush writes: the same vectors under the
    /// same ordinals, built with the same parameters and seed by one
    /// inserter in ordinal order, which is what
    /// `HnswGraphBuilder::new(..).build(count)` does. Under an index sort
    /// (`old_to_new`) it is renumbered to the sorted ordinals, as Lucene
    /// writes the graph its `FieldWriter` built in arrival order. A
    /// half-float format's flush graph is built over the vectors as
    /// indexed, as the kept one is.
    fn kept_graph(
        graphs: &[BufferGraphField],
        old_to_new: Option<&[usize]>,
        field: &FlatVectorsField,
        params: (i32, i32),
        #[cfg_attr(not(test), allow(unused_variables))] segment_id: &[u8; ID_LENGTH],
    ) -> Option<hnsw::OnHeapHnswGraph> {
        let kept = graphs
            .iter()
            .find(|g| g.field_number == field.field_number)?;
        let graph = &kept.graph;
        let count = i32::try_from(field.docs.len()).ok()?;
        let (encoding, similarity, dimension) = graph.shape();
        if graph.params() != params
            || graph.len() != count
            || encoding != field.values.encoding()
            || similarity != field.similarity
            || i32::try_from(dimension).ok() != Some(field.dimension)
        {
            return None;
        }
        let docs = graph.docs()?;
        let renumber = match old_to_new {
            // The ordinals name the same documents.
            None => {
                if docs != field.docs {
                    return None;
                }
                None
            }
            // `Lucene99HnswVectorsWriter.writeSortingField`: the graph was
            // built in arrival order, and each ordinal takes the place of
            // its document in the sorted segment's ordinals.
            Some(old_to_new) => {
                let mut ords = Vec::with_capacity(field.docs.len());
                for &old in &docs {
                    let old = usize::try_from(old).ok()?;
                    let new = i32::try_from(*old_to_new.get(old)?).ok()?;
                    ords.push(i32::try_from(field.docs.binary_search(&new).ok()?).ok()?);
                }
                Some(ords)
            }
        };
        let graph = graph.graph(count).ok()?;
        let graph = match renumber {
            None => graph,
            Some(ords) => graph.renumbered(&ords).ok()?,
        };
        #[cfg(test)]
        KEPT_GRAPHS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(*segment_id);
        Some(graph)
    }

    /// One `PerFieldKnnVectorsFormat` instance's flush: the raw vectors
    /// (`Lucene99FlatVectorsWriter`, `.vec`/`.vemf`), the scalar-quantized
    /// codes of its `FLOAT32` fields when the format quantizes
    /// (`Lucene104ScalarQuantizedVectorsWriter.flush`, `.veq`/`.vemq`), and
    /// the graph when it has one (`Lucene99HnswVectorsWriter`, `.vem`/`.vex`)
    /// -- built on the raw vectors, which is what Java's quantized format
    /// hands the graph at flush too.
    #[allow(clippy::too_many_arguments)]
    fn build_vector_group(
        dir: &dyn Directory,
        segment_name: &str,
        flat_fields: &mut [FlatVectorsField],
        format: per_field_knn_vectors::KnnVectorsFormat,
        max_doc: i32,
        indexes: bool,
        graphs: &[BufferGraphField],
        old_to_new: Option<&[usize]>,
        clusters: &crate::ivf_vectors::ClusterSink,
        ivf_models: &[(i32, std::sync::Arc<lucene_ivf::SharedModel>)],
        segment_id: &[u8; ID_LENGTH],
        suffix: &str,
        partitions: Option<crate::ivf_vectors::Partitions<'_>>,
    ) -> Result<GroupBuilt> {
        // `.vec` is written as it is encoded: the graph below reads it back
        // mapped, so the flush never holds the file besides the vectors.
        let mut early: Vec<&'static str> = vec!["vec"];
        let vec_name = segment_file_name(segment_name, suffix, "vec");
        let vemf_bytes = {
            let mut writer = vectors::FlatVectorsWriter::new(max_doc, segment_id, suffix);
            writer.stream_to(dir.create_output(&vec_name)?);
            for field in flat_fields.iter() {
                writer.write_field(field)?;
            }
            let (out, meta) = writer.finish_streamed();
            if let Some(out) = out {
                out.close()?;
            }
            meta
        };
        let vec_bytes: Vec<u8> = Vec::new();
        let mut files: Vec<(&'static str, Vec<u8>)> = Vec::new();

        if let Some(encoding) = format.quantization() {
            let mut writer =
                lucene_codecs::scalar_quantized_vectors::ScalarQuantizedVectorsWriter::new(
                    encoding, max_doc, segment_id, suffix,
                );
            for field in flat_fields.iter() {
                if let FieldVectorData::Float32(values) = &field.values {
                    writer.write_field(
                        &lucene_codecs::scalar_quantized_vectors::QuantizedVectorsField {
                            field_number: field.field_number,
                            similarity: field.similarity,
                            dimension: field.dimension,
                            docs: &field.docs,
                            vectors: values,
                        },
                    )?;
                }
            }
            let (veq, vemq) = writer.finish();
            files.push(("veq", veq));
            files.push(("vemq", vemq));
        }

        let ivf = if indexes {
            crate::ivf_vectors::IvfGroupWriter::new(format, segment_id, suffix)?.map(|ivf| {
                ivf_models.iter().fold(ivf, |ivf, (number, model)| {
                    ivf.with_model(*number, model.clone())
                })
            })
        } else {
            None
        };
        if let Some(mut ivf) = ivf {
            // Once `.vec` and the codes are written, nothing reads the raw
            // vectors again unless a graph or hnsw codes follow: a cosine
            // field can be normalized where it is instead of copied.
            let in_place = format.graph().is_none() && format.hnsw_quantizer().is_none();
            ivf.stream_vectors_to(dir.create_output(&segment_file_name(
                segment_name,
                suffix,
                "ivv",
            ))?);
            for field in flat_fields.iter_mut() {
                let (number, similarity) = (field.field_number, field.similarity);
                let dimension = usize::try_from(field.dimension).unwrap_or(0);
                if let FieldVectorData::Float32(values) = &mut field.values {
                    let cosine =
                        similarity == lucene_codecs::field_infos::VectorSimilarityFunction::Cosine;
                    let ivf_field = |vectors| crate::ivf_vectors::IvfField {
                        number,
                        similarity,
                        dimension,
                        docs: &field.docs,
                        vectors,
                        partitions,
                    };
                    if in_place && cosine {
                        let fits_half = ivf.fits_half(values);
                        lucene_ivf::prepare_vectors(
                            lucene_ivf::Similarity::Cosine,
                            dimension,
                            values,
                        );
                        ivf.add_prepared(&ivf_field(values), &[], fits_half)?;
                    } else {
                        ivf.add(&ivf_field(values), &[])?;
                    }
                }
            }
            clusters.record(ivf.clusters());
            if let Some((ivf_files, out)) = ivf.finish_streamed() {
                if let Some(out) = out {
                    out.close()?;
                }
                early.push("ivv");
                files.extend(ivf_files);
            }
        }

        if indexes && format.hnsw_quantizer().is_some() {
            let fields: Vec<crate::hnsw_quant_vectors::CodesField<'_>> = flat_fields
                .iter()
                .filter_map(|field| match &field.values {
                    FieldVectorData::Float32(values) => {
                        Some(crate::hnsw_quant_vectors::CodesField {
                            number: field.field_number,
                            similarity: field.similarity,
                            dimension: usize::try_from(field.dimension).unwrap_or(0),
                            vectors: values,
                        })
                    }
                    _ => None,
                })
                .collect();
            if let Some(codes) =
                crate::hnsw_quant_vectors::write(format, &fields, segment_id, suffix)?
            {
                files.extend(codes);
            }
        }

        if let Some((m, beam_width)) = format.graph() {
            // Reopen exactly the bytes just written, and build every graph
            // over *those* -- see this method's doc comment.
            //
            // One graph per field, built in parallel: each depends only on
            // its own vectors and the fixed `DEFAULT_RAND_SEED`, so the bytes
            // do not change. The first error in field order is returned.
            let vec_input = dir.open(&vec_name)?;
            // With nothing else reading the raw vectors, the graph reads
            // them from the file alone.
            if format.hnsw_quantizer().is_none()
                && format.hnsw_disk_quantizer().is_none()
                && !vectors::stores_half_floats(suffix)
            {
                for field in flat_fields.iter_mut() {
                    field.values = match field.values.encoding() {
                        VectorEncoding::Float32 => FieldVectorData::Float32(Vec::new()),
                        VectorEncoding::Byte => FieldVectorData::Byte(Vec::new()),
                    };
                }
            }
            let graphs: Vec<Option<hnsw::OnHeapHnswGraph>> = {
                use rayon::prelude::*;
                // A half-float instance's flush graph is built over the
                // vectors as indexed, as `Lucene99HnswVectorsWriter` builds
                // k-NN's from the field writer's `float[]`s: an in-memory
                // full-precision copy, ordinal for ordinal.
                let full;
                let flat = if vectors::stores_half_floats(suffix) {
                    full = vectors::write_flat_vectors(flat_fields, max_doc, segment_id, "")?;
                    vectors::FlatVectorsReader::open(&full.1, &full.0, segment_id, "")?
                } else {
                    vectors::FlatVectorsReader::open(&vemf_bytes, &vec_input, segment_id, suffix)?
                };
                let build = |field: &FlatVectorsField| -> Result<Option<hnsw::OnHeapHnswGraph>> {
                    let count = field.docs.len() as i32;
                    // `Lucene99HnswVectorsWriter.shouldCreateGraph`: a segment
                    // too small for a graph to beat an exhaustive scan gets
                    // none, and its `.vem` records `numLevels = 0`.
                    if !indexes || !hnsw::should_create_graph(hnsw::HNSW_GRAPH_THRESHOLD, count) {
                        return Ok(None);
                    }
                    // A Hamming format scores bytes its own way: never kept.
                    let kept = if vectors::scores_hamming(suffix) {
                        None
                    } else {
                        Self::kept_graph(graphs, old_to_new, field, (m, beam_width), segment_id)
                    };
                    if let Some(graph) = kept {
                        return Ok(Some(graph));
                    }
                    let graph = match field.values.encoding() {
                        VectorEncoding::Float32 => {
                            let values = flat.float_vector_values(field.field_number)?;
                            hnsw::HnswGraphBuilder::new(
                                values.ord_scorer(),
                                m,
                                beam_width,
                                hnsw::DEFAULT_RAND_SEED,
                            )?
                            .build(count)?
                        }
                        VectorEncoding::Byte => {
                            let values = flat.byte_vector_values(field.field_number)?;
                            hnsw::HnswGraphBuilder::new(
                                values.ord_scorer(),
                                m,
                                beam_width,
                                hnsw::DEFAULT_RAND_SEED,
                            )?
                            .build(count)?
                        }
                    };
                    Ok(Some(graph))
                };
                let built: Vec<Result<Option<hnsw::OnHeapHnswGraph>>> =
                    flat_fields.par_iter().map(build).collect();
                built.into_iter().collect::<Result<_>>()?
            };
            let hnsw_fields: Vec<HnswVectorsField<'_>> = flat_fields
                .iter()
                .zip(&graphs)
                .map(|(field, graph)| HnswVectorsField {
                    field_number: field.field_number,
                    encoding: field.values.encoding(),
                    similarity: field.similarity,
                    dimension: field.dimension,
                    count: field.docs.len() as i32,
                    graph: graph.as_ref(),
                    m,
                })
                .collect();
            let (vex_bytes, vem_bytes) =
                hnsw_vectors::write_hnsw_vectors(&hnsw_fields, segment_id, suffix)?;
            files.push(("vex", vex_bytes));
            files.push(("vem", vem_bytes));
            if indexes && format.hnsw_disk_quantizer().is_some() {
                let disk: Vec<crate::disk_hnsw_vectors::DiskField<'_>> = flat_fields
                    .iter()
                    .zip(&graphs)
                    .filter_map(|(field, graph)| match (&field.values, graph) {
                        (FieldVectorData::Float32(values), Some(graph)) => {
                            Some(crate::disk_hnsw_vectors::DiskField {
                                number: field.field_number,
                                similarity: field.similarity,
                                dimension: usize::try_from(field.dimension).unwrap_or(0),
                                docs: &field.docs,
                                vectors: values,
                                graph,
                            })
                        }
                        _ => None,
                    })
                    .collect();
                if let Some(written) =
                    crate::disk_hnsw_vectors::write(format, &disk, segment_id, suffix)?
                {
                    files.extend(written);
                }
            }
        }
        files.insert(0, ("vemf", vemf_bytes));
        files.insert(0, ("vec", vec_bytes));
        Ok((files, early))
    }
}
