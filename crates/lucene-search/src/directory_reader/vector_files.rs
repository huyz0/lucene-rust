use super::*;

impl SegmentReader {
    /// The bytes of this segment's vector files that a search reads at
    /// random (`.vec`, `.vex`, scalar-quantized and HNSW codes, IVF
    /// vectors, disk-HNSW nodes) and that are small next to the memory this
    /// process may use ([`lucene_store::read_advice::fits_warm`]): what the
    /// open reads in whole when [`lucene_store::read_advice::warm_random`] is
    /// on. A caller that turned that off hands these ranges, of every
    /// segment, to [`lucene_store::prefetch`] when its first vector search
    /// starts, so a store read by range fetches them in one batch and a
    /// search that never reads a vector fetches none. Empty for a segment
    /// with no vectors.
    pub fn vector_warm_ranges(&self) -> Vec<&[u8]> {
        fn add<'a>(ranges: &mut Vec<&'a [u8]>, input: &'a Input) {
            let bytes: &[u8] = input;
            if !bytes.is_empty() && lucene_store::read_advice::fits_warm(bytes.len() as u64) {
                ranges.push(bytes);
            }
        }
        let mut ranges: Vec<&[u8]> = Vec::new();
        for files in &self.vectors {
            add(&mut ranges, &files.data);
            if let Some(index) = &files.index {
                add(&mut ranges, index);
            }
            if let Some(scalar) = &files.scalar_quantized {
                add(&mut ranges, &scalar.data);
            }
            if let Some(codes) = &files.hnsw_codes {
                add(&mut ranges, &codes.data);
            }
            if let Some(ivf) = &files.ivf {
                add(&mut ranges, &ivf.vectors);
            }
            if let Some(disk) = &files.disk_hnsw {
                add(&mut ranges, &disk.nodes);
            }
        }
        ranges
    }

    /// One instance's flat vectors, each field's similarity taken from this
    /// segment's `FieldInfos` where its `.vemf` records none.
    pub(super) fn open_flat_vectors<'s>(
        &'s self,
        f: &'s CodecFiles,
    ) -> crate::Result<lucene_codecs::vectors::FlatVectorsReader<'s>> {
        if let Some(flat) = f.flat_reader.get() {
            return Ok(flat.attach(&f.data));
        }
        // A failed open is not kept: the next read tries again.
        let mut flat = lucene_codecs::vectors::FlatVectorsReader::open(
            &f.meta,
            &f.data,
            &self.segment_id,
            &f.suffix,
        )?;
        flat.resolve_similarities_from(&self.field_infos.fields)?;
        Ok(f.flat_reader
            .get_or_init(|| flat.detached())
            .attach(&f.data))
    }

    /// `f`'s HNSW graph reader, opened once; `None` without a graph.
    pub(super) fn graph_reader<'s>(
        &self,
        f: &'s CodecFiles,
    ) -> crate::Result<Option<&'s lucene_codecs::hnsw_vectors::HnswVectorsReader<'static>>> {
        if let Some(graph) = f.graph_reader.get() {
            return Ok(graph.as_ref());
        }
        let graph = match (&f.graph_meta, &f.index) {
            (Some(meta), Some(index)) => Some(
                lucene_codecs::hnsw_vectors::HnswVectorsReader::open(
                    meta,
                    index,
                    &self.segment_id,
                    &f.suffix,
                )?
                .detached(),
            ),
            _ => None,
        };
        Ok(f.graph_reader.get_or_init(|| graph).as_ref())
    }
}
