//! Reading a segment's disk HNSW files and searching them.

use std::sync::Arc;

use lucene_store::codec_util::{self, FOOTER_LENGTH, ID_LENGTH};
use lucene_store::{DataInput, SliceInput};
use lucene_util::{simd, vector_util};
use lucene_vector_quant::{Quantizer, QuantizerRegistry, QueryScorer};

use crate::layout::NodeLayout;
use crate::{
    similarity_from_ordinal, Error, RangeSource, Result, Similarity, VectorEncoding, CODES_CODEC,
    MAX_LEVELS, META_CODEC, NODES_CODEC, VERSION_CURRENT, VERSION_START,
};

/// A search hit: a document and its exact score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredDoc {
    /// The document.
    pub doc: i32,
    /// Its exact score against the query.
    pub score: f32,
}

/// How deep a search goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchParams {
    /// Hits returned.
    pub k: usize,
    /// Candidates kept (HNSW's `ef`, DiskANN's `L`); raised to `k`.
    pub list_len: usize,
    /// Nodes fetched together per round trip (DiskANN's `W`), at least 1.
    pub beam_width: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            k: 10,
            list_len: 100,
            beam_width: 4,
        }
    }
}

/// What a search cost.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Records read (nodes expanded), each scored exactly.
    pub expanded: usize,
    /// Batches of records fetched together: round trips when nothing is
    /// cached.
    pub rounds: usize,
    /// Codes scored to rank candidates.
    pub estimated: usize,
    /// Bytes of records read.
    pub bytes_read: usize,
}

/// Hits, best first, and what finding them cost.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SearchOutput {
    /// Up to `k` hits by descending exact score, ties by ascending document.
    pub hits: Vec<ScoredDoc>,
    /// The search's work.
    pub stats: SearchStats,
}

/// Choices made when a segment is opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenOptions {
    /// Keep the per-node codes in memory. A field written without inline
    /// codes always does; one with inline codes then ranks candidates from
    /// the codes in memory rather than from the records, which is the same
    /// estimate. Default on.
    pub resident_codes: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        OpenOptions {
            resident_codes: true,
        }
    }
}

/// An upper level: its nodes, ascending, and their neighbours.
#[derive(Debug, Clone)]
struct UpperLevel {
    nodes: Vec<u32>,
    starts: Vec<usize>,
    neighbors: Vec<u32>,
}

impl UpperLevel {
    fn neighbors(&self, node: u32) -> &[u32] {
        match self.nodes.binary_search(&node) {
            // ARITH: `starts` has one more entry than `nodes`.
            #[allow(clippy::arithmetic_side_effects)]
            Ok(i) => &self.neighbors[self.starts[i]..self.starts[i + 1]],
            Err(_) => &[],
        }
    }
}

/// One field of an open segment.
#[derive(Debug)]
pub struct DiskHnswField {
    number: i32,
    similarity: Similarity,
    dimension: usize,
    encoding: VectorEncoding,
    quantizer: Box<dyn Quantizer>,
    centroid: Vec<f32>,
    prepared: Vec<f32>,
    count: usize,
    max_conn: usize,
    layout: NodeLayout,
    nodes_offset: usize,
    entry: Option<u32>,
    levels: Vec<UpperLevel>,
    upper_nodes: Vec<u32>,
    upper_codes: Vec<u8>,
    codes: Option<Vec<u8>>,
}

impl DiskHnswField {
    /// The field number.
    pub fn number(&self) -> i32 {
        self.number
    }

    /// How vectors are compared.
    pub fn similarity(&self) -> Similarity {
        self.similarity
    }

    /// Floats per vector.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Vectors (nodes) in the field.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The graph's `M`.
    pub fn max_conn(&self) -> usize {
        self.max_conn
    }

    /// How records store vectors.
    pub fn encoding(&self) -> VectorEncoding {
        self.encoding
    }

    /// The records' geometry.
    pub fn layout(&self) -> &NodeLayout {
        &self.layout
    }

    /// Levels, level 0 included.
    pub fn num_levels(&self) -> usize {
        self.levels.len().saturating_add(1)
    }

    /// The quantizer that made the codes.
    pub fn quantizer(&self) -> &dyn Quantizer {
        self.quantizer.as_ref()
    }

    /// Where the field's records start in `.dhn`.
    pub fn nodes_offset(&self) -> usize {
        self.nodes_offset
    }

    /// Whether the per-node codes are in memory.
    pub fn resident_codes(&self) -> bool {
        self.codes.is_some()
    }

    /// Bytes this field keeps in memory: codes, upper levels and the mean.
    pub fn resident_bytes(&self) -> usize {
        let levels: usize = self
            .levels
            .iter()
            .map(|l| {
                l.nodes
                    .len()
                    .saturating_add(l.neighbors.len())
                    .saturating_mul(4)
                    .saturating_add(l.starts.len().saturating_mul(8))
            })
            .sum();
        levels
            .saturating_add(self.upper_nodes.len().saturating_mul(4))
            .saturating_add(self.upper_codes.len())
            .saturating_add(self.codes.as_ref().map_or(0, Vec::len))
            .saturating_add(
                self.centroid
                    .len()
                    .saturating_add(self.prepared.len())
                    .saturating_mul(4),
            )
    }

    /// The byte range of `node`'s record in `.dhn`.
    // ARITH: the region of `count` records was checked to fit the file at
    // open, and `node < count`.
    #[allow(clippy::arithmetic_side_effects)]
    fn record_range(&self, node: u32) -> (usize, usize) {
        let at = self.layout.record_offset(node as usize).unwrap_or(0);
        (self.nodes_offset + at, self.layout.record_len())
    }

    fn resident_code<'a>(&self, codes: &'a [u8], node: u32) -> Option<&'a [u8]> {
        let len = self.quantizer.code_len();
        codes.get((node as usize).checked_mul(len)?..)?.get(..len)
    }

    /// The code of an upper-level node, from memory.
    fn upper_code(&self, node: u32) -> Option<&[u8]> {
        if let Some(codes) = &self.codes {
            return self.resident_code(codes, node);
        }
        let i = self.upper_nodes.binary_search(&node).ok()?;
        let len = self.quantizer.code_len();
        self.upper_codes.get(i.checked_mul(len)?..)?.get(..len)
    }
}

/// An open segment.
pub struct DiskHnswReader {
    fields: Vec<DiskHnswField>,
    codes: Arc<dyn RangeSource>,
    nodes: Arc<dyn RangeSource>,
}

impl std::fmt::Debug for DiskHnswReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskHnswReader")
            .field("fields", &self.fields)
            .field("codes", &self.codes.len())
            .field("nodes", &self.nodes.len())
            .finish()
    }
}

fn corrupt<T>(m: String) -> Result<T> {
    Err(Error::Corrupted(m))
}

fn read_count(input: &mut SliceInput<'_>, what: &str) -> Result<usize> {
    let v = input.read_vint()?;
    usize::try_from(v).or_else(|_| corrupt(format!("bad {what}: {v}")))
}

fn read_offset(input: &mut SliceInput<'_>, what: &str) -> Result<usize> {
    let v = input.read_vlong()?;
    usize::try_from(v).or_else(|_| corrupt(format!("bad {what}: {v}")))
}

/// `count` bytes of `input`, after checking they are there.
fn read_vec(input: &mut SliceInput<'_>, count: usize, what: &str) -> Result<Vec<u8>> {
    if count > input.remaining() {
        return corrupt(format!("{what}: {count} bytes past the end"));
    }
    let mut out = vec![0u8; count];
    input.read_bytes(&mut out)?;
    Ok(out)
}

/// Checks a range-read file's header and the shape of its footer, and
/// returns its format version and where its payload ends.
fn check_source(
    source: &dyn RangeSource,
    codec: &str,
    segment_id: &[u8; ID_LENGTH],
    suffix: &str,
) -> Result<(i32, usize)> {
    let header_len = codec_util::index_header_length(codec, suffix);
    let Some(payload_end) = source.len().checked_sub(FOOTER_LENGTH) else {
        return corrupt(format!(
            "{codec}: file of {} bytes has no footer",
            source.len()
        ));
    };
    if payload_end < header_len {
        return corrupt(format!(
            "{codec}: file of {} bytes is truncated",
            source.len()
        ));
    }
    let header = source.read(0, header_len)?;
    let header = codec_util::check_index_header(
        &mut SliceInput::new(&header),
        codec,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        suffix,
    )?;
    let footer = source.read(payload_end, FOOTER_LENGTH)?;
    codec_util::retrieve_checksum(&footer)?;
    Ok((header.version, payload_end))
}

/// Ends of the payloads a field's regions must fit in.
struct Bounds {
    codes_end: usize,
    nodes_end: usize,
}

fn read_upper_levels(
    input: &mut SliceInput<'_>,
    levels: usize,
    count: usize,
    max_conn: usize,
) -> Result<Vec<UpperLevel>> {
    let mut out = Vec::new();
    for level in 1..levels {
        let n = read_count(input, "upper level size")?;
        if n > count || n > input.remaining() {
            return corrupt(format!("level {level} holds {n} nodes"));
        }
        let mut nodes = Vec::with_capacity(n);
        let mut starts = Vec::with_capacity(n.saturating_add(1));
        let mut neighbors = Vec::new();
        starts.push(0);
        let mut prev = 0usize;
        for i in 0..n {
            let gap = read_count(input, "node gap")?;
            let node = prev
                .checked_add(gap)
                .filter(|&node| node < count && (i == 0 || gap > 0))
                .map_or_else(
                    || corrupt(format!("level {level}: node out of order or range")),
                    Ok,
                )?;
            prev = node;
            // Below `count`, which the writer kept inside `i32`.
            nodes.push(u32::try_from(node).unwrap_or(u32::MAX));
            let degree = read_count(input, "neighbour count")?;
            if degree > max_conn.saturating_mul(2) || degree > input.remaining() {
                return corrupt(format!("level {level}: {degree} neighbours"));
            }
            for _ in 0..degree {
                let nb = read_count(input, "neighbour")?;
                if nb >= count {
                    return corrupt(format!("level {level}: neighbour {nb} of {count}"));
                }
                neighbors.push(u32::try_from(nb).unwrap_or(u32::MAX));
            }
            starts.push(neighbors.len());
        }
        out.push(UpperLevel {
            nodes,
            starts,
            neighbors,
        });
    }
    Ok(out)
}

fn read_field(
    input: &mut SliceInput<'_>,
    registry: &QuantizerRegistry,
    bounds: &Bounds,
    codes: &dyn RangeSource,
    options: &OpenOptions,
) -> Result<DiskHnswField> {
    let number = input.read_vint()?;
    let similarity = similarity_from_ordinal(input.read_byte()?)
        .map_or_else(|| corrupt("unknown similarity".into()), Ok)?;
    let dimension = read_count(input, "dimension")?;
    if dimension == 0 {
        return corrupt("zero dimension".into());
    }
    let encoding = VectorEncoding::from_ordinal(input.read_byte()?)
        .map_or_else(|| corrupt("unknown vector encoding".into()), Ok)?;
    let name = input.read_string()?;
    let state_len = read_count(input, "quantizer state length")?;
    let state = read_vec(input, state_len, "quantizer state")?;
    let quantizer = registry.get(&name)?.read(dimension, similarity, &state)?;
    let centroid_bytes = read_vec(
        input,
        dimension
            .checked_mul(4)
            .map_or_else(|| corrupt("dimension overflows".into()), Ok)?,
        "mean vector",
    )?;
    let centroid: Vec<f32> = centroid_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let prepared = quantizer
        .prepare_centroid(&centroid)
        .unwrap_or_else(|| centroid.clone());
    let count = read_count(input, "node count")?;
    let max_conn = read_count(input, "M")?;
    let page_len = read_count(input, "page length")?;
    let max_neighbors = read_count(input, "record neighbours")?;
    if max_neighbors > usize::from(u16::MAX) || max_conn == 0 {
        return corrupt(format!("M {max_conn}, {max_neighbors} record neighbours"));
    }
    let inline = match input.read_byte()? {
        0 => false,
        1 => true,
        b => return corrupt(format!("inline codes flag {b}")),
    };
    let nodes_offset = read_offset(input, "nodes offset")?;
    let codes_offset = read_offset(input, "codes offset")?;
    let code_len = quantizer.code_len();
    let vector_bytes = encoding
        .vector_bytes(dimension)
        .map_or_else(|| corrupt("dimension overflows".into()), Ok)?;
    let layout = NodeLayout::new(
        page_len,
        vector_bytes,
        max_neighbors,
        if inline { code_len } else { 0 },
    )
    .map_or_else(|| corrupt(format!("page length {page_len}")), Ok)?;
    let region = layout
        .region_len(count)
        .and_then(|len| nodes_offset.checked_add(len));
    if region.is_none_or(|end| end > bounds.nodes_end) || !nodes_offset.is_multiple_of(page_len) {
        return corrupt(format!("{count} records at {nodes_offset} do not fit"));
    }
    let codes_len = count.checked_mul(code_len);
    let codes_stop = codes_len.and_then(|len| codes_offset.checked_add(len));
    if codes_stop.is_none_or(|end| end > bounds.codes_end) {
        return corrupt(format!("{count} codes at {codes_offset} do not fit"));
    }
    let entry_plus_one = read_count(input, "entry node")?;
    let entry = match entry_plus_one {
        0 if count == 0 => None,
        e if e >= 1 && e <= count => Some(u32::try_from(e.saturating_sub(1)).unwrap_or(u32::MAX)),
        e => return corrupt(format!("entry node {e} of {count}")),
    };
    let levels = read_count(input, "level count")?;
    if levels == 0 || levels > MAX_LEVELS {
        return corrupt(format!("{levels} levels"));
    }
    let upper = read_upper_levels(input, levels, count, max_conn)?;
    let upper_count = read_count(input, "upper node count")?;
    if upper_count > count {
        return corrupt(format!("{upper_count} upper nodes of {count}"));
    }
    let mut upper_nodes: Vec<u32> = upper.iter().flat_map(|l| l.nodes.iter().copied()).collect();
    upper_nodes.sort_unstable();
    upper_nodes.dedup();
    if upper_nodes.len() != upper_count
        || upper
            .iter()
            .flat_map(|l| l.neighbors.iter())
            .any(|nb| upper_nodes.binary_search(nb).is_err())
        || entry.is_some_and(|e| levels > 1 && upper_nodes.binary_search(&e).is_err())
    {
        return corrupt("upper levels do not match their nodes".into());
    }
    let upper_codes = read_vec(
        input,
        upper_count
            .checked_mul(code_len)
            .map_or_else(|| corrupt("upper codes overflow".into()), Ok)?,
        "upper codes",
    )?;
    let codes = if !inline || options.resident_codes {
        // `codes_len` was checked to fit the file just above.
        let len = codes_len.unwrap_or(0);
        Some(codes.read(codes_offset, len)?.into_owned())
    } else {
        None
    };
    Ok(DiskHnswField {
        number,
        similarity,
        dimension,
        encoding,
        quantizer,
        centroid,
        prepared,
        count,
        max_conn,
        layout,
        nodes_offset,
        entry,
        levels: upper,
        upper_nodes,
        upper_codes,
        codes,
    })
}

/// A visited set over a field's nodes.
struct Visited(Vec<u64>);

impl Visited {
    fn new(count: usize) -> Self {
        Visited(vec![0; count.div_ceil(64)])
    }

    /// Marks `node`, returning whether it was already marked.
    fn test_and_set(&mut self, node: u32) -> bool {
        let (word, bit) = ((node / 64) as usize, node % 64);
        let mask = 1u64 << bit;
        let was = self.0[word] & mask != 0;
        self.0[word] |= mask;
        was
    }
}

/// A candidate on the search list.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    score: f32,
    node: u32,
    expanded: bool,
}

/// Inserts `c` into `list`, kept by descending score and at most `cap`
/// long.
fn insert(list: &mut Vec<Candidate>, c: Candidate, cap: usize) {
    if list.len() >= cap && list.last().is_some_and(|l| l.score >= c.score) {
        return;
    }
    let at = list.partition_point(|x| x.score >= c.score);
    list.insert(at, c);
    list.truncate(cap);
}

impl DiskHnswReader {
    /// Opens a segment: `meta` is the whole `.dhm` file, `codes` and
    /// `nodes` the `.dhc` and `.dhn` files. Reads each field's codes (unless
    /// left out, [`OpenOptions::resident_codes`]) and nothing of `nodes`
    /// but its header and footer.
    ///
    /// # Errors
    ///
    /// A header, footer or checksum mismatch; a region that does not fit
    /// its file; a quantizer the registry does not know.
    pub fn open(
        meta: &[u8],
        codes: Arc<dyn RangeSource>,
        nodes: Arc<dyn RangeSource>,
        segment_id: &[u8; ID_LENGTH],
        suffix: &str,
        registry: &QuantizerRegistry,
        options: OpenOptions,
    ) -> Result<Self> {
        let (codes_version, codes_end) =
            check_source(codes.as_ref(), CODES_CODEC, segment_id, suffix)?;
        let (nodes_version, nodes_end) =
            check_source(nodes.as_ref(), NODES_CODEC, segment_id, suffix)?;
        codec_util::check_whole_file_footer(meta, meta.len().saturating_sub(FOOTER_LENGTH))?;
        let mut input = SliceInput::new(meta);
        let version = codec_util::check_index_header(
            &mut input,
            META_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            suffix,
        )?
        .version;
        if codes_version != version || nodes_version != version {
            return corrupt(format!(
                "format versions differ: meta {version}, codes {codes_version}, nodes \
                 {nodes_version}"
            ));
        }
        let bounds = Bounds {
            codes_end,
            nodes_end,
        };
        let field_count = read_count(&mut input, "field count")?;
        let mut fields: Vec<DiskHnswField> = Vec::new();
        for _ in 0..field_count {
            let field = read_field(&mut input, registry, &bounds, codes.as_ref(), &options)?;
            if fields.iter().any(|f| f.number == field.number) {
                return corrupt(format!("field {} recorded twice", field.number));
            }
            fields.push(field);
        }
        if input.remaining() != FOOTER_LENGTH {
            return corrupt(format!(
                "{} bytes after the fields",
                input.remaining().saturating_sub(FOOTER_LENGTH)
            ));
        }
        Ok(DiskHnswReader {
            fields,
            codes,
            nodes,
        })
    }

    /// Verifies the CRC footers of the codes and nodes files, reading both
    /// whole.
    ///
    /// # Errors
    ///
    /// A checksum mismatch or a read failure.
    pub fn check_integrity(&self) -> Result<()> {
        for source in [&self.codes, &self.nodes] {
            let all = source.read(0, source.len())?;
            let end = all.len().saturating_sub(FOOTER_LENGTH);
            codec_util::check_whole_file_footer(&all, end)?;
        }
        Ok(())
    }

    /// The fields, in the order written.
    pub fn fields(&self) -> &[DiskHnswField] {
        &self.fields
    }

    /// The field numbered `number`.
    pub fn field(&self, number: i32) -> Option<&DiskHnswField> {
        self.fields.iter().find(|f| f.number == number)
    }

    /// Bytes every field keeps in memory.
    pub fn resident_bytes(&self) -> usize {
        self.fields
            .iter()
            .map(DiskHnswField::resident_bytes)
            .fold(0, usize::saturating_add)
    }

    /// The `params.k` documents of field `field` whose vectors score best
    /// against `query`, among those `accept` takes (all without a filter).
    /// Rejected documents are still traversed, as Lucene's filtered graph
    /// search does.
    ///
    /// # Errors
    ///
    /// An unknown field, a query of the wrong dimension, a record that does
    /// not decode, or a read failure.
    pub fn search(
        &self,
        field: i32,
        query: &[f32],
        params: &SearchParams,
        accept: Option<&dyn Fn(i32) -> bool>,
    ) -> Result<SearchOutput> {
        let f = self
            .field(field)
            .ok_or_else(|| Error::IllegalArgument(format!("no disk HNSW field {field}")))?;
        if query.len() != f.dimension {
            return Err(Error::IllegalArgument(format!(
                "query of dimension {} for a field of dimension {}",
                query.len(),
                f.dimension
            )));
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Err(Error::IllegalArgument("query must be finite".into()));
        }
        let mut stats = SearchStats::default();
        let Some(entry) = f.entry else {
            return Ok(SearchOutput {
                hits: Vec::new(),
                stats,
            });
        };
        let mut scorer = f.quantizer.query_scorer(query);
        scorer.set_prepared_centroid(&f.prepared);
        let scorer: &dyn QueryScorer = scorer.as_ref();
        let estimate_upper = |node: u32, stats: &mut SearchStats| -> Result<f32> {
            let code = f
                .upper_code(node)
                .map_or_else(|| corrupt(format!("no code for node {node}")), Ok)?;
            stats.estimated = stats.estimated.saturating_add(1);
            Ok(scorer.score(code))
        };

        // Greedy descent through the upper levels.
        let mut best = entry;
        let mut best_score = if f.levels.is_empty() && f.codes.is_none() {
            f32::INFINITY
        } else {
            estimate_upper(entry, &mut stats)?
        };
        for level in f.levels.iter().rev() {
            loop {
                let mut moved = false;
                for &nb in level.neighbors(best) {
                    let s = estimate_upper(nb, &mut stats)?;
                    if s > best_score {
                        best_score = s;
                        best = nb;
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
        }

        // Beam search on level 0.
        // A rejected node does not use up a place on the list, so a
        // selective filter searches as deep among the accepted nodes as an
        // unfiltered search does among all of them.
        let mut cap = params.list_len.max(params.k).max(1);
        let beam = params.beam_width.max(1);
        let mut visited = Visited::new(f.count);
        visited.test_and_set(best);
        let mut list = Vec::with_capacity(cap.saturating_add(1));
        list.push(Candidate {
            score: best_score,
            node: best,
            expanded: false,
        });
        let mut hits: Vec<ScoredDoc> = Vec::new();
        let mut batch: Vec<u32> = Vec::with_capacity(beam);
        let mut exact: Vec<(u32, f32)> = Vec::with_capacity(beam);
        let mut fresh: Vec<(usize, u32)> = Vec::with_capacity(f.layout.max_neighbors());
        let mut scratch = vec![0f32; f.dimension];
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(beam);
        let layout = f.layout;
        loop {
            batch.clear();
            for c in list.iter_mut().filter(|c| !c.expanded).take(beam) {
                c.expanded = true;
                batch.push(c.node);
            }
            if batch.is_empty() {
                break;
            }
            exact.clear();
            stats.rounds = stats.rounds.saturating_add(1);
            // The whole beam's records in one batch where the file is read
            // by page; into the CPU cache where it is in memory.
            ranges.clear();
            for &node in &batch {
                let (at, len) = f.record_range(node);
                self.nodes.prefetch(at, len);
                ranges.push((at, len));
            }
            self.nodes.prefetch_ranges(&ranges);
            for &node in &batch {
                let (at, len) = f.record_range(node);
                let record = self.nodes.read(at, len)?;
                stats.expanded = stats.expanded.saturating_add(1);
                stats.bytes_read = stats.bytes_read.saturating_add(len);
                let doc = i32::from_le_bytes([record[0], record[1], record[2], record[3]]);
                let degree = usize::from(u16::from_le_bytes([record[4], record[5]]));
                if degree > layout.max_neighbors() {
                    return corrupt(format!("node {node} has {degree} neighbours"));
                }
                let vector = &record[layout.vector_start()..layout.neighbors_start()];
                let score = match (f.encoding, vector_util::le_f32s(vector)) {
                    (VectorEncoding::F32, Some(v)) => f.similarity.score(query, v),
                    _ => {
                        f.encoding.decode(vector, &f.centroid, &mut scratch);
                        f.similarity.score(query, &scratch)
                    }
                };
                if accept.is_none_or(|a| a(doc)) {
                    hits.push(ScoredDoc { doc, score });
                } else {
                    cap = cap.saturating_add(1).min(f.count.max(1));
                }
                exact.push((node, score));
                let neighbors = &record[layout.neighbors_start()..layout.codes_start()];
                fresh.clear();
                for (i, nb) in neighbors.chunks_exact(4).take(degree).enumerate() {
                    let nb = u32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]);
                    if nb as usize >= f.count {
                        return corrupt(format!("node {node}: neighbour {nb} of {}", f.count));
                    }
                    if !visited.test_and_set(nb) {
                        fresh.push((i, nb));
                    }
                }
                if let Some(codes) = &f.codes {
                    // Resident codes are scattered; start loading them all
                    // before the first is scored.
                    let len = f.quantizer.code_len();
                    for &(_, nb) in &fresh {
                        if let Some(at) = (nb as usize).checked_mul(len) {
                            simd::prefetch(codes, at);
                        }
                    }
                }
                for &(i, nb) in &fresh {
                    let code = match &f.codes {
                        Some(codes) => f.resident_code(codes, nb),
                        None => {
                            let len = layout.inline_code_len();
                            i.checked_mul(len)
                                .and_then(|o| o.checked_add(layout.codes_start()))
                                .and_then(|o| record.get(o..)?.get(..len))
                        }
                    }
                    .map_or_else(|| corrupt(format!("no code for node {nb}")), Ok)?;
                    stats.estimated = stats.estimated.saturating_add(1);
                    insert(
                        &mut list,
                        Candidate {
                            score: scorer.score(code),
                            node: nb,
                            expanded: false,
                        },
                        cap,
                    );
                }
            }
            // Expanded nodes rank by their exact scores from now on.
            for &(node, score) in &exact {
                if let Some(at) = list.iter().position(|c| c.node == node) {
                    let mut c = list.remove(at);
                    c.score = score;
                    let to = list.partition_point(|x| x.score >= score);
                    list.insert(to, c);
                }
            }
        }
        hits.sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));
        hits.truncate(params.k);
        Ok(SearchOutput { hits, stats })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_keeps_the_best() {
        let mut l = Vec::new();
        for (i, s) in [0.5f32, 0.9, 0.1, 0.7, 0.3].iter().enumerate() {
            insert(
                &mut l,
                Candidate {
                    score: *s,
                    node: i as u32,
                    expanded: false,
                },
                3,
            );
        }
        let nodes: Vec<u32> = l.iter().map(|c| c.node).collect();
        assert_eq!(nodes, vec![1, 3, 0]);
    }

    #[test]
    fn visited_marks_once() {
        let mut v = Visited::new(130);
        assert!(!v.test_and_set(129));
        assert!(v.test_and_set(129));
        assert!(!v.test_and_set(0));
    }
}
