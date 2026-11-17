//! Faiss's own index serialization (`faiss/impl/index_write.cpp`,
//! `index_read.cpp`): the bytes `faiss::write_index` produces and
//! `faiss::read_index` reads, for the index types a Lucene Faiss field uses.
//!
//! Everything is little-endian. A vector is `u64 count` then the elements; an
//! index starts with a four-character tag and a header:
//!
//! ```text
//! header  --> d(i32) ntotal(i64) dummy(i64)=1<<20 dummy(i64)=1<<20
//!             is_trained(u8) metric_type(i32) [metric_arg(f32) if metric_type > 1]
//! IxF2/IxFI/IxFl  flat (L2 / inner product / other)   --> header, floats(vec<f32>)
//! IHNf            HNSW over flat storage              --> header, hnsw, storage index
//! hnsw            --> assign_probas(vec<f64>) cum_nneighbor_per_level(vec<i32>)
//!                     levels(vec<i32>) offsets(vec<u64>) neighbors(vec<i32>)
//!                     entry_point max_level efConstruction efSearch upper_beam (i32 each)
//! IwFl            IVF over flat codes                 --> header, nlist(u64) nprobe(u64),
//!                     quantizer index, direct_map, invlists
//! direct_map      --> type(u8) array(vec<i64>)
//! ilar            --> "ilar" nlist(u64) code_size(u64) "full" sizes(vec<u64>),
//!                     then per non-empty list: codes, ids(i64 each)
//! IxMp            id map                              --> header, sub index, ids(vec<i64>)
//! ```

use crate::error::{corrupt, Error, Result};

/// Faiss's `MetricType` for the metrics a Lucene field maps to
/// (`DOT_PRODUCT`/`COSINE` to inner product, `EUCLIDEAN` to L2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// `METRIC_INNER_PRODUCT` (0): larger is nearer.
    InnerProduct,
    /// `METRIC_L2` (1): squared distance, smaller is nearer.
    L2,
}

impl Metric {
    pub(crate) fn from_code(code: i32) -> Result<Metric> {
        match code {
            0 => Ok(Metric::InnerProduct),
            1 => Ok(Metric::L2),
            other => Err(Error::Unsupported(format!("metric type {other}"))),
        }
    }

    pub(crate) fn code(self) -> i32 {
        match self {
            Metric::InnerProduct => 0,
            Metric::L2 => 1,
        }
    }
}

const DUMMY: i64 = 1 << 20;

/// Vectors held whole: `IndexFlat`'s codes, an HNSW's storage, an IVF's
/// centroids.
#[derive(Debug, Clone, PartialEq)]
pub struct Flat {
    /// Dimension.
    pub d: usize,
    /// Distance.
    pub metric: Metric,
    /// `ntotal * d` floats, row-major.
    pub data: Vec<f32>,
}

impl Flat {
    /// Vectors held.
    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.d).unwrap_or(0)
    }

    /// Whether no vector is held.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Vector `i`.
    pub fn vector(&self, i: usize) -> &[f32] {
        &self.data[i * self.d..(i + 1) * self.d]
    }
}

/// `faiss::HNSW`: the graph, over storage held elsewhere.
#[derive(Debug, Clone, PartialEq)]
pub struct Hnsw {
    /// Probability of a vertex reaching each level.
    pub assign_probas: Vec<f64>,
    /// Prefix sums of the neighbour slots per level.
    pub cum_nneighbor_per_level: Vec<i32>,
    /// Level of each vertex, plus one.
    pub levels: Vec<i32>,
    /// Start of each vertex's slots in `neighbors`.
    pub offsets: Vec<u64>,
    /// Neighbour slots, `-1` for an empty one.
    pub neighbors: Vec<i32>,
    /// Vertex the search enters at, `-1` for an empty graph.
    pub entry_point: i32,
    /// Highest level of the graph.
    pub max_level: i32,
    /// Beam width when building.
    pub ef_construction: i32,
    /// Beam width when searching.
    pub ef_search: i32,
    /// Faiss's `upper_beam` (1).
    pub upper_beam: i32,
}

/// One inverted list of an IVF index.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InvList {
    /// Vector ordinals of the list's entries.
    pub ids: Vec<i64>,
    /// The entries' vectors, `ids.len() * d` floats.
    pub codes: Vec<f32>,
}

/// `faiss::IndexIVFFlat`.
#[derive(Debug, Clone, PartialEq)]
pub struct IvfFlat {
    /// Dimension.
    pub d: usize,
    /// Distance.
    pub metric: Metric,
    /// Lists probed per search.
    pub nprobe: usize,
    /// The coarse centroids, one per list.
    pub quantizer: Flat,
    /// The lists.
    pub lists: Vec<InvList>,
}

/// A Faiss index of a supported kind.
#[derive(Debug, Clone, PartialEq)]
pub enum Index {
    /// `IndexFlat`.
    Flat(Flat),
    /// `IndexHNSWFlat`: graph plus the vectors.
    Hnsw {
        /// The graph.
        hnsw: Hnsw,
        /// The vectors the graph is over.
        storage: Flat,
    },
    /// `IndexIVFFlat`.
    IvfFlat(IvfFlat),
    /// `IndexIDMap`: external ids over a sub index.
    IdMap {
        /// The wrapped index, whose ordinals `ids` translates.
        sub: Box<Index>,
        /// External id of each ordinal.
        ids: Vec<i64>,
    },
}

impl Index {
    /// Dimension.
    pub fn d(&self) -> usize {
        match self {
            Index::Flat(f) => f.d,
            Index::Hnsw { storage, .. } => storage.d,
            Index::IvfFlat(i) => i.d,
            Index::IdMap { sub, .. } => sub.d(),
        }
    }

    /// Distance.
    pub fn metric(&self) -> Metric {
        match self {
            Index::Flat(f) => f.metric,
            Index::Hnsw { storage, .. } => storage.metric,
            Index::IvfFlat(i) => i.metric,
            Index::IdMap { sub, .. } => sub.metric(),
        }
    }

    /// Vectors held.
    pub fn ntotal(&self) -> usize {
        match self {
            Index::Flat(f) => f.len(),
            Index::Hnsw { storage, .. } => storage.len(),
            Index::IvfFlat(i) => i.lists.iter().map(|l| l.ids.len()).sum(),
            Index::IdMap { ids, .. } => ids.len(),
        }
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let Some(end) = end else {
            return corrupt(format!(
                "{n} bytes wanted at {} of {}",
                self.pos,
                self.buf.len()
            ));
        };
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(
            self.take(4)?.try_into().unwrap_or_default(),
        ))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(
            self.take(8)?.try_into().unwrap_or_default(),
        ))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(
            self.take(4)?.try_into().unwrap_or_default(),
        ))
    }

    fn size(&mut self) -> Result<usize> {
        let v = u64::from_le_bytes(self.take(8)?.try_into().unwrap_or_default());
        usize::try_from(v).or_else(|_| corrupt(format!("size {v}")))
    }

    /// A count prefix that must fit what is left of the buffer at `width`
    /// bytes per element.
    fn count(&mut self, width: usize) -> Result<usize> {
        let n = self.size()?;
        match n.checked_mul(width) {
            Some(bytes) if bytes <= self.buf.len() - self.pos => Ok(n),
            _ => corrupt(format!("vector of {n} elements overruns the index")),
        }
    }

    fn vec_f64(&mut self) -> Result<Vec<f64>> {
        let n = self.count(8)?;
        Ok(self
            .take(n * 8)?
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap_or_default()))
            .collect())
    }

    fn vec_i32(&mut self) -> Result<Vec<i32>> {
        let n = self.count(4)?;
        Ok(self
            .take(n * 4)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap_or_default()))
            .collect())
    }

    fn vec_u64(&mut self) -> Result<Vec<u64>> {
        let n = self.count(8)?;
        Ok(self
            .take(n * 8)?
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap_or_default()))
            .collect())
    }

    fn vec_i64(&mut self) -> Result<Vec<i64>> {
        let n = self.count(8)?;
        Ok(self
            .take(n * 8)?
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap_or_default()))
            .collect())
    }

    fn floats(&mut self, n: usize) -> Result<Vec<f32>> {
        let bytes = n.checked_mul(4).filter(|&b| b <= self.buf.len() - self.pos);
        let Some(bytes) = bytes else {
            return corrupt(format!("{n} floats overrun the index"));
        };
        Ok(self
            .take(bytes)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap_or_default()))
            .collect())
    }
}

struct Header {
    d: usize,
    ntotal: usize,
    metric: Metric,
}

fn read_header(r: &mut Reader<'_>) -> Result<Header> {
    let d = r.i32()?;
    let ntotal = r.i64()?;
    r.i64()?;
    r.i64()?;
    let _trained = r.u8()?;
    let metric_type = r.i32()?;
    if metric_type > 1 {
        r.f32()?;
    }
    let metric = Metric::from_code(metric_type)?;
    let (Ok(d), Ok(ntotal)) = (usize::try_from(d), usize::try_from(ntotal)) else {
        return corrupt(format!("header d={d} ntotal={ntotal}"));
    };
    Ok(Header { d, ntotal, metric })
}

fn read_flat_body(r: &mut Reader<'_>, h: &Header) -> Result<Flat> {
    let n = r.count(4)?;
    if h.d == 0 || n != h.d.saturating_mul(h.ntotal) {
        return corrupt(format!(
            "flat index of {n} floats for {} vectors of {} dimensions",
            h.ntotal, h.d
        ));
    }
    Ok(Flat {
        d: h.d,
        metric: h.metric,
        data: r.floats(n)?,
    })
}

fn read_hnsw(r: &mut Reader<'_>, ntotal: usize) -> Result<Hnsw> {
    let hnsw = Hnsw {
        assign_probas: r.vec_f64()?,
        cum_nneighbor_per_level: r.vec_i32()?,
        levels: r.vec_i32()?,
        offsets: r.vec_u64()?,
        neighbors: r.vec_i32()?,
        entry_point: r.i32()?,
        max_level: r.i32()?,
        ef_construction: r.i32()?,
        ef_search: r.i32()?,
        upper_beam: r.i32()?,
    };
    let levels_ok = hnsw.levels.len() == ntotal && hnsw.offsets.len() == ntotal + 1;
    let slots = hnsw.neighbors.len() as u64;
    let offsets_ok = hnsw
        .offsets
        .windows(2)
        .all(|w| w[0] <= w[1] && w[1] <= slots);
    let cum_ok = !hnsw.cum_nneighbor_per_level.is_empty()
        && hnsw
            .cum_nneighbor_per_level
            .windows(2)
            .all(|w| w[0] <= w[1])
        && hnsw
            .levels
            .iter()
            .all(|&l| l >= 1 && (l as usize) < hnsw.cum_nneighbor_per_level.len());
    let entry_ok = if ntotal == 0 {
        hnsw.entry_point == -1
    } else {
        usize::try_from(hnsw.entry_point).is_ok_and(|e| e < ntotal)
    };
    let nbr_ok = hnsw
        .neighbors
        .iter()
        .all(|&n| n < 0 || (n as usize) < ntotal);
    if !(levels_ok && offsets_ok && cum_ok && entry_ok && nbr_ok) {
        return corrupt("inconsistent hnsw graph");
    }
    Ok(hnsw)
}

fn read_index_at(r: &mut Reader<'_>, depth: u32) -> Result<Index> {
    if depth > 4 {
        return corrupt("index nesting too deep");
    }
    let tag = r.take(4)?;
    match tag {
        b"IxF2" | b"IxFI" | b"IxFl" => {
            let h = read_header(r)?;
            Ok(Index::Flat(read_flat_body(r, &h)?))
        }
        b"IHNf" => {
            let h = read_header(r)?;
            let hnsw = read_hnsw(r, h.ntotal)?;
            match read_index_at(r, depth + 1)? {
                Index::Flat(storage) if storage.len() == h.ntotal && storage.d == h.d => {
                    Ok(Index::Hnsw { hnsw, storage })
                }
                _ => corrupt("hnsw storage is not a flat index of the graph's vectors"),
            }
        }
        b"IwFl" => {
            let h = read_header(r)?;
            let nlist = r.size()?;
            let nprobe = r.size()?;
            let Index::Flat(quantizer) = read_index_at(r, depth + 1)? else {
                return corrupt("ivf quantizer is not a flat index");
            };
            if quantizer.len() != nlist || quantizer.d != h.d {
                return corrupt("ivf quantizer does not hold nlist centroids");
            }
            let direct_map = r.u8()?;
            let direct_array = r.vec_i64()?;
            if direct_map != 0 || !direct_array.is_empty() {
                return Err(Error::Unsupported(format!(
                    "ivf direct map type {direct_map}"
                )));
            }
            let lists = read_invlists(r, nlist, h.d)?;
            let total: usize = lists.iter().map(|l| l.ids.len()).sum();
            if total != h.ntotal {
                return corrupt("ivf lists do not hold ntotal vectors");
            }
            Ok(Index::IvfFlat(IvfFlat {
                d: h.d,
                metric: h.metric,
                nprobe,
                quantizer,
                lists,
            }))
        }
        b"IxMp" => {
            let h = read_header(r)?;
            let sub = read_index_at(r, depth + 1)?;
            let ids = r.vec_i64()?;
            if ids.len() != h.ntotal || sub.ntotal() != h.ntotal || sub.d() != h.d {
                return corrupt("id map does not cover the wrapped index");
            }
            Ok(Index::IdMap {
                sub: Box::new(sub),
                ids,
            })
        }
        other => Err(Error::Unsupported(format!(
            "index type {:?}",
            String::from_utf8_lossy(other)
        ))),
    }
}

fn read_invlists(r: &mut Reader<'_>, nlist: usize, d: usize) -> Result<Vec<InvList>> {
    if r.take(4)? != b"ilar" {
        return Err(Error::Unsupported("inverted lists other than ilar".into()));
    }
    if r.size()? != nlist || r.size()? != d.saturating_mul(4) {
        return corrupt("inverted lists do not match the index");
    }
    let sizes = match r.take(4)? {
        b"full" => r.vec_u64()?,
        b"sprs" => {
            // Only the non-empty lists: their numbers, then their sizes.
            let which = r.vec_u64()?;
            let counts = r.vec_u64()?;
            if which.len() != counts.len() {
                return corrupt("sparse inverted list sizes");
            }
            let mut sizes = vec![0u64; nlist];
            for (&l, &c) in which.iter().zip(&counts) {
                let Some(slot) = usize::try_from(l).ok().and_then(|l| sizes.get_mut(l)) else {
                    return corrupt("sparse inverted list number");
                };
                *slot = c;
            }
            sizes
        }
        _ => return corrupt("inverted list sizes tag"),
    };
    if sizes.len() != nlist {
        return corrupt("inverted list sizes do not match nlist");
    }
    let mut lists = Vec::with_capacity(nlist);
    for &s in &sizes {
        let Ok(n) = usize::try_from(s) else {
            return corrupt("inverted list size");
        };
        if n == 0 {
            lists.push(InvList::default());
            continue;
        }
        let codes = r.floats(n.saturating_mul(d))?;
        let id_bytes = r.take(n.saturating_mul(8).min(r.buf.len()))?;
        if id_bytes.len() != n * 8 {
            return corrupt("inverted list ids overrun the index");
        }
        let ids = id_bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap_or_default()))
            .collect();
        lists.push(InvList { ids, codes });
    }
    Ok(lists)
}

/// Reads the index `faiss::write_index` wrote into `bytes`, which must be
/// all of it.
pub fn read_index(bytes: &[u8]) -> Result<Index> {
    let mut r = Reader { buf: bytes, pos: 0 };
    let index = read_index_at(&mut r, 0)?;
    if r.pos != bytes.len() {
        return corrupt(format!("{} bytes after the index", bytes.len() - r.pos));
    }
    Ok(index)
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_header(out: &mut Vec<u8>, d: usize, ntotal: usize, metric: Metric) {
    put_i32(out, d as i32);
    out.extend_from_slice(&(ntotal as i64).to_le_bytes());
    out.extend_from_slice(&DUMMY.to_le_bytes());
    out.extend_from_slice(&DUMMY.to_le_bytes());
    out.push(1);
    put_i32(out, metric.code());
}

fn write_flat(out: &mut Vec<u8>, f: &Flat) {
    out.extend_from_slice(match f.metric {
        Metric::L2 => b"IxF2",
        Metric::InnerProduct => b"IxFI",
    });
    write_header(out, f.d, f.len(), f.metric);
    put_u64(out, f.data.len() as u64);
    for v in &f.data {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn write_vec<T: Copy>(out: &mut Vec<u8>, v: &[T], put: impl Fn(&mut Vec<u8>, T)) {
    put_u64(out, v.len() as u64);
    for &x in v {
        put(out, x);
    }
}

/// Writes `index` the way `faiss::write_index` does; [`read_index`] and
/// Faiss itself read it back.
pub fn write_index(index: &Index) -> Vec<u8> {
    let mut out = Vec::new();
    write_index_to(&mut out, index);
    out
}

fn write_index_to(out: &mut Vec<u8>, index: &Index) {
    match index {
        Index::Flat(f) => write_flat(out, f),
        Index::Hnsw { hnsw, storage } => {
            out.extend_from_slice(b"IHNf");
            write_header(out, storage.d, storage.len(), storage.metric);
            write_vec(out, &hnsw.assign_probas, |o, v| {
                o.extend_from_slice(&v.to_le_bytes())
            });
            write_vec(out, &hnsw.cum_nneighbor_per_level, put_i32);
            write_vec(out, &hnsw.levels, put_i32);
            write_vec(out, &hnsw.offsets, put_u64);
            write_vec(out, &hnsw.neighbors, put_i32);
            for v in [
                hnsw.entry_point,
                hnsw.max_level,
                hnsw.ef_construction,
                hnsw.ef_search,
                hnsw.upper_beam,
            ] {
                put_i32(out, v);
            }
            write_flat(out, storage);
        }
        Index::IvfFlat(ivf) => {
            out.extend_from_slice(b"IwFl");
            write_header(out, ivf.d, index.ntotal(), ivf.metric);
            put_u64(out, ivf.lists.len() as u64);
            put_u64(out, ivf.nprobe as u64);
            write_flat(out, &ivf.quantizer);
            out.push(0);
            put_u64(out, 0);
            out.extend_from_slice(b"ilar");
            put_u64(out, ivf.lists.len() as u64);
            put_u64(out, (ivf.d * 4) as u64);
            out.extend_from_slice(b"full");
            let sizes: Vec<u64> = ivf.lists.iter().map(|l| l.ids.len() as u64).collect();
            write_vec(out, &sizes, put_u64);
            for l in &ivf.lists {
                if l.ids.is_empty() {
                    continue;
                }
                for v in &l.codes {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                for id in &l.ids {
                    out.extend_from_slice(&id.to_le_bytes());
                }
            }
        }
        Index::IdMap { sub, ids } => {
            out.extend_from_slice(b"IxMp");
            write_header(out, sub.d(), ids.len(), sub.metric());
            write_index_to(out, sub);
            write_vec(out, ids, |o, v| o.extend_from_slice(&v.to_le_bytes()));
        }
    }
}
