//! Java CrankyDocValuesFormat and its inherited DocValuesConsumer merge.
//! Generic callbacks preserve errors, identity and random draws. The native
//! adapter uses complete Lucene90 buffers and owned columns: it decodes eagerly,
//! snapshots each requested getter once and publishes files on close. Java's
//! iterator specializations, merge-reader cache, Directory/IOContext and routing
//! are outside this seam. Rc/RefCell replaces Java's shared thread-safe Random.
use lucene_codecs::doc_values::{
    self as dv, BinaryColumn, SortedNumericColumn, SortedOrdsColumn, SortedSetOrdsColumn,
};
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_store::codec_util;
use lucene_util::java_random::JavaRandom;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    Read(#[from] dv::Error),
    #[error(transparent)]
    Write(#[from] dv::WriteError),
    #[error("{0}")]
    Message(String),
}
pub type Result<T> = std::result::Result<T, Error>;
fn failure(message: impl Into<String>) -> Error {
    Error::Message(message.into())
}
fn guard(random: &RefCell<JavaRandom>, operation: &str) -> Result<()> {
    if random.borrow_mut().next_int_bounded(100) == 0 {
        Err(failure(format!("Fake IOException from {operation}()")))
    } else {
        Ok(())
    }
}
pub type NumericColumn = Vec<(i32, i64)>;
/// All five getter results keep sparse document IDs and dictionary ordinals.
#[derive(Debug, Clone)]
pub enum Column {
    Numeric(NumericColumn),
    Binary(BinaryColumn),
    Sorted(SortedOrdsColumn),
    SortedNumeric(SortedNumericColumn),
    SortedSet(SortedSetOrdsColumn),
}
impl Column {
    pub fn kind(&self) -> DocValuesType {
        match self {
            Self::Numeric(_) => DocValuesType::Numeric,
            Self::Binary(_) => DocValuesType::Binary,
            Self::Sorted(_) => DocValuesType::Sorted,
            Self::SortedNumeric(_) => DocValuesType::SortedNumeric,
            Self::SortedSet(_) => DocValuesType::SortedSet,
        }
    }
}
pub trait DocValuesProducer {
    fn get_numeric(&self, _field: &FieldInfo) -> Result<Cow<'_, NumericColumn>> {
        Err(failure("unsupported numeric getter"))
    }
    fn get_binary(&self, _field: &FieldInfo) -> Result<Cow<'_, BinaryColumn>> {
        Err(failure("unsupported binary getter"))
    }
    fn get_sorted(&self, _field: &FieldInfo) -> Result<Cow<'_, SortedOrdsColumn>> {
        Err(failure("unsupported sorted getter"))
    }
    fn get_sorted_numeric(&self, _field: &FieldInfo) -> Result<Cow<'_, SortedNumericColumn>> {
        Err(failure("unsupported sorted numeric getter"))
    }
    fn get_sorted_set(&self, _field: &FieldInfo) -> Result<Cow<'_, SortedSetOrdsColumn>> {
        Err(failure("unsupported sorted set getter"))
    }
    fn get_skipper(&self, _field: &FieldInfo) -> Result<Option<&dv::DocValuesSkipIndex>> {
        Ok(None)
    }
    fn check_integrity(&self) -> Result<()>;
    fn close(&mut self) -> Result<()>;
}
pub struct MergeSource<'a> {
    pub producer: Option<&'a dyn DocValuesProducer>,
    pub field_infos: &'a FieldInfos,
    pub max_doc: i32,
    pub doc_map: &'a [i32],
    /// Separate from DocMap: only this mask prunes pre-merge dictionaries.
    pub live_docs: Option<&'a [bool]>,
}
pub struct MergeState<'a> {
    pub sources: &'a [MergeSource<'a>],
    pub field_infos: &'a FieldInfos,
    pub check_aborted: &'a dyn Fn() -> Result<()>,
}
pub trait DocValuesConsumer {
    fn add_numeric_field(
        &mut self,
        field: &FieldInfo,
        values: &dyn DocValuesProducer,
    ) -> Result<()>;
    fn add_binary_field(&mut self, field: &FieldInfo, values: &dyn DocValuesProducer)
        -> Result<()>;
    fn add_sorted_field(&mut self, field: &FieldInfo, values: &dyn DocValuesProducer)
        -> Result<()>;
    fn add_sorted_numeric_field(
        &mut self,
        field: &FieldInfo,
        values: &dyn DocValuesProducer,
    ) -> Result<()>;
    fn add_sorted_set_field(
        &mut self,
        field: &FieldInfo,
        values: &dyn DocValuesProducer,
    ) -> Result<()>;
    fn close(&mut self) -> Result<()>;
    /// Java: DocValuesConsumer.merge, deliberately inherited by cranky consumers.
    fn merge(&mut self, state: &MergeState<'_>) -> Result<()> {
        for source in state.sources {
            if let Some(producer) = source.producer {
                (state.check_aborted)()?;
                producer.check_integrity()?;
            }
        }
        let mut fields: Vec<_> = state.field_infos.fields.iter().collect();
        fields.sort_by_key(|field| field.number);
        for field in fields {
            match field.doc_values_type {
                DocValuesType::None => {}
                DocValuesType::Numeric => self.merge_numeric_field(field, state)?,
                DocValuesType::Binary => self.merge_binary_field(field, state)?,
                DocValuesType::Sorted => self.merge_sorted_field(field, state)?,
                DocValuesType::SortedNumeric => self.merge_sorted_numeric_field(field, state)?,
                DocValuesType::SortedSet => self.merge_sorted_set_field(field, state)?,
            }
        }
        Ok(())
    }
    /// Java: DocValuesConsumer.mergeNumericField, lazy until the delegate getter.
    fn merge_numeric_field(&mut self, field: &FieldInfo, state: &MergeState<'_>) -> Result<()> {
        self.add_numeric_field(field, &MappedProducer::new(field, state))
    }
    /// Java: DocValuesConsumer.mergeBinaryField.
    fn merge_binary_field(&mut self, field: &FieldInfo, state: &MergeState<'_>) -> Result<()> {
        self.add_binary_field(field, &MappedProducer::new(field, state))
    }
    /// Java: DocValuesConsumer.mergeSortedNumericField.
    fn merge_sorted_numeric_field(
        &mut self,
        field: &FieldInfo,
        state: &MergeState<'_>,
    ) -> Result<()> {
        self.add_sorted_numeric_field(field, &MappedProducer::new(field, state))
    }
    /// Java builds the OrdinalMap before invoking guarded addSortedField.
    fn merge_sorted_field(&mut self, field: &FieldInfo, state: &MergeState<'_>) -> Result<()> {
        let mut producer = MappedProducer::new(field, state);
        producer.ordinals = Some(ordinal_map(field, state)?);
        self.add_sorted_field(field, &producer)
    }
    fn merge_sorted_set_field(&mut self, field: &FieldInfo, state: &MergeState<'_>) -> Result<()> {
        let mut producer = MappedProducer::new(field, state);
        producer.ordinals = Some(ordinal_map(field, state)?);
        self.add_sorted_set_field(field, &producer)
    }
}
fn read_column(producer: &dyn DocValuesProducer, field: &FieldInfo) -> Result<Column> {
    Ok(match field.doc_values_type {
        DocValuesType::Numeric => Column::Numeric(producer.get_numeric(field)?.into_owned()),
        DocValuesType::Binary => Column::Binary(producer.get_binary(field)?.into_owned()),
        DocValuesType::Sorted => Column::Sorted(producer.get_sorted(field)?.into_owned()),
        DocValuesType::SortedNumeric => {
            Column::SortedNumeric(producer.get_sorted_numeric(field)?.into_owned())
        }
        DocValuesType::SortedSet => Column::SortedSet(producer.get_sorted_set(field)?.into_owned()),
        DocValuesType::None => return Err(failure("field has no doc values")),
    })
}
fn source_field<'a>(source: &'a MergeSource<'_>, field: &FieldInfo) -> Option<&'a FieldInfo> {
    source
        .field_infos
        .field_by_name(&field.name)
        .filter(|source| source.doc_values_type == field.doc_values_type)
}
struct OrdinalMap {
    dictionary: Vec<Vec<u8>>,
    maps: Vec<Vec<Option<i64>>>,
}
fn ordinal_map(field: &FieldInfo, state: &MergeState<'_>) -> Result<OrdinalMap> {
    let mut dictionaries = Vec::new();
    let mut all = BTreeSet::new();
    for source in state.sources {
        let mut selected = BTreeMap::new();
        if let (Some(producer), Some(info)) = (source.producer, source_field(source, field)) {
            let column = read_column(producer, info)?;
            validate_column(&column, source.max_doc)?;
            let (dictionary, docs, counts, ords) = match column {
                Column::Sorted(c) => {
                    let counts = vec![1; c.docs.len()];
                    (c.dict, c.docs, counts, c.ords)
                }
                Column::SortedSet(c) => (c.dict, c.docs, c.counts, c.ords),
                _ => return Err(failure("invalid ordinal column")),
            };
            if let Some(live) = source.live_docs {
                let mut offset = 0usize;
                for (&doc, &count) in docs.iter().zip(&counts) {
                    let count =
                        usize::try_from(count).map_err(|_| failure("invalid ordinal count"))?;
                    let end = offset
                        .checked_add(count)
                        .ok_or_else(|| failure("ordinal count overflow"))?;
                    let values = ords
                        .get(offset..end)
                        .ok_or_else(|| failure("short ordinal column"))?;
                    let index =
                        usize::try_from(doc).map_err(|_| failure("negative source document"))?;
                    if *live
                        .get(index)
                        .ok_or_else(|| failure("short live documents"))?
                    {
                        for &ord in values {
                            let index =
                                usize::try_from(ord).map_err(|_| failure("negative ordinal"))?;
                            selected.insert(
                                index,
                                dictionary
                                    .get(index)
                                    .ok_or_else(|| failure("ordinal out of bounds"))?
                                    .clone(),
                            );
                        }
                    }
                    offset = end;
                }
            } else {
                selected.extend(dictionary.into_iter().enumerate());
            }
        }
        all.extend(selected.values().cloned());
        dictionaries.push(selected);
    }
    let dictionary: Vec<_> = all.into_iter().collect();
    let mut maps = Vec::new();
    for selected in dictionaries {
        let len = selected.last_key_value().map_or(Ok(0), |(&ord, _)| {
            ord.checked_add(1)
                .ok_or_else(|| failure("ordinal map overflow"))
        })?;
        let mut map = vec![None; len];
        for (ord, term) in selected {
            map[ord] = Some(
                i64::try_from(
                    dictionary
                        .binary_search(&term)
                        .map_err(|_| failure("missing global term"))?,
                )
                .map_err(|_| failure("too many global terms"))?,
            );
        }
        maps.push(map);
    }
    Ok(OrdinalMap { dictionary, maps })
}
struct MappedProducer<'a, 'b> {
    field: &'a FieldInfo,
    state: &'a MergeState<'b>,
    ordinals: Option<OrdinalMap>,
}
impl<'a, 'b> MappedProducer<'a, 'b> {
    fn new(field: &'a FieldInfo, state: &'a MergeState<'b>) -> Self {
        Self {
            field,
            state,
            ordinals: None,
        }
    }
    fn require_kind(&self, kind: DocValuesType) -> Result<()> {
        if self.field.doc_values_type == kind {
            Ok(())
        } else {
            Err(failure("unsupported mapped getter"))
        }
    }
    fn materialize(&self, field: &FieldInfo) -> Result<Column> {
        if !std::ptr::eq(field, self.field) {
            return Err(failure("wrong merged field identity"));
        }
        let mut rows = Vec::new();
        for (source_index, source) in self.state.sources.iter().enumerate() {
            let (Some(producer), Some(info)) = (source.producer, source_field(source, field))
            else {
                continue;
            };
            let column = read_column(producer, info)?;
            for (doc, mut values) in column_rows(&column)? {
                if doc < 0 || doc >= source.max_doc {
                    return Err(failure("source document out of bounds"));
                }
                let index =
                    usize::try_from(doc).map_err(|_| failure("negative source document"))?;
                let mapped = *source
                    .doc_map
                    .get(index)
                    .ok_or_else(|| failure("short document map"))?;
                // SENTINEL: Java MergeState.DocMap returns -1 for deleted documents.
                if mapped == -1 {
                    continue;
                }
                if mapped < 0 {
                    return Err(failure("invalid mapped document"));
                }
                if let Some(map) = &self.ordinals {
                    for value in &mut values {
                        if let Value::Long(ord) = value {
                            let index =
                                usize::try_from(*ord).map_err(|_| failure("negative ordinal"))?;
                            *ord = map.maps[source_index]
                                .get(index)
                                .copied()
                                .flatten()
                                .ok_or_else(|| failure("unmapped live ordinal"))?;
                        }
                    }
                }
                rows.push((mapped, values));
            }
        }
        rows.sort_by_key(|row| row.0);
        build_column(
            field.doc_values_type,
            rows,
            self.ordinals
                .as_ref()
                .map(|map| map.dictionary.clone())
                .unwrap_or_default(),
        )
    }
}
impl DocValuesProducer for MappedProducer<'_, '_> {
    fn get_numeric(&self, f: &FieldInfo) -> Result<Cow<'_, NumericColumn>> {
        self.require_kind(DocValuesType::Numeric)?;
        match self.materialize(f)? {
            Column::Numeric(c) => Ok(Cow::Owned(c)),
            _ => Err(failure("unsupported numeric getter")),
        }
    }
    fn get_binary(&self, f: &FieldInfo) -> Result<Cow<'_, BinaryColumn>> {
        self.require_kind(DocValuesType::Binary)?;
        match self.materialize(f)? {
            Column::Binary(c) => Ok(Cow::Owned(c)),
            _ => Err(failure("unsupported binary getter")),
        }
    }
    fn get_sorted(&self, f: &FieldInfo) -> Result<Cow<'_, SortedOrdsColumn>> {
        self.require_kind(DocValuesType::Sorted)?;
        match self.materialize(f)? {
            Column::Sorted(c) => Ok(Cow::Owned(c)),
            _ => Err(failure("unsupported sorted getter")),
        }
    }
    fn get_sorted_numeric(&self, f: &FieldInfo) -> Result<Cow<'_, SortedNumericColumn>> {
        self.require_kind(DocValuesType::SortedNumeric)?;
        match self.materialize(f)? {
            Column::SortedNumeric(c) => Ok(Cow::Owned(c)),
            _ => Err(failure("unsupported sorted numeric getter")),
        }
    }
    fn get_sorted_set(&self, f: &FieldInfo) -> Result<Cow<'_, SortedSetOrdsColumn>> {
        self.require_kind(DocValuesType::SortedSet)?;
        match self.materialize(f)? {
            Column::SortedSet(c) => Ok(Cow::Owned(c)),
            _ => Err(failure("unsupported sorted set getter")),
        }
    }
    fn check_integrity(&self) -> Result<()> {
        Err(failure("unsupported mapped integrity"))
    }
    fn close(&mut self) -> Result<()> {
        Err(failure("unsupported mapped close"))
    }
}
#[derive(Clone)]
enum Value {
    Long(i64),
    Bytes(Vec<u8>),
}
fn column_rows(column: &Column) -> Result<Vec<(i32, Vec<Value>)>> {
    match column {
        Column::Numeric(c) => Ok(c
            .iter()
            .map(|&(doc, value)| (doc, vec![Value::Long(value)]))
            .collect()),
        Column::Sorted(c) => {
            if c.docs.len() != c.ords.len() {
                return Err(failure("ordinal length mismatch"));
            }
            Ok(c.docs
                .iter()
                .zip(&c.ords)
                .map(|(&doc, &ord)| (doc, vec![Value::Long(ord)]))
                .collect())
        }
        Column::Binary(c) => {
            if c.docs.len() != c.lengths.len() {
                return Err(failure("binary length mismatch"));
            }
            let mut offset = 0usize;
            let mut rows = Vec::new();
            for (&doc, &len) in c.docs.iter().zip(&c.lengths) {
                let end = offset
                    .checked_add(
                        usize::try_from(len).map_err(|_| failure("binary length overflow"))?,
                    )
                    .ok_or_else(|| failure("binary length overflow"))?;
                let value = c
                    .bytes
                    .get(offset..end)
                    .ok_or_else(|| failure("short binary column"))?;
                rows.push((doc, vec![Value::Bytes(value.to_vec())]));
                offset = end;
            }
            if offset != c.bytes.len() {
                return Err(failure("trailing binary bytes"));
            }
            Ok(rows)
        }
        Column::SortedNumeric(c) => multi_rows(&c.docs, &c.counts, &c.values),
        Column::SortedSet(c) => multi_rows(&c.docs, &c.counts, &c.ords),
    }
}
fn multi_rows(docs: &[i32], counts: &[u32], values: &[i64]) -> Result<Vec<(i32, Vec<Value>)>> {
    if docs.len() != counts.len() {
        return Err(failure("count length mismatch"));
    }
    let mut rows = Vec::new();
    let mut offset = 0usize;
    for (&doc, &count) in docs.iter().zip(counts) {
        if count == 0 {
            return Err(failure("zero value count"));
        }
        let end = offset
            .checked_add(usize::try_from(count).map_err(|_| failure("value count overflow"))?)
            .ok_or_else(|| failure("value count overflow"))?;
        rows.push((
            doc,
            values
                .get(offset..end)
                .ok_or_else(|| failure("short multivalue column"))?
                .iter()
                .copied()
                .map(Value::Long)
                .collect(),
        ));
        offset = end;
    }
    if offset != values.len() {
        return Err(failure("trailing values"));
    }
    Ok(rows)
}
fn build_column(
    kind: DocValuesType,
    rows: Vec<(i32, Vec<Value>)>,
    dictionary: Vec<Vec<u8>>,
) -> Result<Column> {
    let mut docs = Vec::new();
    let mut counts = Vec::new();
    let mut longs = Vec::new();
    let mut lengths = Vec::new();
    let mut bytes = Vec::new();
    for (doc, values) in rows {
        docs.push(doc);
        counts.push(u32::try_from(values.len()).map_err(|_| failure("too many document values"))?);
        for value in values {
            match value {
                Value::Long(v) => longs.push(v),
                Value::Bytes(v) => {
                    lengths.push(
                        u32::try_from(v.len()).map_err(|_| failure("binary value too long"))?,
                    );
                    bytes.extend(v);
                }
            }
        }
    }
    Ok(match kind {
        DocValuesType::Numeric => Column::Numeric(docs.into_iter().zip(longs).collect()),
        DocValuesType::Binary => Column::Binary(BinaryColumn {
            docs,
            lengths,
            bytes,
        }),
        DocValuesType::Sorted => Column::Sorted(SortedOrdsColumn {
            dict: dictionary,
            docs,
            ords: longs,
        }),
        DocValuesType::SortedNumeric => Column::SortedNumeric(SortedNumericColumn {
            docs,
            counts,
            values: longs,
        }),
        DocValuesType::SortedSet => Column::SortedSet(SortedSetOrdsColumn {
            dict: dictionary,
            docs,
            counts,
            ords: longs,
        }),
        DocValuesType::None => return Err(failure("field has no doc values")),
    })
}
pub type Files = BTreeMap<String, Vec<u8>>;
#[derive(Clone, Copy)]
pub struct ReadState<'a> {
    pub files: &'a Files,
    pub segment: SegmentReadState<'a>,
}
pub trait DocValuesFormat {
    type Consumer<'a>: DocValuesConsumer;
    type Producer: DocValuesProducer;
    fn name(&self) -> &str;
    fn fields_consumer<'a>(&mut self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>>;
    fn fields_producer(&mut self, state: ReadState<'_>) -> Result<Self::Producer>;
}
pub struct CrankyDocValuesFormat<F> {
    delegate: F,
    random: Rc<RefCell<JavaRandom>>,
}
impl<F> CrankyDocValuesFormat<F> {
    pub fn new(delegate: F, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &F {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut F {
        &mut self.delegate
    }
}
impl<F: DocValuesFormat> DocValuesFormat for CrankyDocValuesFormat<F> {
    type Consumer<'a> = CrankyDocValuesConsumer<F::Consumer<'a>>;
    type Producer = F::Producer;
    fn name(&self) -> &str {
        self.delegate.name()
    }
    fn fields_consumer<'a>(&mut self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>> {
        guard(&self.random, "DocValuesFormat.fieldsConsumer")?;
        Ok(CrankyDocValuesConsumer::new(
            self.delegate.fields_consumer(state)?,
            self.random.clone(),
        ))
    }
    fn fields_producer(&mut self, state: ReadState<'_>) -> Result<Self::Producer> {
        self.delegate.fields_producer(state)
    }
}
pub struct CrankyDocValuesConsumer<C> {
    delegate: C,
    random: Rc<RefCell<JavaRandom>>,
}
impl<C> CrankyDocValuesConsumer<C> {
    pub fn new(delegate: C, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &C {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut C {
        &mut self.delegate
    }
}
impl<C: DocValuesConsumer> DocValuesConsumer for CrankyDocValuesConsumer<C> {
    fn add_numeric_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        guard(&self.random, "DocValuesConsumer.addNumericField")?;
        self.delegate.add_numeric_field(f, p)
    }
    fn add_binary_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        guard(&self.random, "DocValuesConsumer.addBinaryField")?;
        self.delegate.add_binary_field(f, p)
    }
    fn add_sorted_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        guard(&self.random, "DocValuesConsumer.addSortedField")?;
        self.delegate.add_sorted_field(f, p)
    }
    fn add_sorted_numeric_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        guard(&self.random, "DocValuesConsumer.addSortedNumericField")?;
        self.delegate.add_sorted_numeric_field(f, p)
    }
    fn add_sorted_set_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        guard(&self.random, "DocValuesConsumer.addSortedSetField")?;
        self.delegate.add_sorted_set_field(f, p)
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        guard(&self.random, "DocValuesConsumer.close")
    }
}
/// Owned producer useful for native codecs and for callers' sparse columns.
#[derive(Default)]
pub struct OwnedDocValuesProducer {
    pub columns: BTreeMap<i32, Column>,
    pub skippers: BTreeMap<i32, dv::DocValuesSkipIndex>,
    closed: bool,
    files: Files,
}
impl OwnedDocValuesProducer {
    pub fn new(columns: BTreeMap<i32, Column>) -> Self {
        Self {
            columns,
            ..Self::default()
        }
    }
    fn column(&self, field: &FieldInfo) -> Result<&Column> {
        if self.closed {
            return Err(failure("closed doc values producer"));
        }
        self.columns
            .get(&field.number)
            .filter(|column| column.kind() == field.doc_values_type)
            .ok_or_else(|| failure("missing or incompatible doc values field"))
    }
}
impl DocValuesProducer for OwnedDocValuesProducer {
    fn get_numeric(&self, f: &FieldInfo) -> Result<Cow<'_, NumericColumn>> {
        match self.column(f)? {
            Column::Numeric(c) => Ok(Cow::Borrowed(c)),
            _ => Err(failure("unsupported numeric getter")),
        }
    }
    fn get_binary(&self, f: &FieldInfo) -> Result<Cow<'_, BinaryColumn>> {
        match self.column(f)? {
            Column::Binary(c) => Ok(Cow::Borrowed(c)),
            _ => Err(failure("unsupported binary getter")),
        }
    }
    fn get_sorted(&self, f: &FieldInfo) -> Result<Cow<'_, SortedOrdsColumn>> {
        match self.column(f)? {
            Column::Sorted(c) => Ok(Cow::Borrowed(c)),
            _ => Err(failure("unsupported sorted getter")),
        }
    }
    fn get_sorted_numeric(&self, f: &FieldInfo) -> Result<Cow<'_, SortedNumericColumn>> {
        match self.column(f)? {
            Column::SortedNumeric(c) => Ok(Cow::Borrowed(c)),
            _ => Err(failure("unsupported sorted numeric getter")),
        }
    }
    fn get_sorted_set(&self, f: &FieldInfo) -> Result<Cow<'_, SortedSetOrdsColumn>> {
        match self.column(f)? {
            Column::SortedSet(c) => Ok(Cow::Borrowed(c)),
            _ => Err(failure("unsupported sorted set getter")),
        }
    }
    fn get_skipper(&self, f: &FieldInfo) -> Result<Option<&dv::DocValuesSkipIndex>> {
        self.column(f)?;
        Ok(self.skippers.get(&f.number))
    }
    fn check_integrity(&self) -> Result<()> {
        if self.closed {
            return Err(failure("closed doc values producer"));
        }
        for bytes in self.files.values() {
            checksum(bytes)?;
        }
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        self.columns.clear();
        self.skippers.clear();
        self.files.clear();
        self.closed = true;
        Ok(())
    }
}
fn checksum(bytes: &[u8]) -> Result<()> {
    let footer = bytes
        .len()
        .checked_sub(codec_util::FOOTER_LENGTH)
        .ok_or_else(|| failure("missing doc values footer"))?;
    codec_util::check_whole_file_footer(bytes, footer)?;
    Ok(())
}
fn file<'a>(files: &'a Files, extension: &str) -> Result<&'a [u8]> {
    files
        .get(extension)
        .map(Vec::as_slice)
        .ok_or_else(|| failure(format!("missing doc values file .{extension}")))
}
/// Native Lucene90 format with the same skip interval setting as Java.
#[derive(Clone, Copy, Default)]
pub struct Lucene90DocValuesFormat(pub dv::Lucene90DocValuesFormat);
pub struct Lucene90DocValuesConsumer<'a> {
    state: SegmentWriteState<'a>,
    format: dv::Lucene90DocValuesFormat,
    columns: Vec<(i32, Column)>,
    output: Option<Files>,
    closed: bool,
}
impl Lucene90DocValuesConsumer<'_> {
    pub fn output(&self) -> Option<&Files> {
        self.output.as_ref()
    }
    fn add(
        &mut self,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
        kind: DocValuesType,
    ) -> Result<()> {
        if self.closed {
            return Err(failure("closed doc values consumer"));
        }
        if field.doc_values_type != kind
            || self
                .state
                .field_info(field.number)
                .is_none_or(|info| info.name != field.name || info.doc_values_type != kind)
        {
            return Err(failure("incompatible doc values field"));
        }
        if self
            .columns
            .iter()
            .any(|(number, _)| *number == field.number)
        {
            return Err(failure("duplicate doc values field"));
        }
        let column = read_column(producer, field)?;
        validate_column(&column, self.state.max_doc)?;
        self.columns.push((field.number, column));
        Ok(())
    }
}
impl DocValuesConsumer for Lucene90DocValuesConsumer<'_> {
    fn add_numeric_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.add(f, p, DocValuesType::Numeric)
    }
    fn add_binary_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.add(f, p, DocValuesType::Binary)
    }
    fn add_sorted_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.add(f, p, DocValuesType::Sorted)
    }
    fn add_sorted_numeric_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.add(f, p, DocValuesType::SortedNumeric)
    }
    fn add_sorted_set_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.add(f, p, DocValuesType::SortedSet)
    }
    fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        let fields: Vec<_> = self
            .columns
            .iter()
            .map(|(number, column)| match column {
                Column::Numeric(c) => dv::DenseField::SparseNumeric(*number, c),
                Column::Binary(c) => dv::DenseField::BinaryColumn(*number, c),
                Column::Sorted(c) => dv::DenseField::SortedOrds(*number, c),
                Column::SortedNumeric(c) => dv::DenseField::SortedNumericColumn(*number, c),
                Column::SortedSet(c) => dv::DenseField::SortedSetOrds(*number, c),
            })
            .collect();
        let skip: Vec<_> = self
            .columns
            .iter()
            .filter_map(|(number, _)| {
                self.state
                    .field_info(*number)
                    .filter(|field| {
                        field.doc_values_skip_index_type != DocValuesSkipIndexType::None
                    })
                    .map(|_| *number)
            })
            .collect();
        let (meta, data, skip) = if fields.is_empty() {
            dv::write_no_fields(self.state.segment_id, self.state.segment_suffix)
        } else {
            dv::write_fields_with_format(
                &fields,
                &skip,
                self.state.max_doc,
                self.state.segment_id,
                self.state.segment_suffix,
                self.format,
            )?
        };
        self.output = Some(BTreeMap::from([
            ("dvm".into(), meta),
            ("dvd".into(), data),
            ("dvs".into(), skip),
        ]));
        self.closed = true;
        Ok(())
    }
}
// The measured native-open probe spent 1.21 us/open in validation's temporary
// Vec<Value> reconstruction. Validate the existing slices without copying.
fn validate_column(column: &Column, max_doc: i32) -> Result<()> {
    // Preserve the old column_rows failure priority: check the complete shape
    // before dictionary, document or value order, even for mixed defects.
    match column {
        Column::Numeric(_) => (),
        Column::Sorted(c) => {
            if c.docs.len() != c.ords.len() {
                return Err(failure("ordinal length mismatch"));
            }
        }
        Column::Binary(c) => {
            if c.docs.len() != c.lengths.len() {
                return Err(failure("binary length mismatch"));
            }
            let mut offset = 0usize;
            for &len in &c.lengths {
                let end = offset
                    .checked_add(
                        usize::try_from(len).map_err(|_| failure("binary length overflow"))?,
                    )
                    .ok_or_else(|| failure("binary length overflow"))?;
                c.bytes
                    .get(offset..end)
                    .ok_or_else(|| failure("short binary column"))?;
                offset = end;
            }
            if offset != c.bytes.len() {
                return Err(failure("trailing binary bytes"));
            }
        }
        Column::SortedNumeric(c) => validate_multi_shape(c.docs.len(), &c.counts, c.values.len())?,
        Column::SortedSet(c) => validate_multi_shape(c.docs.len(), &c.counts, c.ords.len())?,
    }
    let dictionary = match column {
        Column::Sorted(c) => Some(c.dict.as_slice()),
        Column::SortedSet(c) => Some(c.dict.as_slice()),
        _ => None,
    };
    if dictionary.is_some_and(|dict| dict.windows(2).any(|pair| pair[0] >= pair[1])) {
        return Err(failure("dictionary is not strictly sorted"));
    }
    let mut previous = None;
    match column {
        Column::Numeric(c) => {
            for &(doc, _) in c {
                validate_document(doc, max_doc, &mut previous)?;
            }
        }
        Column::Binary(c) => {
            for &doc in &c.docs {
                validate_document(doc, max_doc, &mut previous)?;
            }
        }
        Column::Sorted(c) => {
            for (&doc, &ord) in c.docs.iter().zip(&c.ords) {
                validate_document(doc, max_doc, &mut previous)?;
                validate_ordinal(ord, &c.dict)?;
            }
        }
        Column::SortedNumeric(c) => {
            validate_multi_values(&c.docs, &c.counts, &c.values, None, max_doc)?
        }
        Column::SortedSet(c) => {
            validate_multi_values(&c.docs, &c.counts, &c.ords, Some(&c.dict), max_doc)?
        }
    }
    Ok(())
}
fn validate_multi_shape(docs: usize, counts: &[u32], values: usize) -> Result<()> {
    if docs != counts.len() {
        return Err(failure("count length mismatch"));
    }
    let mut offset = 0usize;
    for &count in counts {
        if count == 0 {
            return Err(failure("zero value count"));
        }
        let end = offset
            .checked_add(usize::try_from(count).map_err(|_| failure("value count overflow"))?)
            .ok_or_else(|| failure("value count overflow"))?;
        if end > values {
            return Err(failure("short multivalue column"));
        }
        offset = end;
    }
    if offset != values {
        return Err(failure("trailing values"));
    }
    Ok(())
}
fn validate_document(doc: i32, max_doc: i32, previous: &mut Option<i32>) -> Result<()> {
    if doc < 0 || doc >= max_doc || previous.is_some_and(|previous| previous >= doc) {
        return Err(failure("invalid doc values document order"));
    }
    *previous = Some(doc);
    Ok(())
}
fn validate_ordinal(value: i64, dictionary: &[Vec<u8>]) -> Result<()> {
    let index = usize::try_from(value).map_err(|_| failure("negative ordinal"))?;
    if index >= dictionary.len() {
        return Err(failure("ordinal out of bounds"));
    }
    Ok(())
}
fn validate_multi_values(
    docs: &[i32],
    counts: &[u32],
    values: &[i64],
    dictionary: Option<&[Vec<u8>]>,
    max_doc: i32,
) -> Result<()> {
    let mut previous = None;
    let mut offset = 0usize;
    for (&doc, &count) in docs.iter().zip(counts) {
        validate_document(doc, max_doc, &mut previous)?;
        // validate_multi_shape checked conversion, addition and the full slice.
        let end = offset
            .checked_add(usize::try_from(count).expect("validated value count"))
            .expect("validated multivalue end");
        let mut prior = None;
        for &value in &values[offset..end] {
            if let Some(dictionary) = dictionary {
                validate_ordinal(value, dictionary)?;
            }
            if prior.is_some_and(|prior| prior > value || (dictionary.is_some() && prior == value))
            {
                return Err(failure("invalid multivalue order"));
            }
            prior = Some(value);
        }
        offset = end;
    }
    Ok(())
}
impl DocValuesFormat for Lucene90DocValuesFormat {
    type Consumer<'a> = Lucene90DocValuesConsumer<'a>;
    type Producer = OwnedDocValuesProducer;
    fn name(&self) -> &str {
        dv::Lucene90DocValuesFormat::NAME
    }
    fn fields_consumer<'a>(&mut self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>> {
        if state.max_doc < 0 {
            return Err(failure("negative maximum document"));
        }
        Ok(Lucene90DocValuesConsumer {
            state,
            format: self.0,
            columns: Vec::new(),
            output: None,
            closed: false,
        })
    }
    fn fields_producer(&mut self, state: ReadState<'_>) -> Result<Self::Producer> {
        let segment = state.segment;
        if segment.max_doc < 0 {
            return Err(failure("negative maximum document"));
        }
        let meta_bytes = file(state.files, "dvm")?;
        checksum(meta_bytes)?;
        let (version, meta) = dv::parse_meta(
            meta_bytes,
            segment.segment_id,
            segment.segment_suffix,
            segment.field_infos,
        )?;
        let data = file(state.files, "dvd")?;
        let skip = file(state.files, "dvs")?;
        if dv::check_data_header_footer(data, segment.segment_id, segment.segment_suffix)?
            != version
            || dv::check_skip_index_header_footer(skip, segment.segment_id, segment.segment_suffix)?
                != version
        {
            return Err(failure("doc values version mismatch"));
        }
        let mut columns = BTreeMap::new();
        // The paired asserting read probe measured native open/decode at 7.66 us.
        // Build the already-public column representation directly: temporary
        // Vec<Value> rows otherwise allocate and copy every decoded value twice.
        // As before, finish decoding a field before reporting length/count
        // conversion failures; a later byte-decoding error takes precedence.
        for field in &segment.field_infos.fields {
            let column = match field.doc_values_type {
                DocValuesType::None => continue,
                DocValuesType::Numeric => {
                    let entry = meta
                        .numeric_entry(field.number)
                        .ok_or_else(|| failure("missing numeric metadata"))?;
                    let mut reader = dv::NumericReader::new(data, entry);
                    let mut column = Vec::new();
                    for doc in 0..segment.max_doc {
                        if let Some(value) = reader.value(doc)? {
                            column.push((doc, value));
                        }
                    }
                    Column::Numeric(column)
                }
                DocValuesType::Binary => {
                    let entry = meta
                        .binary_entry(field.number)
                        .ok_or_else(|| failure("missing binary metadata"))?;
                    let mut reader = dv::BinaryReader::new(data, entry);
                    let mut column = BinaryColumn::default();
                    let mut length_error = None;
                    for doc in 0..segment.max_doc {
                        if let Some(value) = reader.value(doc)? {
                            column.docs.push(doc);
                            column
                                .lengths
                                .push(u32::try_from(value.len()).unwrap_or_else(|_| {
                                    length_error = Some(failure("binary value too long"));
                                    0
                                }));
                            column.bytes.extend_from_slice(value);
                        }
                    }
                    if let Some(error) = length_error {
                        return Err(error);
                    }
                    Column::Binary(column)
                }
                DocValuesType::Sorted => {
                    let entry = meta
                        .sorted_entry(field.number)
                        .ok_or_else(|| failure("missing sorted metadata"))?;
                    let mut column = SortedOrdsColumn {
                        dict: lucene_codecs::terms_dict::decode_all_terms(data, &entry.terms)?,
                        ..SortedOrdsColumn::default()
                    };
                    for doc in 0..segment.max_doc {
                        if let Some(ord) = dv::sorted_ord(data, entry, doc)? {
                            column.docs.push(doc);
                            column.ords.push(ord);
                        }
                    }
                    Column::Sorted(column)
                }
                DocValuesType::SortedNumeric => {
                    let entry = meta
                        .sorted_numeric_entry(field.number)
                        .ok_or_else(|| failure("missing sorted numeric metadata"))?;
                    let mut column = SortedNumericColumn::default();
                    let mut count_error = None;
                    for doc in 0..segment.max_doc {
                        let values = dv::sorted_numeric_values(data, entry, doc)?;
                        if !values.is_empty() {
                            column.docs.push(doc);
                            column
                                .counts
                                .push(u32::try_from(values.len()).unwrap_or_else(|_| {
                                    count_error = Some(failure("too many document values"));
                                    0
                                }));
                            column.values.extend(values);
                        }
                    }
                    if let Some(error) = count_error {
                        return Err(error);
                    }
                    Column::SortedNumeric(column)
                }
                DocValuesType::SortedSet => {
                    let entry = meta
                        .sorted_set_entry(field.number)
                        .ok_or_else(|| failure("missing sorted set metadata"))?;
                    let mut column = SortedSetOrdsColumn::default();
                    let mut count_error = None;
                    match &entry.kind {
                        dv::SortedSetKind::Single(entry) => {
                            column.dict =
                                lucene_codecs::terms_dict::decode_all_terms(data, &entry.terms)?;
                            for doc in 0..segment.max_doc {
                                if let Some(ord) = dv::sorted_ord(data, entry, doc)? {
                                    column.docs.push(doc);
                                    column.counts.push(1);
                                    column.ords.push(ord);
                                }
                            }
                        }
                        dv::SortedSetKind::Multi { ords, terms } => {
                            column.dict = lucene_codecs::terms_dict::decode_all_terms(data, terms)?;
                            for doc in 0..segment.max_doc {
                                let values = dv::sorted_numeric_values(data, ords, doc)?;
                                if !values.is_empty() {
                                    column.docs.push(doc);
                                    column
                                        .counts
                                        .push(u32::try_from(values.len()).unwrap_or_else(|_| {
                                            count_error = Some(failure("too many document values"));
                                            0
                                        }));
                                    column.ords.extend(values);
                                }
                            }
                        }
                    }
                    if let Some(error) = count_error {
                        return Err(error);
                    }
                    Column::SortedSet(column)
                }
            };
            validate_column(&column, segment.max_doc)?;
            columns.insert(field.number, column);
        }
        let mut skippers = BTreeMap::new();
        for (&number, entry) in &meta.skippers {
            skippers.insert(
                number,
                dv::parse_skip_index(skip, segment.segment_id, segment.segment_suffix, entry)?,
            );
        }
        Ok(OwnedDocValuesProducer {
            columns,
            skippers,
            closed: false,
            files: state.files.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn allocation_baseline_validate_column(column: &Column, max_doc: i32) -> Result<()> {
        let rows = column_rows(column)?;
        let mut previous = None;
        let dictionary = match column {
            Column::Sorted(c) => Some(&c.dict),
            Column::SortedSet(c) => Some(&c.dict),
            _ => None,
        };
        if dictionary.is_some_and(|dict| dict.windows(2).any(|pair| pair[0] >= pair[1])) {
            return Err(failure("dictionary is not strictly sorted"));
        }
        for (doc, values) in rows {
            if doc < 0 || doc >= max_doc || previous.is_some_and(|previous| previous >= doc) {
                return Err(failure("invalid doc values document order"));
            }
            previous = Some(doc);
            let mut prior = None;
            for value in values {
                if let Value::Long(value) = value {
                    if let Some(dict) = dictionary {
                        let index =
                            usize::try_from(value).map_err(|_| failure("negative ordinal"))?;
                        if index >= dict.len() {
                            return Err(failure("ordinal out of bounds"));
                        }
                    }
                    if matches!(column, Column::SortedNumeric(_) | Column::SortedSet(_))
                        && prior.is_some_and(|prior| {
                            prior > value
                                || (matches!(column, Column::SortedSet(_)) && prior == value)
                        })
                    {
                        return Err(failure("invalid multivalue order"));
                    }
                    prior = Some(value);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn slice_validation_preserves_allocation_baseline_and_mixed_defect_priority() {
        fn dictionary() -> Vec<Vec<u8>> {
            vec![b"a".to_vec(), b"b".to_vec()]
        }
        fn binary(docs: Vec<i32>, lengths: Vec<u32>, bytes: Vec<u8>) -> Column {
            Column::Binary(BinaryColumn {
                docs,
                lengths,
                bytes,
            })
        }
        fn sorted(docs: Vec<i32>, ords: Vec<i64>, dict: Vec<Vec<u8>>) -> Column {
            Column::Sorted(SortedOrdsColumn { docs, ords, dict })
        }
        fn numbers(docs: Vec<i32>, counts: Vec<u32>, values: Vec<i64>) -> Column {
            Column::SortedNumeric(SortedNumericColumn {
                docs,
                counts,
                values,
            })
        }
        fn set(docs: Vec<i32>, counts: Vec<u32>, ords: Vec<i64>, dict: Vec<Vec<u8>>) -> Column {
            Column::SortedSet(SortedSetOrdsColumn {
                docs,
                counts,
                ords,
                dict,
            })
        }
        let mut cases = vec![
            Column::Numeric(vec![]),
            Column::Numeric(vec![(0, -1), (2, 9)]),
            Column::Numeric(vec![(-1, 0)]),
            Column::Numeric(vec![(0, 0), (0, 1)]),
            Column::Numeric(vec![(2, 0), (1, 1)]),
            Column::Numeric(vec![(3, 0)]),
            binary(vec![], vec![], vec![]),
            binary(vec![0, 2], vec![0, 2], vec![0, 255]),
            binary(vec![-1], vec![], vec![1]),
            binary(vec![-1], vec![2], vec![1]),
            binary(vec![-1], vec![0], vec![1]),
            binary(vec![0, 0], vec![0, 0], vec![]),
            binary(vec![0], vec![u32::MAX], vec![]),
            sorted(vec![], vec![], vec![]),
            sorted(vec![0, 2], vec![0, 1], dictionary()),
            sorted(vec![-1], vec![], vec![b"b".to_vec(), b"a".to_vec()]),
            sorted(vec![-1], vec![-1], vec![b"b".to_vec(), b"a".to_vec()]),
            sorted(vec![0, -1], vec![-1, 0], dictionary()),
            sorted(vec![0, -1], vec![2, 0], dictionary()),
            sorted(vec![0, 0], vec![0, -1], dictionary()),
            numbers(vec![], vec![], vec![]),
            numbers(vec![0, 2], vec![2, 1], vec![-1, -1, 5]),
            numbers(vec![-1], vec![], vec![]),
            numbers(vec![-1], vec![0], vec![1]),
            numbers(vec![-1], vec![2], vec![1]),
            numbers(vec![-1], vec![1], vec![1, 2]),
            numbers(vec![-1], vec![2], vec![2, 1]),
            numbers(vec![0], vec![2], vec![2, 1]),
            numbers(vec![0, -1], vec![2, 1], vec![2, 1, 0]),
            numbers(vec![0], vec![u32::MAX], vec![]),
            set(vec![], vec![], vec![], vec![]),
            set(vec![0, 2], vec![2, 1], vec![0, 1, 1], dictionary()),
            set(
                vec![-1],
                vec![],
                vec![-1],
                vec![b"b".to_vec(), b"a".to_vec()],
            ),
            set(
                vec![-1],
                vec![0],
                vec![-1],
                vec![b"b".to_vec(), b"a".to_vec()],
            ),
            set(
                vec![-1],
                vec![2],
                vec![-1],
                vec![b"b".to_vec(), b"a".to_vec()],
            ),
            set(
                vec![-1],
                vec![1],
                vec![-1],
                vec![b"b".to_vec(), b"a".to_vec()],
            ),
            set(vec![-1], vec![1], vec![-1], dictionary()),
            set(vec![0], vec![1], vec![-1], dictionary()),
            set(vec![0], vec![1], vec![2], dictionary()),
            set(vec![0], vec![2], vec![1, 1], dictionary()),
            set(vec![0], vec![2], vec![1, 0], dictionary()),
            set(vec![0], vec![2], vec![1, 2], dictionary()),
            set(vec![0, -1], vec![2, 1], vec![1, 0, -1], dictionary()),
            set(vec![0, 0], vec![1, 1], vec![0, -1], dictionary()),
        ];
        // Dictionary comparison checks equal adjacent entries as well as reversal.
        cases.push(sorted(vec![0], vec![0], vec![b"a".to_vec(), b"a".to_vec()]));
        for column in cases {
            for max_doc in [-1, 0, 3] {
                let normalize = |result: Result<()>| {
                    result.map_err(|error| (std::mem::discriminant(&error), error.to_string()))
                };
                assert_eq!(
                    normalize(validate_column(&column, max_doc)),
                    normalize(allocation_baseline_validate_column(&column, max_doc)),
                    "{column:?}, max_doc={max_doc}"
                );
            }
        }
    }
    fn field(kind: DocValuesType) -> FieldInfo {
        FieldInfo::new("f", 0).with_doc_values(kind, DocValuesSkipIndexType::None, -1)
    }
    fn infos(kind: DocValuesType) -> FieldInfos {
        FieldInfos::new(vec![field(kind)]).unwrap()
    }
    fn state<'a>(infos: &'a FieldInfos, max_doc: i32) -> SegmentWriteState<'a> {
        SegmentWriteState {
            max_doc,
            segment_id: &[0; 16],
            segment_suffix: "",
            field_infos: infos,
        }
    }
    fn error<T>(result: Result<T>, text: &str) {
        assert!(result.err().unwrap().to_string().contains(text));
    }
    #[test]
    fn malformed_owned_columns_are_rejected_without_panics() {
        let bad = [
            Column::Numeric(vec![(-1, 1)]),
            Column::Numeric(vec![(1, 1), (1, 2)]),
            Column::Numeric(vec![(4, 1)]),
            Column::Binary(BinaryColumn {
                docs: vec![0],
                lengths: vec![],
                bytes: vec![],
            }),
            Column::Binary(BinaryColumn {
                docs: vec![0],
                lengths: vec![2],
                bytes: vec![1],
            }),
            Column::Binary(BinaryColumn {
                docs: vec![],
                lengths: vec![],
                bytes: vec![1],
            }),
            Column::Sorted(SortedOrdsColumn {
                dict: vec![b"b".to_vec(), b"a".to_vec()],
                docs: vec![],
                ords: vec![],
            }),
            Column::Sorted(SortedOrdsColumn {
                dict: vec![],
                docs: vec![0],
                ords: vec![0],
            }),
            Column::Sorted(SortedOrdsColumn {
                dict: vec![vec![]],
                docs: vec![0],
                ords: vec![-1],
            }),
            Column::Sorted(SortedOrdsColumn {
                dict: vec![],
                docs: vec![0],
                ords: vec![],
            }),
            Column::SortedNumeric(SortedNumericColumn {
                docs: vec![0],
                counts: vec![],
                values: vec![],
            }),
            Column::SortedNumeric(SortedNumericColumn {
                docs: vec![0],
                counts: vec![0],
                values: vec![],
            }),
            Column::SortedNumeric(SortedNumericColumn {
                docs: vec![0],
                counts: vec![2],
                values: vec![1],
            }),
            Column::SortedNumeric(SortedNumericColumn {
                docs: vec![],
                counts: vec![],
                values: vec![1],
            }),
            Column::SortedNumeric(SortedNumericColumn {
                docs: vec![0],
                counts: vec![2],
                values: vec![2, 1],
            }),
            Column::SortedSet(SortedSetOrdsColumn {
                dict: vec![vec![]],
                docs: vec![0],
                counts: vec![2],
                ords: vec![0, 0],
            }),
        ];
        for column in bad {
            assert!(validate_column(&column, 4).is_err(), "{column:?}");
        }
        validate_column(
            &Column::SortedNumeric(SortedNumericColumn {
                docs: vec![0],
                counts: vec![2],
                values: vec![1, 1],
            }),
            1,
        )
        .unwrap();
    }
    #[test]
    fn native_consumer_lifecycle_shape_and_field_errors() {
        let infos = infos(DocValuesType::Numeric);
        let f = &infos.fields[0];
        let mut format = Lucene90DocValuesFormat::default();
        error(
            format.fields_consumer(state(&infos, -1)),
            "negative maximum",
        );
        let good =
            OwnedDocValuesProducer::new(BTreeMap::from([(0, Column::Numeric(vec![(0, 7)]))]));
        let mut consumer = format.fields_consumer(state(&infos, 1)).unwrap();
        assert!(consumer.output().is_none());
        error(consumer.add_binary_field(f, &good), "incompatible");
        error(
            consumer.add_numeric_field(
                &FieldInfo::new("wrong", 0).with_doc_values(
                    DocValuesType::Numeric,
                    DocValuesSkipIndexType::None,
                    -1,
                ),
                &good,
            ),
            "incompatible",
        );
        let bad = OwnedDocValuesProducer::new(BTreeMap::from([(0, Column::Numeric(vec![(2, 7)]))]));
        error(consumer.add_numeric_field(f, &bad), "document order");
        consumer.add_numeric_field(f, &good).unwrap();
        error(consumer.add_numeric_field(f, &good), "duplicate");
        consumer.close().unwrap();
        let output = consumer.output().unwrap().clone();
        consumer.close().unwrap();
        assert_eq!(consumer.output().unwrap(), &output);
        error(consumer.add_numeric_field(f, &good), "closed");
        let mut producer = format
            .fields_producer(ReadState {
                files: &output,
                segment: SegmentReadState {
                    max_doc: 1,
                    segment_id: &[0; 16],
                    segment_suffix: "",
                    field_infos: &infos,
                },
            })
            .unwrap();
        assert_eq!(&*producer.get_numeric(f).unwrap(), &[(0, 7)]);
        assert!(producer.get_skipper(f).unwrap().is_none());
        error(producer.get_binary(f), "unsupported");
        error(producer.get_sorted(f), "unsupported");
        error(producer.get_sorted_numeric(f), "unsupported");
        error(producer.get_sorted_set(f), "unsupported");
        error(
            producer.get_numeric(&field(DocValuesType::Binary)),
            "incompatible",
        );
        producer.check_integrity().unwrap();
        producer.close().unwrap();
        producer.close().unwrap();
        error(producer.check_integrity(), "closed");
        error(producer.get_numeric(f), "closed");
        for extension in ["dvm", "dvd", "dvs"] {
            let mut damaged = output.clone();
            damaged.remove(extension);
            error(
                format.fields_producer(ReadState {
                    files: &damaged,
                    segment: SegmentReadState {
                        max_doc: 1,
                        segment_id: &[0; 16],
                        segment_suffix: "",
                        field_infos: &infos,
                    },
                }),
                "missing",
            );
        }
        let mut damaged = output.clone();
        damaged.get_mut("dvm").unwrap()[8] ^= 1;
        assert!(format
            .fields_producer(ReadState {
                files: &damaged,
                segment: SegmentReadState {
                    max_doc: 1,
                    segment_id: &[0; 16],
                    segment_suffix: "",
                    field_infos: &infos
                }
            })
            .is_err());
        let mut producer = format
            .fields_producer(ReadState {
                files: &output,
                segment: SegmentReadState {
                    max_doc: 1,
                    segment_id: &[0; 16],
                    segment_suffix: "",
                    field_infos: &infos,
                },
            })
            .unwrap();
        producer.files.get_mut("dvd").unwrap()[8] ^= 1;
        assert!(producer.check_integrity().is_err());
        error(checksum(&[]), "missing");
    }
    #[test]
    fn mapped_views_validate_identity_maps_masks_and_keep_unpruned_terms_without_live_docs() {
        let infos = infos(DocValuesType::Sorted);
        let f = &infos.fields[0];
        let producer = OwnedDocValuesProducer::new(BTreeMap::from([(
            0,
            Column::Sorted(SortedOrdsColumn {
                dict: vec![b"deleted".to_vec(), b"live".to_vec()],
                docs: vec![0, 1],
                ords: vec![0, 1],
            }),
        )]));
        for live in [None, Some(&[false, true][..])] {
            let sources = [MergeSource {
                producer: Some(&producer),
                field_infos: &infos,
                max_doc: 2,
                doc_map: &[-1, 0],
                live_docs: live,
            }];
            let merge = MergeState {
                sources: &sources,
                field_infos: &infos,
                check_aborted: &|| Ok(()),
            };
            let mut mapped = MappedProducer::new(f, &merge);
            mapped.ordinals = Some(ordinal_map(f, &merge).unwrap());
            let column = mapped.get_sorted(f).unwrap();
            assert_eq!(column.docs, vec![0]);
            assert_eq!(column.dict.len(), if live.is_some() { 1 } else { 2 });
            assert_eq!(column.ords, vec![if live.is_some() { 0 } else { 1 }]);
            error(mapped.get_sorted(&f.clone()), "identity");
            error(mapped.get_numeric(f), "unsupported");
            error(mapped.get_binary(f), "unsupported");
            error(mapped.get_sorted_numeric(f), "unsupported");
            error(mapped.get_sorted_set(f), "unsupported");
            error(mapped.check_integrity(), "unsupported");
            error(mapped.close(), "unsupported");
        }
        for map in [&[][..], &[-2, 0][..]] {
            let sources = [MergeSource {
                producer: Some(&producer),
                field_infos: &infos,
                max_doc: 2,
                doc_map: map,
                live_docs: None,
            }];
            let merge = MergeState {
                sources: &sources,
                field_infos: &infos,
                check_aborted: &|| Ok(()),
            };
            assert!(MappedProducer::new(f, &merge).get_sorted(f).is_err());
        }
        let sources = [MergeSource {
            producer: Some(&producer),
            field_infos: &infos,
            max_doc: 1,
            doc_map: &[0, 1],
            live_docs: Some(&[true]),
        }];
        let merge = MergeState {
            sources: &sources,
            field_infos: &infos,
            check_aborted: &|| Ok(()),
        };
        assert!(ordinal_map(f, &merge).is_err());
        assert!(MappedProducer::new(f, &merge).get_sorted(f).is_err());
    }
    #[test]
    fn default_getters_and_no_value_fields_are_typed_errors() {
        struct Empty;
        impl DocValuesProducer for Empty {
            fn check_integrity(&self) -> Result<()> {
                Ok(())
            }
            fn close(&mut self) -> Result<()> {
                Ok(())
            }
        }
        let empty = Empty;
        let f = field(DocValuesType::Numeric);
        error(empty.get_numeric(&f), "unsupported");
        error(empty.get_binary(&f), "unsupported");
        error(empty.get_sorted(&f), "unsupported");
        error(empty.get_sorted_numeric(&f), "unsupported");
        error(empty.get_sorted_set(&f), "unsupported");
        assert!(empty.get_skipper(&f).unwrap().is_none());
        error(
            read_column(&empty, &FieldInfo::new("f", 0)),
            "no doc values",
        );
        error(
            build_column(DocValuesType::None, Vec::new(), Vec::new()),
            "no doc values",
        );
    }
    #[test]
    fn merge_skips_absent_incompatible_sources_and_plain_fields() {
        let infos = FieldInfos::new(vec![
            field(DocValuesType::Numeric),
            FieldInfo::new("plain", 1),
        ])
        .unwrap();
        let missing = FieldInfos::new(Vec::new()).unwrap();
        let incompatible = self::infos(DocValuesType::Binary);
        let producer =
            OwnedDocValuesProducer::new(BTreeMap::from([(0, Column::Numeric(vec![(0, 7)]))]));
        let sources = [
            MergeSource {
                producer: None,
                field_infos: &infos,
                max_doc: 1,
                doc_map: &[0],
                live_docs: None,
            },
            MergeSource {
                producer: Some(&producer),
                field_infos: &missing,
                max_doc: 1,
                doc_map: &[0],
                live_docs: None,
            },
            MergeSource {
                producer: Some(&producer),
                field_infos: &incompatible,
                max_doc: 1,
                doc_map: &[0],
                live_docs: None,
            },
            MergeSource {
                producer: Some(&producer),
                field_infos: &infos,
                max_doc: 1,
                doc_map: &[0],
                live_docs: None,
            },
        ];
        let mut format = CrankyDocValuesFormat::new(
            Lucene90DocValuesFormat::default(),
            Rc::new(RefCell::new(JavaRandom::new(42))),
        );
        assert_eq!(format.name(), "Lucene90");
        assert_eq!(format.delegate().name(), "Lucene90");
        assert_eq!(format.delegate_mut().name(), "Lucene90");
        let mut consumer = format.fields_consumer(state(&infos, 1)).unwrap();
        consumer
            .merge(&MergeState {
                sources: &sources,
                field_infos: &infos,
                check_aborted: &|| Ok(()),
            })
            .unwrap();
        consumer.close().unwrap();
        let output = consumer.delegate_mut().output().unwrap().clone();
        assert_eq!(consumer.delegate().output(), Some(&output));
        let restored = format
            .fields_producer(ReadState {
                files: &output,
                segment: SegmentReadState {
                    max_doc: 1,
                    segment_id: &[0; 16],
                    segment_suffix: "",
                    field_infos: &infos,
                },
            })
            .unwrap();
        assert_eq!(&*restored.get_numeric(&infos.fields[0]).unwrap(), &[(0, 7)]);
        assert_eq!(restored.columns.len(), 1);
        let binary = field(DocValuesType::Binary);
        let producer = OwnedDocValuesProducer::new(BTreeMap::from([(
            0,
            Column::Binary(BinaryColumn::default()),
        )]));
        error(producer.get_numeric(&binary), "unsupported numeric");
    }

    #[test]
    fn native_read_rejects_negative_max_doc_and_valid_but_mismatched_versions() {
        let infos = FieldInfos::new(Vec::new()).unwrap();
        let mut format = Lucene90DocValuesFormat::default();
        let mut consumer = format.fields_consumer(state(&infos, 0)).unwrap();
        consumer.close().unwrap();
        let output = consumer.output().unwrap().clone();
        let segment = SegmentReadState {
            max_doc: 0,
            segment_id: &[0; 16],
            segment_suffix: "",
            field_infos: &infos,
        };
        error(
            format.fields_producer(ReadState {
                files: &output,
                segment: SegmentReadState {
                    max_doc: -1,
                    ..segment
                },
            }),
            "negative maximum",
        );
        for (extension, codec) in [
            ("dvd", "Lucene90DocValuesData"),
            ("dvs", "Lucene90DocValuesSkipIndex"),
        ] {
            let mut damaged = output.clone();
            let mut bytes = Vec::new();
            codec_util::write_index_header(&mut bytes, codec, 1, &[0; 16], "");
            codec_util::write_footer(&mut bytes);
            damaged.insert(extension.to_string(), bytes);
            error(
                format.fields_producer(ReadState {
                    files: &damaged,
                    segment,
                }),
                "version mismatch",
            );
        }
    }
}

#[cfg(test)]
mod callback_order_tests {
    use super::*;
    struct Source(RefCell<usize>);
    impl DocValuesProducer for Source {
        fn get_sorted(&self, _: &FieldInfo) -> Result<Cow<'_, SortedOrdsColumn>> {
            *self.0.borrow_mut() += 1;
            Ok(Cow::Owned(SortedOrdsColumn::default()))
        }
        fn check_integrity(&self) -> Result<()> {
            Ok(())
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }
    #[test]
    fn unsupported_mapped_getters_do_not_reenter_sources() {
        let infos = FieldInfos::new(vec![FieldInfo::new("f", 0).with_doc_values(
            DocValuesType::Sorted,
            DocValuesSkipIndexType::None,
            -1,
        )])
        .unwrap();
        let source = Source(RefCell::new(0));
        let sources = [MergeSource {
            producer: Some(&source),
            field_infos: &infos,
            max_doc: 0,
            doc_map: &[],
            live_docs: None,
        }];
        let state = MergeState {
            sources: &sources,
            field_infos: &infos,
            check_aborted: &|| Ok(()),
        };
        let mapped = MappedProducer::new(&infos.fields[0], &state);
        assert!(mapped.get_numeric(&infos.fields[0]).is_err());
        assert!(mapped.get_binary(&infos.fields[0]).is_err());
        assert!(mapped.get_sorted_numeric(&infos.fields[0]).is_err());
        assert!(mapped.get_sorted_set(&infos.fields[0]).is_err());
        assert_eq!(*source.0.borrow(), 0);
        assert!(mapped.get_sorted(&infos.fields[0]).unwrap().docs.is_empty());
        assert_eq!(*source.0.borrow(), 1);
    }
}

#[cfg(test)]
mod lazy_merge_error_order_tests {
    use super::*;

    struct Callbacks {
        events: RefCell<Vec<String>>,
        random: Rc<RefCell<JavaRandom>>,
    }
    impl Callbacks {
        fn record(&self, event: impl std::fmt::Display) {
            // Inspect continuation without consuming the wrapper's shared RNG.
            let continuation = self.random.borrow().clone().next_int();
            self.events
                .borrow_mut()
                .push(format!("{event}:{continuation}"));
        }
    }
    struct Source {
        index: usize,
        fails: bool,
        calls: Rc<Callbacks>,
    }
    impl Source {
        fn getter(&self, kind: &str) -> Result<()> {
            self.calls.record(format!("getter:{}:{kind}", self.index));
            if self.fails {
                Err(failure(format!("source {} getter failure", self.index)))
            } else {
                Ok(())
            }
        }
    }
    impl DocValuesProducer for Source {
        fn get_numeric(&self, _: &FieldInfo) -> Result<Cow<'_, NumericColumn>> {
            self.getter("numeric")?;
            Ok(Cow::Owned(vec![(0, 17)]))
        }
        fn get_binary(&self, _: &FieldInfo) -> Result<Cow<'_, BinaryColumn>> {
            self.getter("binary")?;
            Ok(Cow::Owned(BinaryColumn {
                docs: vec![0],
                lengths: vec![2],
                bytes: vec![0, 255],
            }))
        }
        fn get_sorted_numeric(&self, _: &FieldInfo) -> Result<Cow<'_, SortedNumericColumn>> {
            self.getter("sortedNumeric")?;
            Ok(Cow::Owned(SortedNumericColumn {
                docs: vec![0],
                counts: vec![2],
                values: vec![3, 3],
            }))
        }
        fn check_integrity(&self) -> Result<()> {
            self.calls.record(format!("integrity:{}", self.index));
            Ok(())
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }
    struct RequestingConsumer(Rc<Callbacks>);
    impl RequestingConsumer {
        fn request(
            &self,
            kind: &str,
            field: &FieldInfo,
            producer: &dyn DocValuesProducer,
        ) -> Result<()> {
            self.0.record(format!("delegate:{kind}"));
            // Unlike the differential trace's identity-only consumer, this
            // delegate requests the mapped getter and observes its source errors.
            read_column(producer, field)?;
            Ok(())
        }
    }
    impl DocValuesConsumer for RequestingConsumer {
        fn add_numeric_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
            self.request("numeric", f, p)
        }
        fn add_binary_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
            self.request("binary", f, p)
        }
        fn add_sorted_numeric_field(
            &mut self,
            f: &FieldInfo,
            p: &dyn DocValuesProducer,
        ) -> Result<()> {
            self.request("sortedNumeric", f, p)
        }
        fn add_sorted_field(&mut self, _: &FieldInfo, _: &dyn DocValuesProducer) -> Result<()> {
            Err(failure("unexpected sorted add"))
        }
        fn add_sorted_set_field(&mut self, _: &FieldInfo, _: &dyn DocValuesProducer) -> Result<()> {
            Err(failure("unexpected sorted set add"))
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn lazy_source_getter_errors_follow_the_guard_and_preserve_callback_rng_order() {
        for (kind, name, operation) in [
            (DocValuesType::Numeric, "numeric", "addNumericField"),
            (DocValuesType::Binary, "binary", "addBinaryField"),
            (
                DocValuesType::SortedNumeric,
                "sortedNumeric",
                "addSortedNumericField",
            ),
        ] {
            for failing_source in [0, 1] {
                // Seed 42 passes the first guard; seed 18 fails it. The latter
                // proves getters remain lazy, even when they would throw.
                for seed in [42, 18] {
                    let random = Rc::new(RefCell::new(JavaRandom::new(seed)));
                    let calls = Rc::new(Callbacks {
                        events: RefCell::new(Vec::new()),
                        random: random.clone(),
                    });
                    let infos = FieldInfos::new(vec![FieldInfo::new("f", 0).with_doc_values(
                        kind,
                        DocValuesSkipIndexType::None,
                        -1,
                    )])
                    .unwrap();
                    let first = Source {
                        index: 0,
                        fails: failing_source == 0,
                        calls: calls.clone(),
                    };
                    let second = Source {
                        index: 1,
                        fails: failing_source == 1,
                        calls: calls.clone(),
                    };
                    let sources = [
                        MergeSource {
                            producer: Some(&first),
                            field_infos: &infos,
                            max_doc: 1,
                            doc_map: &[0],
                            live_docs: None,
                        },
                        MergeSource {
                            producer: Some(&second),
                            field_infos: &infos,
                            max_doc: 1,
                            doc_map: &[1],
                            live_docs: None,
                        },
                    ];
                    let abort = || {
                        calls.record("abort");
                        Ok(())
                    };
                    let merge = MergeState {
                        sources: &sources,
                        field_infos: &infos,
                        check_aborted: &abort,
                    };
                    let mut consumer = CrankyDocValuesConsumer::new(
                        RequestingConsumer(calls.clone()),
                        random.clone(),
                    );
                    let before = JavaRandom::new(seed).next_int();
                    let mut expected_random = JavaRandom::new(seed);
                    let fails_guard = expected_random.next_int_bounded(100) == 0;
                    assert_eq!(fails_guard, seed == 18);
                    let after = expected_random.next_int();
                    let mut expected = vec![
                        format!("abort:{before}"),
                        format!("integrity:0:{before}"),
                        format!("abort:{before}"),
                        format!("integrity:1:{before}"),
                    ];
                    let error = consumer.merge(&merge).unwrap_err().to_string();
                    if fails_guard {
                        assert_eq!(
                            error,
                            format!("Fake IOException from DocValuesConsumer.{operation}()")
                        );
                    } else {
                        expected.push(format!("delegate:{name}:{after}"));
                        expected.push(format!("getter:0:{name}:{after}"));
                        if failing_source == 1 {
                            expected.push(format!("getter:1:{name}:{after}"));
                        }
                        assert_eq!(error, format!("source {failing_source} getter failure"));
                    }
                    assert_eq!(
                        *calls.events.borrow(),
                        expected,
                        "{name}, source {failing_source}, seed {seed}"
                    );
                    assert_eq!(
                        random.borrow_mut().next_int(),
                        after,
                        "getter failure must not consume another RNG draw"
                    );
                }
            }
        }
    }
}
