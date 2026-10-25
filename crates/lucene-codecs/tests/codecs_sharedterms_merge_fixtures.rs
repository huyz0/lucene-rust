//! Merge actual Java shared-terms segments, rather than rebuilding a merge
//! from its already-merged postings. The default-merge callback is exercised
//! with the sorted fixture and an independent field-by-field source walk.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use lucene_codecs::field_infos::{self, FieldInfos, IndexOptions};
use lucene_codecs::postings::{self, DocInput, PayInput, PosInput, PostingsFlags};
use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::uniformsplit::sharedterms::{
    StMergeSegment, StUniformSplitPostingsFormat, StUniformSplitTermsReader,
};
use lucene_codecs::uniformsplit::UniformSplitOutput;

fn root(dataset: &str, case: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(dataset)
        .join(case)
}

fn format() -> StUniformSplitPostingsFormat {
    StUniformSplitPostingsFormat::new(8, 2, None, None, true).unwrap()
}

struct Segment {
    path: PathBuf,
    name: String,
    id: [u8; 16],
    infos: FieldInfos,
    suffix: String,
    reader: StUniformSplitTermsReader,
    doc: Vec<u8>,
    pos: Vec<u8>,
    pay: Vec<u8>,
}

impl Segment {
    fn open(path: PathBuf, max_doc: i32) -> Self {
        let name = fs::read_dir(&path)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .find_map(|name| name.strip_suffix(".si").map(str::to_string))
            .unwrap();
        let si = fs::read(path.join(format!("{name}.si"))).unwrap();
        let at = 4 + 1 + usize::from(si[4]) + 4;
        let id = si[at..at + 16].try_into().unwrap();
        let infos = field_infos::parse(
            &fs::read(path.join(format!("{name}.fnm"))).unwrap(),
            &id,
            "",
        )
        .unwrap();
        let field = infos.field_by_name("docs").unwrap();
        let attr = |name: &str| {
            field
                .attributes
                .iter()
                .find(|(k, _)| k == name)
                .unwrap()
                .1
                .as_str()
        };
        let suffix = format!(
            "{}_{}",
            attr("PerFieldPostingsFormat.format"),
            attr("PerFieldPostingsFormat.suffix")
        );
        let file = |ext| fs::read(path.join(format!("{name}_{suffix}.{ext}"))).unwrap();
        let reader = format()
            .open(
                file("stustb"),
                &file("stustd"),
                &SegmentReadState {
                    max_doc,
                    segment_id: &id,
                    segment_suffix: &suffix,
                    field_infos: &infos,
                },
            )
            .unwrap();
        let (doc, pos, pay) = (file("doc"), file("pos"), file("pay"));
        Self {
            path,
            name,
            id,
            infos,
            suffix,
            reader,
            doc,
            pos,
            pay,
        }
    }

    fn state(&self, max_doc: i32) -> SegmentWriteState<'_> {
        SegmentWriteState {
            max_doc,
            segment_id: &self.id,
            segment_suffix: &self.suffix,
            field_infos: &self.infos,
        }
    }

    fn file(&self, ext: &str) -> Vec<u8> {
        fs::read(
            self.path
                .join(format!("{}_{}.{ext}", self.name, self.suffix)),
        )
        .unwrap()
    }
}

fn maps(root: &Path) -> Vec<Vec<i32>> {
    fs::read_to_string(root.join("docmaps.tsv"))
        .unwrap()
        .lines()
        .map(|line| {
            line.split('\t')
                .skip(1)
                .map(|n| n.parse().unwrap())
                .collect()
        })
        .collect()
}

fn norms(root: &Path) -> Vec<(i32, Vec<i64>)> {
    fs::read_to_string(root.join("norms.tsv"))
        .unwrap()
        .lines()
        .map(|line| {
            let mut parts = line.split('\t');
            (
                parts.next().unwrap().parse().unwrap(),
                parts.map(|v| v.parse().unwrap()).collect(),
            )
        })
        .collect()
}

fn assert_files(output: &UniformSplitOutput, segment: &Segment) {
    for (ext, actual) in [
        ("stustb", &output.ustb),
        ("stustd", &output.ustd),
        ("doc", &output.postings.doc),
        ("pos", &output.postings.pos),
        ("pay", &output.postings.pay),
        ("psm", &output.postings.psm),
    ] {
        let expected = segment.file(ext);
        let mismatch = actual.iter().zip(&expected).position(|(a, b)| a != b);
        assert!(
            mismatch.is_none() && actual.len() == expected.len(),
            "{}: {ext}: first difference {mismatch:?}, lengths Rust={} Java={}",
            segment.path.display(),
            actual.len(),
            expected.len()
        );
    }
}

/// Independent default-merge adapter: walk every field separately, map docs,
/// sort the survivors, then flush. It intentionally does not use the union
/// dictionary, merge queue, or StMergingTermsEnum under test.
fn default_merge(
    sources: &[StMergeSegment<'_>],
    norms: &[FieldNorms<'_>],
    state: &SegmentWriteState<'_>,
    sorted: bool,
) -> UniformSplitOutput {
    struct Doc {
        id: i32,
        freq: i32,
        positions: Vec<postings::Position>,
    }
    let mut fields = Vec::new();
    let mut targets: Vec<_> = state.field_infos.fields.iter().collect();
    if sorted {
        // FieldsConsumer.merge uses MappedMultiFields: field names are
        // lexicographically ordered. The specialized merge instead uses
        // mergeFieldInfos, which iterates in field-number order.
        targets.sort_by(|a, b| a.name.cmp(&b.name));
    } else {
        targets.sort_by_key(|field| field.number);
    }
    for target in targets {
        let mut terms = BTreeMap::<Vec<u8>, Vec<Doc>>::new();
        for source in sources {
            if source.field_infos.field_by_name(&target.name).is_none() {
                continue;
            }
            let Some(reader) = source.reader else {
                continue;
            };
            let Some(field) = reader.terms(&target.name) else {
                continue;
            };
            let mut walk = field.iter();
            while let Some(term) = walk.next().unwrap().map(<[u8]>::to_vec) {
                let docs = walk.postings(source.doc, PostingsFlags::Freqs).unwrap();
                let term_state = walk.term_state().unwrap();
                let positions = if target.index_options.subsumes_positions() {
                    postings::read_positions(
                        source.pos.unwrap(),
                        source.pay,
                        term_state.meta,
                        &docs.freqs,
                        term_state.total_term_freq,
                        field.field_info().index_options,
                        field.has_payloads(),
                    )
                    .unwrap()
                } else {
                    Vec::new()
                };
                for (index, &doc) in docs.docs.iter().enumerate() {
                    let mapped = source.doc_map[doc as usize];
                    if mapped < 0 {
                        continue;
                    }
                    terms.entry(term.clone()).or_default().push(Doc {
                        id: mapped,
                        freq: if target.index_options == IndexOptions::Docs {
                            1
                        } else {
                            docs.freqs[index]
                        },
                        positions: positions.get(index).cloned().unwrap_or_default(),
                    });
                }
            }
        }
        let mut postings = Vec::new();
        let mut seen = BTreeSet::new();
        for (term, mut docs) in terms {
            docs.sort_by_key(|doc| doc.id);
            let mut term = TermPostings {
                term,
                ..TermPostings::default()
            };
            for doc in docs {
                seen.insert(doc.id);
                term.docs.push((doc.id, doc.freq));
                if target.index_options.subsumes_positions() {
                    term.positions
                        .push(doc.positions.iter().map(|p| p.position).collect());
                    if target.index_options.subsumes_offsets() {
                        term.offsets.push(
                            doc.positions
                                .iter()
                                .map(|p| (p.start_offset, p.end_offset))
                                .collect(),
                        );
                    }
                    if target.store_payloads {
                        for position in &doc.positions {
                            term.payload_lengths.push(position.payload.len() as u32);
                            term.payload_bytes.extend_from_slice(&position.payload);
                        }
                    }
                }
            }
            postings.push(term);
        }
        if !postings.is_empty() {
            fields.push((target, seen.len() as i32, postings));
        }
    }
    let inputs: Vec<_> = fields
        .iter()
        .map(|(field, count, terms)| FieldPostingsInput {
            field_number: field.number,
            index_options: field.index_options,
            has_payloads: field.store_payloads,
            doc_count: *count,
            terms: terms.as_slice(),
        })
        .collect();
    format().write(&inputs, norms, state).unwrap()
}

#[test]
fn specialized_and_sorted_fallback_merges_equal_every_java_file() {
    check_merges("codecs_sharedterms_merge", false);
}

#[test]
fn full_position_blocks_tails_and_deleted_position_fields_equal_java() {
    check_merges("codecs_sharedterms_merge_blocks", true);
}

fn check_position_boundaries(sources: &[StMergeSegment<'_>]) {
    let block = i64::from(postings::BLOCK_SIZE);
    for source in sources {
        let reader = source.reader.unwrap();
        let mut common = reader.terms("pay").unwrap().iter();
        assert!(common.seek_exact(b"common").unwrap());
        let total = common.total_term_freq().unwrap();
        assert!(
            total > block && total % block != 0,
            "source common must span a full position block and a tail: {total}"
        );
        let docs = common.postings(source.doc, PostingsFlags::Freqs).unwrap();
        // Rows distinguish live/deleted documents; columns distinguish
        // occurrences in the full blocks and in the following tail.
        let mut seen = [[false; 2]; 2];
        let tail_start = total / block * block;
        let mut start = 0;
        let mut crosses_block = false;
        for (&doc, &freq) in docs.docs.iter().zip(&docs.freqs) {
            let end = start + i64::from(freq);
            crosses_block |= start / block != (end - 1) / block;
            let deleted = usize::from(source.doc_map[doc as usize] == -1);
            if start < tail_start {
                seen[deleted][0] = true;
            }
            if end > tail_start {
                seen[deleted][1] = true;
            }
            start = end;
        }
        assert_eq!(start, total);
        assert!(
            crosses_block,
            "a document must span a position block boundary"
        );
        assert_eq!(seen, [[true; 2]; 2], "deletions must occur on both sides");

        let mut removed = reader.terms("gonepos").unwrap().iter();
        assert!(removed.seek_exact(b"removed").unwrap());
        let total = removed.total_term_freq().unwrap();
        assert!(total > block && total % block != 0);
        let docs = removed.postings(source.doc, PostingsFlags::Freqs).unwrap();
        assert!(!docs.docs.is_empty());
        assert!(docs
            .docs
            .iter()
            .all(|&doc| source.doc_map[doc as usize] == -1));
    }
}

fn check_merges(dataset: &str, position_blocks: bool) {
    for (case, sorted) in [("unsorted", false), ("sorted", true)] {
        let root = root(dataset, case);
        let max_doc = fs::read_to_string(root.join("max-doc.tsv"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let target = Segment::open(root.join("merged"), max_doc);
        let segments: Vec<_> = (0..3)
            .map(|i| Segment::open(root.join(format!("source{i}")), 300))
            .collect();
        assert_ne!(
            segments[0].infos.field_by_name("pay").unwrap().number,
            segments[1].infos.field_by_name("pay").unwrap().number
        );
        let maps = maps(&root);
        assert!(maps.iter().flatten().any(|&doc| doc == -1));
        let docs: Vec<_> = segments
            .iter()
            .map(|s| DocInput::open(&s.doc, &s.id, &s.suffix).unwrap())
            .collect();
        let positions: Vec<_> = segments
            .iter()
            .map(|s| PosInput::open(&s.pos, &s.id, &s.suffix).unwrap())
            .collect();
        let payloads: Vec<_> = segments
            .iter()
            .map(|s| PayInput::open(&s.pay, &s.id, &s.suffix).unwrap())
            .collect();
        let sources: Vec<_> = segments
            .iter()
            .enumerate()
            .map(|(i, s)| StMergeSegment {
                reader: Some(&s.reader),
                field_infos: &s.infos,
                doc_map: &maps[i],
                doc: &docs[i],
                pos: Some(&positions[i]),
                pay: Some(&payloads[i]),
            })
            .collect();
        if position_blocks {
            check_position_boundaries(&sources);
        }
        let owned_norms = norms(&root);
        let norms: Vec<_> = owned_norms
            .iter()
            .map(|(number, values)| FieldNorms {
                field_number: *number,
                values,
            })
            .collect();
        let state = target.state(max_doc);
        let mut fallback_calls = 0;
        let output = format()
            .merge(&sources, &norms, &state, sorted, || {
                fallback_calls += 1;
                Ok(default_merge(&sources, &norms, &state, sorted))
            })
            .unwrap();
        assert_eq!(fallback_calls, usize::from(sorted));
        assert_files(&output, &target);
        // The field whose every posting was deleted has no terms metadata.
        let reader = format()
            .open(
                output.ustb,
                &output.ustd,
                &SegmentReadState {
                    max_doc,
                    segment_id: &target.id,
                    segment_suffix: &target.suffix,
                    field_infos: &target.infos,
                },
            )
            .unwrap();
        assert!(reader.terms("gone").is_none());
        if position_blocks {
            assert!(reader.terms("gonepos").is_none());
        }
        for line in fs::read_to_string(root.join("stats.tsv")).unwrap().lines() {
            let row: Vec<_> = line.split('\t').collect();
            let field = reader.terms(row[0]).unwrap();
            assert_eq!(field.size(), row[1].parse::<i64>().unwrap());
            assert_eq!(field.doc_count(), row[2].parse::<i32>().unwrap());
            assert_eq!(field.sum_doc_freq(), row[3].parse::<i64>().unwrap());
            assert_eq!(field.sum_total_term_freq(), row[4].parse::<i64>().unwrap());
        }
        if !sorted {
            let generic = default_merge(&sources, &norms, &state, false);
            assert_files(&generic, &target);
        }
    }
}
