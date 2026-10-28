//! Microbenchmarks for the `lucene-util` core structures the perf review
//! covered: bitset range scans and word-wise range ops, the sparse bitset's
//! HNSW-style `get_and_set`, run-automaton stepping over a term list,
//! determinization, `BytesRefHash` add/find/sort, packed bulk decoding and
//! single-block random access, and `bytes_difference`.
//!
//! Run with:
//! `cargo bench -p lucene-util --bench util_core -- --warm-up-time 1 --measurement-time 3`

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use lucene_util::automaton::{
    operations, ByteRunAutomaton, ByteRunnable, CharacterRunAutomaton, CompiledAutomaton,
    LevenshteinAutomata, NfaRunAutomaton, RegExp, Utf32ToUtf8,
};
use lucene_util::bits::{AnyBitSet, Bits};
use lucene_util::bytes_ref_hash::BytesRefHash;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::live_docs::{LiveDocs, SparseLiveDocs};
use lucene_util::packed::{BulkOperation, Format, Packed64SingleBlock, Reader};
use lucene_util::sparse_fixed_bit_set::SparseFixedBitSet;
use lucene_util::string_helper::bytes_difference;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 17
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn random_terms(n: usize, min: usize, max: usize, alphabet: &[u8], seed: u64) -> Vec<Vec<u8>> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| {
            let len = min + r.below((max - min + 1) as u64) as usize;
            (0..len)
                .map(|_| alphabet[r.below(alphabet.len() as u64) as usize])
                .collect()
        })
        .collect()
}

fn half_set(n: usize, seed: u64) -> FixedBitSet {
    let mut r = Lcg(seed);
    let mut b = FixedBitSet::new(n);
    for i in 0..n {
        if r.next() & 1 == 1 {
            b.set(i);
        }
    }
    b
}

fn bench_fixed_bit_set(c: &mut Criterion) {
    const N: usize = 2_000_000;
    let mut sparse = FixedBitSet::new(N);
    sparse.set(N - 1);
    c.bench_function("fixed_bit_set/next_set_bit_in_range_sparse", |b| {
        b.iter(|| {
            let mut hits = 0usize;
            for i in (0..N - 64).step_by(8192) {
                if sparse.next_set_bit_in_range(black_box(i), i + 64).is_some() {
                    hits += 1;
                }
            }
            black_box(hits)
        })
    });
    let mut full = FixedBitSet::new(N);
    full.set_range(0, N);
    full.clear(N - 1);
    c.bench_function("fixed_bit_set/next_clear_bit_in_range_full", |b| {
        b.iter(|| {
            let mut hits = 0usize;
            for i in (0..N - 64).step_by(8192) {
                if full.next_clear_bit_in_range(black_box(i), i + 64).is_some() {
                    hits += 1;
                }
            }
            black_box(hits)
        })
    });

    const M: usize = 1 << 20;
    let src = half_set(M + 64, 1);
    let dst0 = half_set(M + 64, 2);
    c.bench_function("fixed_bit_set/or_range_misaligned", |b| {
        b.iter_batched_ref(
            || dst0.clone(),
            |d| FixedBitSet::or_range(&src, 3, d, 17, M),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("fixed_bit_set/and_range_misaligned", |b| {
        b.iter_batched_ref(
            || dst0.clone(),
            |d| FixedBitSet::and_range(&src, 3, d, 17, M),
            BatchSize::SmallInput,
        )
    });
    // Live docs with 1% deletions, masked through `&dyn Bits` as the
    // live-docs writer does.
    let mut live = FixedBitSet::new(M);
    live.set_range(0, M);
    let mut r = Lcg(3);
    for _ in 0..M / 100 {
        live.clear(r.below(M as u64) as usize);
    }
    let mut chunk0 = FixedBitSet::new(1024);
    chunk0.set_range(0, 1024);
    c.bench_function("fixed_bit_set/apply_mask_dyn_bits_1m", |b| {
        let bits: &dyn Bits = &live;
        b.iter(|| {
            let mut total = 0usize;
            let mut chunk = chunk0.clone();
            for off in (0..M).step_by(1024) {
                chunk.set_range(0, 1024);
                bits.apply_mask(&mut chunk, off);
                total += chunk.cardinality();
            }
            black_box(total)
        })
    });
    let a = half_set(M, 4);
    let bb = half_set(M, 5);
    c.bench_function("fixed_bit_set/cardinality_range", |b| {
        b.iter(|| black_box(a.cardinality_range(black_box(1), M - 1)))
    });
    c.bench_function("fixed_bit_set/union_count", |b| {
        b.iter(|| black_box(FixedBitSet::union_count(&a, &bb)))
    });
    c.bench_function("fixed_bit_set/and_not_count", |b| {
        b.iter(|| black_box(FixedBitSet::and_not_count(&a, &bb)))
    });
    let docs: Vec<i32> = (0..M as i32).collect();
    c.bench_function("bits/any_bit_set_of_dense_1m", |b| {
        b.iter(|| {
            let s = AnyBitSet::of(&mut docs.iter().copied(), M as u64, M);
            black_box(s.as_bit_set().cardinality())
        })
    });
}

fn bench_sparse(c: &mut Criterion) {
    const N: usize = 10_000_000;
    let mut r = Lcg(7);
    let ids: Vec<usize> = (0..200_000).map(|_| r.below(N as u64) as usize).collect();
    c.bench_function("sparse_fixed_bit_set/get_and_set_random", |b| {
        b.iter_batched_ref(
            || SparseFixedBitSet::new(N),
            |s| {
                let mut seen = 0usize;
                for &i in &ids {
                    seen += s.get_and_set(i) as usize;
                }
                black_box(seen)
            },
            BatchSize::LargeInput,
        )
    });
    let mut filled = SparseFixedBitSet::new(N);
    for &i in &ids {
        filled.set(i);
    }
    c.bench_function("sparse_fixed_bit_set/get_random", |b| {
        b.iter(|| {
            let mut n = 0usize;
            for &i in &ids {
                n += filled.get(i) as usize;
            }
            black_box(n)
        })
    });
    let mut dense = SparseFixedBitSet::new(1_000_000);
    dense.set_range(0, 900_000);
    c.bench_function("sparse_fixed_bit_set/next_clear_bit_dense", |b| {
        b.iter(|| black_box(dense.next_clear_bit(black_box(0))))
    });
    let mut del = SparseFixedBitSet::new(N);
    for &i in ids.iter().take(10_000) {
        del.set(i);
    }
    let ld = SparseLiveDocs::build(del, N as i32, None).unwrap();
    c.bench_function("live_docs/sparse_live_iter_10m", |b| {
        b.iter(|| black_box(ld.live_docs_iter().0.count()))
    });
}

fn bench_automaton(c: &mut Criterion) {
    let terms = random_terms(100_000, 8, 12, b"abcdefghijn", 11);
    let ing = RegExp::new("a.*ing").unwrap().to_automaton().unwrap();
    let ing = operations::determinize(&ing, i32::MAX).unwrap();
    let run = ByteRunAutomaton::new(&ing, false).unwrap();
    c.bench_function("run_automaton/byte_run_terms", |b| {
        b.iter(|| black_box(terms.iter().filter(|t| run.run(t)).count()))
    });
    let lev = LevenshteinAutomata::new("abcdefghij", true)
        .to_automaton(2)
        .unwrap();
    let compiled = CompiledAutomaton::new(&lev).unwrap();
    let dynr: &dyn ByteRunnable = compiled.get_byte_runnable().unwrap();
    c.bench_function("run_automaton/lev2_dyn_run_terms", |b| {
        b.iter(|| black_box(terms.iter().filter(|t| dynr.run(t)).count()))
    });
    c.bench_function("run_automaton/lev2_step_terms", |b| {
        let r = compiled.run_automaton.as_ref().unwrap();
        b.iter(|| {
            let mut acc = 0usize;
            for t in &terms {
                let mut s = 0;
                for &x in t.iter() {
                    s = r.step(s, i32::from(x));
                    if s == -1 {
                        break;
                    }
                }
                acc += (s != -1 && r.is_accept(s)) as usize;
            }
            black_box(acc)
        })
    });
    let words = RegExp::new("[a-z ]*").unwrap().to_automaton().unwrap();
    let words = operations::determinize(&words, i32::MAX).unwrap();
    let cra = CharacterRunAutomaton::new(&words).unwrap();
    let text: String = random_terms(1, 1 << 20, 1 << 20, b"abcdefgh ", 12)[0]
        .iter()
        .map(|&b| b as char)
        .collect();
    c.bench_function("run_automaton/char_run_text_1mb", |b| {
        b.iter(|| black_box(cra.run(&text)))
    });

    let nfa = RegExp::new("(a|b)*a(a|b){12}")
        .unwrap()
        .to_automaton()
        .unwrap();
    c.bench_function("operations/determinize_regex", |b| {
        b.iter(|| {
            black_box(
                operations::determinize(&nfa, i32::MAX)
                    .unwrap()
                    .get_num_states(),
            )
        })
    });
    c.bench_function("operations/levenshtein_n2_compile", |b| {
        b.iter(|| {
            let a = LevenshteinAutomata::new("abcdefghij", true)
                .to_automaton(2)
                .unwrap();
            black_box(CompiledAutomaton::new(&a).unwrap().sink_state)
        })
    });
    let uni = RegExp::new(".*[\u{100}-\u{ffff}]{3}.*")
        .unwrap()
        .to_automaton()
        .unwrap();
    c.bench_function("operations/utf8_determinize_unicode", |b| {
        b.iter(|| {
            let u = Utf32ToUtf8::new().convert(&uni);
            black_box(
                operations::determinize(&u, i32::MAX)
                    .unwrap()
                    .get_num_states(),
            )
        })
    });
    let big_word = "abcdefghijklmnopqrstuvwxyzabcd";
    let big = LevenshteinAutomata::new(big_word, true)
        .to_automaton(2)
        .unwrap();
    let big_bin = Utf32ToUtf8::new().convert(&big);
    let big_bin = operations::determinize(&big_bin, i32::MAX).unwrap();
    c.bench_function("compiled_automaton/binary_big_dfa", |b| {
        b.iter(|| {
            black_box(
                CompiledAutomaton::with_options(&big_bin, false, true, true)
                    .unwrap()
                    .sink_state,
            )
        })
    });
    let mut sorted = random_terms(200_000, 6, 12, b"abcdefghijklmnop", 13);
    sorted.sort();
    sorted.dedup();
    c.bench_function("automata/make_string_union_200k", |b| {
        b.iter(|| {
            let a = lucene_util::automaton::strings_to_automaton(
                sorted.iter().map(|t| t.as_slice()),
                true,
            )
            .unwrap();
            black_box(a.get_num_states())
        })
    });
    let nfa2 = RegExp::new("(a|ab)*b(a|b){20}")
        .unwrap()
        .to_automaton()
        .unwrap();
    let nfa_bin = Utf32ToUtf8::new().convert(&nfa2);
    let nra = NfaRunAutomaton::with_alphabet(nfa_bin, 0xff);
    let ab_terms = random_terms(100_000, 8, 30, b"ab", 14);
    for t in &ab_terms {
        ByteRunnable::run(&nra, t);
    }
    c.bench_function("nfa_run_automaton/run_terms_warm", |b| {
        b.iter(|| {
            black_box(
                ab_terms
                    .iter()
                    .filter(|t| ByteRunnable::run(&nra, t))
                    .count(),
            )
        })
    });
}

fn bench_bytes_ref_hash(c: &mut Criterion) {
    let terms = random_terms(
        1_000_000,
        4,
        16,
        b"abcdefghijklmnopqrstuvwxyz0123456789",
        21,
    );
    c.bench_function("bytes_ref_hash/add_1m", |b| {
        b.iter(|| {
            let mut h = BytesRefHash::with_capacity(16, 42).unwrap();
            for t in &terms {
                h.add(t).unwrap();
            }
            black_box(h.size())
        })
    });
    let mut h = BytesRefHash::with_capacity(16, 42).unwrap();
    for t in &terms {
        h.add(t).unwrap();
    }
    c.bench_function("bytes_ref_hash/find_1m", |b| {
        b.iter(|| {
            let mut s = 0i64;
            for t in &terms {
                s += i64::from(h.find(t).unwrap());
            }
            black_box(s)
        })
    });
    c.bench_function("bytes_ref_hash/sort_1m", |b| {
        b.iter_batched_ref(
            || h.clone(),
            |h| black_box(h.sort()[0]),
            BatchSize::LargeInput,
        )
    });
}

fn bench_packed(c: &mut Criterion) {
    const VALUES: usize = 65536;
    for bpv in [3u32, 7, 12, 17, 24] {
        let op = BulkOperation::of(Format::Packed, bpv);
        let iterations = VALUES / op.byte_value_count();
        let mut r = Lcg(u64::from(bpv));
        let vals: Vec<i64> = (0..iterations * op.byte_value_count())
            .map(|_| r.below(1 << bpv) as i64)
            .collect();
        let mut bytes = vec![0u8; iterations * op.byte_block_count()];
        op.encode_bytes(&vals, &mut bytes, iterations);
        let mut out = vec![0i64; vals.len()];
        c.bench_function(&format!("bulk_operation/decode_bytes_bpv{bpv}"), |b| {
            b.iter(|| {
                op.decode_bytes(black_box(&bytes), &mut out, iterations);
                black_box(out[out.len() - 1])
            })
        });
        if bpv == 12 || bpv == 17 {
            let mut outi = vec![0i32; vals.len()];
            c.bench_function(
                &format!("bulk_operation/decode_bytes_to_ints_bpv{bpv}"),
                |b| {
                    b.iter(|| {
                        op.decode_bytes_to_ints(black_box(&bytes), &mut outi, iterations);
                        black_box(outi[outi.len() - 1])
                    })
                },
            );
            let li = VALUES / op.long_value_count();
            let mut longs = vec![0u64; li * op.long_block_count()];
            op.encode_longs(&vals[..li * op.long_value_count()], &mut longs, li);
            let mut outl = vec![0i64; li * op.long_value_count()];
            c.bench_function(&format!("bulk_operation/decode_longs_bpv{bpv}"), |b| {
                b.iter(|| {
                    op.decode_longs(black_box(&longs), &mut outl, li);
                    black_box(outl[outl.len() - 1])
                })
            });
        }
    }
    const N: usize = 1 << 20;
    let mut r = Lcg(99);
    let idx: Vec<usize> = (0..100_000).map(|_| r.below(N as u64) as usize).collect();
    for bpv in [3u32, 4, 5, 21] {
        let mut p = Packed64SingleBlock::new(N, bpv);
        for i in 0..N {
            lucene_util::packed::Mutable::set(&mut p, i, (i as i64) & ((1 << bpv) - 1));
        }
        c.bench_function(&format!("packed64_single_block/get_random_bpv{bpv}"), |b| {
            b.iter(|| {
                let mut s = 0i64;
                for &i in &idx {
                    s = s.wrapping_add(p.get(i));
                }
                black_box(s)
            })
        });
    }
}

fn bench_string_helper(c: &mut Criterion) {
    for prefix in [0usize, 8, 24] {
        let mut terms = random_terms(10_000, 4, 12, b"abcdefghij", 31 + prefix as u64);
        let p: Vec<u8> = (0..prefix).map(|i| b'a' + (i % 26) as u8).collect();
        for t in &mut terms {
            let mut v = p.clone();
            v.extend_from_slice(t);
            *t = v;
        }
        terms.sort();
        terms.dedup();
        c.bench_function(
            &format!("string_helper/bytes_difference_prefix{prefix}"),
            |b| {
                b.iter(|| {
                    let mut s = 0usize;
                    for w in terms.windows(2) {
                        s += bytes_difference(&w[0], &w[1]).unwrap_or(0);
                    }
                    black_box(s)
                })
            },
        );
    }
}

criterion_group!(
    benches,
    bench_fixed_bit_set,
    bench_sparse,
    bench_automaton,
    bench_bytes_ref_hash,
    bench_packed,
    bench_string_helper
);
criterion_main!(benches);
