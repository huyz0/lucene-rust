//! M13 T13.3e against Lucene 10.5.0 (`fixtures/src/GenSpellDistances.java`):
//! every string distance over every ordered pair of `words.tsv`, bit for
//! bit; `SuggestWord.toString`; the two comparators as sort orders; the
//! queue's pop order at several sizes; and `PlainTextDictionary`'s entries.

use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;

use lucene_search::spell::distance::{
    JaroWinklerDistance, LevenshteinDistance, LuceneLevenshteinDistance, NGramDistance,
    StringDistance,
};
use lucene_search::spell::plain_text::PlainTextDictionary;
use lucene_search::spell::suggest_word::{
    compare_by_frequency, compare_by_score, frequency_comparator, score_comparator, SuggestWord,
    SuggestWordQueue,
};
use lucene_search::spell::Dictionary;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/spell_distances")
}

fn read(name: &str) -> String {
    fs::read_to_string(dir().join(name)).unwrap()
}

fn words() -> Vec<String> {
    let text = read("words.tsv");
    let mut words: Vec<String> = text.split('\n').map(str::to_string).collect();
    // the file ends with a newline
    assert_eq!(words.pop().as_deref(), Some(""));
    words
}

fn distance(name: &str) -> Box<dyn StringDistance> {
    match name {
        "levenshtein" => Box::new(LevenshteinDistance::new()),
        "lucene_levenshtein" => Box::new(LuceneLevenshteinDistance::new()),
        "jarowinkler" => Box::new(JaroWinklerDistance::new()),
        "jarowinkler_0" | "jarowinkler_1.1" => {
            let mut jw = JaroWinklerDistance::new();
            jw.set_threshold(if name.ends_with("_0") { 0.0 } else { 1.1 });
            Box::new(jw)
        }
        _ => {
            let n = name.strip_prefix("ngram_").unwrap().parse().unwrap();
            Box::new(NGramDistance::with_size(n))
        }
    }
}

#[test]
fn every_distance_matches_lucene_bit_for_bit() {
    let words = words();
    let mut rows = 0;
    let mut nans = 0;
    for line in read("distances.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let d = distance(f[0]);
        let (i, j): (usize, usize) = (f[1].parse().unwrap(), f[2].parse().unwrap());
        let want = f32::from_bits(u32::from_str_radix(f[3], 16).unwrap());
        let got = d.get_distance(&words[i], &words[j]);
        if want.is_nan() {
            nans += 1;
            assert!(
                got.is_nan(),
                "{} {:?} {:?}: {got}",
                f[0],
                words[i],
                words[j]
            );
        } else {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{} {:?} {:?}: {got} != {want}",
                f[0],
                words[i],
                words[j]
            );
        }
        rows += 1;
    }
    assert_eq!(rows, 10 * words.len() * words.len());
    // ngram(0) of two non-empty strings is Java's 0f / 0
    assert!(nans > 0);
}

fn suggest_words() -> Vec<SuggestWord> {
    read("suggest_words.tsv")
        .lines()
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            let w = SuggestWord::new(
                f[0],
                f32::from_bits(u32::from_str_radix(f[1], 16).unwrap()),
                f[2].parse().unwrap(),
            );
            assert_eq!(w.to_string(), f[3]);
            w
        })
        .collect()
}

fn indexes(s: &str) -> Vec<usize> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',').map(|x| x.parse().unwrap()).collect()
}

/// `List.sort` below `TimSort.MIN_MERGE` (32): the leading run made
/// ascending, then a binary insertion sort. The score comparators order NaN
/// neither before nor after anything, so they are not transitive and the
/// result depends on the algorithm, not only on the comparator.
fn java_small_sort<T: Copy>(a: &mut [T], c: impl Fn(&T, &T) -> Ordering) {
    let hi = a.len();
    assert!(hi < 32);
    if hi < 2 {
        return;
    }
    // countRunAndMakeAscending
    let mut run_hi = 2;
    if c(&a[1], &a[0]) == Ordering::Less {
        while run_hi < hi && c(&a[run_hi], &a[run_hi - 1]) == Ordering::Less {
            run_hi += 1;
        }
        a[..run_hi].reverse();
    } else {
        while run_hi < hi && c(&a[run_hi], &a[run_hi - 1]) != Ordering::Less {
            run_hi += 1;
        }
    }
    // binarySort
    for start in run_hi..hi {
        let pivot = a[start];
        let (mut left, mut right) = (0, start);
        while left < right {
            let mid = (left + right) / 2;
            if c(&pivot, &a[mid]) == Ordering::Less {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        a.copy_within(left..start, left + 1);
        a[left] = pivot;
    }
}

#[test]
fn the_comparators_sort_as_lucene_does() {
    let words = suggest_words();
    let mut rows = 0;
    for line in read("orders.tsv").lines() {
        let (name, order) = line.split_once('\t').unwrap();
        let cmp = match name.trim_end_matches("_reverse") {
            "score" => compare_by_score,
            _ => compare_by_frequency,
        };
        let reverse = name.ends_with("_reverse");
        let mut idx: Vec<usize> = (0..words.len()).collect();
        java_small_sort(&mut idx, |&x, &y| {
            let o = cmp(&words[x], &words[y]);
            if reverse {
                o.reverse()
            } else {
                o
            }
        });
        assert_eq!(idx, indexes(order), "{name}");
        rows += 1;
    }
    assert_eq!(rows, 4);
}

#[test]
fn the_queue_pops_in_lucenes_order() {
    let words = suggest_words();
    let mut rows = 0;
    for line in read("queues.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let size: usize = f[1].parse().unwrap();
        let mut q = match f[0] {
            "score" if size % 2 == 1 => SuggestWordQueue::new(size),
            "score" => SuggestWordQueue::with_comparator(size, score_comparator()),
            _ => SuggestWordQueue::with_comparator(size, frequency_comparator()),
        };
        for w in &words {
            q.insert_with_overflow(w.clone());
        }
        let mut popped = Vec::new();
        while let Some(w) = q.pop() {
            popped.push(w.to_string());
        }
        // by content: the two equal "apple" words are interchangeable
        let want: Vec<String> = indexes(f[2])
            .into_iter()
            .map(|i| words[i].to_string())
            .collect();
        assert_eq!(popped, want, "{} {size}", f[0]);
        rows += 1;
    }
    assert_eq!(rows, 14);
}

#[test]
fn plain_text_dictionaries_read_lines_as_lucene_does() {
    let texts = [
        ("empty", ""),
        ("one", "word"),
        ("lf", "alpha\nbeta\ngamma\n"),
        ("crlf", "alpha\r\nbeta\r\n\r\ngamma"),
        ("cr", "alpha\rbeta\r\rgamma\r"),
        ("blank_lines", "\n\n\nx\n\n"),
        ("unicode", "café\n日本語\n😀\n"),
        ("tabs", "a\tb\tc\nd e\n"),
    ];
    let lines: Vec<String> = read("plain_text.tsv").lines().map(str::to_string).collect();
    assert_eq!(lines.len(), texts.len());
    for ((name, text), line) in texts.iter().zip(&lines) {
        for dict in [
            PlainTextDictionary::from_text(*text),
            PlainTextDictionary::from_utf8(text.as_bytes()).unwrap(),
        ] {
            // twice: each call reads the text from the start
            for _ in 0..2 {
                let mut it = dict.get_entry_iterator().unwrap();
                let mut got = format!("{name}\t{}\t{}", it.has_payloads(), it.has_contexts());
                while let Some(b) = it.next() {
                    let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
                    got.push_str(&format!("\t{hex}:{}", it.weight()));
                }
                assert_eq!(&got, line);
            }
        }
    }
    assert!(PlainTextDictionary::from_utf8(&[0xff]).is_err());
}
