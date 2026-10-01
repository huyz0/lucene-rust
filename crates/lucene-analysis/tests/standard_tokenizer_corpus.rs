//! `StandardTokenizer` and `StandardAnalyzer` against Lucene 10.5.0 over a
//! few megabytes of real multilingual text, token for token:
//! `fixtures/src/GenStandardTokenizerCorpus.java`.
//!
//! The conformance half (`standard_tokenizer_fixtures.rs`) runs Unicode's
//! word-break and emoji test data; this runs prose -- 4,000 lines of
//! `lucene-test-framework`'s `europarl.lines.txt.gz` (twenty-one European
//! languages, Latin, Greek and Cyrillic script, numbers, abbreviations, lines
//! cut mid-character) and every stopword list `lucene-analysis-common` ships
//! (Thai, Devanagari, Bengali, Tamil, Telugu, Arabic, Persian, Armenian, ...)
//! -- about 4.9 MB and 1.3 million tokens per configuration.
//!
//! The text is not committed: it is read out of the Lucene jars themselves
//! (zip entries, and the corpus's own gzip member), found through
//! `$LUCENE_TEST_FRAMEWORK_JAR`/`$LUCENE_ANALYSIS_COMMON_JAR`, a
//! `$JARS`/`$LUCENE_JARS` directory (the container), `fixtures/.jars`, or the
//! local Gradle cache. The fixture is digests: per 100-line chunk the text's
//! (so a different jar or a different UTF-8 decoding is reported as that, not
//! as a tokenizer bug), the tokens', and one per line. A mismatch names the
//! line, prints this port's tokens for it, and the command that prints
//! Lucene's. Without the jars the test says so and passes;
//! `scripts/verify-write-path.sh` hands it the jars and fails on a skip.
#![allow(clippy::arithmetic_side_effects)]

use std::path::{Path, PathBuf};

use lucene_analysis::reader::StrReader;
use lucene_analysis::token_stream::{TokenStream, Tokenizer};
use lucene_analysis::{Analyzer, StandardAnalyzer, StandardTokenizer};

fn manifest() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/standard_tokenizer_corpus/manifest.tsv"
    ))
    .expect("run scripts/gen-fixtures.sh --only GenStandardTokenizerCorpus")
}

/// The Lucene 10.5.0 jar of `module`, wherever this machine keeps it.
fn find_jar(module: &str) -> Option<PathBuf> {
    let env = format!("{}_JAR", module.to_uppercase().replace('-', "_"));
    if let Some(jar) = std::env::var_os(env) {
        return Some(PathBuf::from(jar)).filter(|p| p.is_file());
    }
    let file = format!("{module}-10.5.0.jar");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut dirs: Vec<PathBuf> = ["JARS", "LUCENE_JARS"]
        .iter()
        .filter_map(std::env::var_os)
        .map(PathBuf::from)
        .collect();
    dirs.push(repo.join("fixtures/.jars"));
    if let Some(jar) = dirs.iter().map(|d| d.join(&file)).find(|p| p.is_file()) {
        return Some(jar);
    }
    let gradle = PathBuf::from(std::env::var_os("HOME")?)
        .join(".gradle/caches/modules-2/files-2.1/org.apache.lucene")
        .join(module)
        .join("10.5.0");
    std::fs::read_dir(gradle)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join(&file))
        .find(|p| p.is_file())
}

fn u16_at(b: &[u8], at: usize) -> usize {
    usize::from(u16::from_le_bytes([b[at], b[at + 1]]))
}

fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
}

/// One entry of a zip archive, inflated: the central directory found from
/// its end record, the entry's local header skipped, `stored` or `deflate`.
fn zip_entry(zip: &[u8], name: &str) -> Vec<u8> {
    let eocd = (0..=zip.len() - 22)
        .rev()
        .find(|&i| zip[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .expect("zip end of central directory");
    let entries = u16_at(zip, eocd + 10);
    let mut at = u32_at(zip, eocd + 16);
    for _ in 0..entries {
        assert_eq!(zip[at..at + 4], [0x50, 0x4b, 0x01, 0x02], "central header");
        let method = u16_at(zip, at + 10);
        let compressed = u32_at(zip, at + 20);
        let (name_len, extra_len, comment_len) = (
            u16_at(zip, at + 28),
            u16_at(zip, at + 30),
            u16_at(zip, at + 32),
        );
        let local = u32_at(zip, at + 42);
        let entry_name = &zip[at + 46..at + 46 + name_len];
        if entry_name == name.as_bytes() {
            let data = local + 30 + u16_at(zip, local + 26) + u16_at(zip, local + 28);
            let raw = &zip[data..data + compressed];
            return match method {
                0 => raw.to_vec(),
                8 => miniz_oxide::inflate::decompress_to_vec(raw).expect("inflate zip entry"),
                m => panic!("zip method {m}"),
            };
        }
        at += 46 + name_len + extra_len + comment_len;
    }
    panic!("{name} is not in the jar");
}

/// A gzip file's payload, as `GZIPInputStream` reads it: every member in
/// turn (the corpus is several concatenated members), each one's RFC 1952
/// header (with its optional fields) skipped, its raw DEFLATE stream
/// inflated, its 8-byte trailer stepped over.
fn gunzip(gz: &[u8]) -> Vec<u8> {
    use miniz_oxide::inflate::stream::{inflate, InflateState};
    use miniz_oxide::{DataFormat, MZFlush, MZStatus};
    let mut out = Vec::new();
    let mut member = 0;
    while gz.len() - member >= 18 && gz[member..member + 2] == [0x1f, 0x8b] {
        assert_eq!(gz[member + 2], 8, "gzip method");
        let flags = gz[member + 3];
        let mut at = member + 10;
        if flags & 4 != 0 {
            at += 2 + u16_at(gz, at);
        }
        for bit in [8u8, 16] {
            if flags & bit != 0 {
                while gz[at] != 0 {
                    at += 1;
                }
                at += 1;
            }
        }
        if flags & 2 != 0 {
            at += 2;
        }
        let mut state = InflateState::new_boxed(DataFormat::Raw);
        let mut buf = vec![0u8; 1 << 16];
        let mut input = &gz[at..];
        loop {
            let r = inflate(&mut state, input, &mut buf, MZFlush::None);
            input = &input[r.bytes_consumed..];
            out.extend_from_slice(&buf[..r.bytes_written]);
            match r.status {
                Ok(MZStatus::StreamEnd) => break,
                Ok(_) => {}
                Err(e) => panic!("inflate gzip member: {e:?}"),
            }
        }
        member = gz.len() - input.len() + 8;
    }
    out
}

/// `BufferedReader.readLine` over an `InputStreamReader(UTF-8)`: malformed
/// UTF-8 replaced, lines split at `\n`, `\r` and `\r\n`, the first `limit`.
fn lines(bytes: &[u8], limit: usize) -> Vec<String> {
    let text = String::from_utf8_lossy(bytes);
    let mut out = Vec::new();
    let mut rest: &str = &text;
    while !rest.is_empty() && out.len() < limit {
        let end = rest.find(['\n', '\r']).unwrap_or(rest.len());
        out.push(rest[..end].to_string());
        rest = &rest[end..];
        if rest.starts_with("\r\n") {
            rest = &rest[2..];
        } else if !rest.is_empty() {
            rest = &rest[1..];
        }
    }
    out
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `GenStandardTokenizerCorpus.tokens`: the line's tokens in digest form,
/// and how many.
fn tokens(ts: &mut dyn TokenStream) -> (String, usize) {
    let mut s = String::new();
    let mut n = 0;
    ts.reset().unwrap();
    while ts.increment_token().unwrap() {
        let a = ts.attributes();
        s.push_str(&String::from_utf8_lossy(a.term_bytes()));
        s.push('\0');
        s.push_str(&format!(
            "{},{},{},{}\n",
            a.start_offset(),
            a.end_offset(),
            a.position_increment(),
            a.token_type()
        ));
        n += 1;
    }
    ts.end().unwrap();
    let a = ts.attributes();
    s.push_str(&format!(
        "end:{},{}\n",
        a.end_offset(),
        a.position_increment()
    ));
    ts.close().unwrap();
    (s, n)
}

struct Source {
    lines: Vec<String>,
}

#[test]
fn standard_tokenizer_matches_lucene_over_a_multilingual_corpus() {
    let manifest = manifest();
    let mut jars: std::collections::HashMap<String, Vec<u8>> = Default::default();
    let mut sources: std::collections::HashMap<String, Source> = Default::default();
    for line in manifest.lines().filter(|l| l.starts_with("S\t")) {
        let f: Vec<&str> = line.split('\t').collect();
        let (id, module, entry, gz, count) = (f[1], f[2], f[3], f[4] == "1", f[5]);
        if !jars.contains_key(module) {
            let Some(path) = find_jar(module) else {
                eprintln!(
                    "standard_tokenizer_corpus: skipped -- no {module}-10.5.0.jar \
                     (set {}_JAR, $LUCENE_JARS, or run scripts/gen-fixtures.sh)",
                    module.to_uppercase().replace('-', "_")
                );
                return;
            };
            jars.insert(module.to_string(), std::fs::read(path).unwrap());
        }
        let mut bytes = zip_entry(&jars[module], entry);
        if gz {
            bytes = gunzip(&bytes);
        }
        let count: usize = count.parse().unwrap();
        let lines = lines(&bytes, count);
        assert_eq!(lines.len(), count, "{id}: line count");
        sources.insert(id.to_string(), Source { lines });
    }

    let mut tokenizer = StandardTokenizer::new();
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let (mut chunks, mut total_tokens, mut failures) = (0usize, 0usize, Vec::new());
    for line in manifest.lines().filter(|l| l.starts_with("C\t")) {
        let f: Vec<&str> = line.split('\t').collect();
        let (id, first, n, text_digest, config, count, digest, per_line) = (
            f[1],
            f[2].parse::<usize>().unwrap(),
            f[3].parse::<usize>().unwrap(),
            u64::from_str_radix(f[4], 16).unwrap(),
            f[5],
            f[6].parse::<usize>().unwrap(),
            u64::from_str_radix(f[7], 16).unwrap(),
            f[8].split(',').collect::<Vec<_>>(),
        );
        let src = &sources[id];
        let text = &src.lines[first..first + n];
        let got_text = text
            .iter()
            .fold(FNV_OFFSET, |h, l| fnv(fnv(h, l.as_bytes()), b"\n"));
        assert_eq!(
            got_text,
            text_digest,
            "{id} lines {first}..{}: the text read is not the text Lucene read \
             (a different jar, or a different UTF-8 decoding)",
            first + n
        );
        let (mut got, mut got_count) = (FNV_OFFSET, 0);
        for (k, t) in text.iter().enumerate() {
            let (toks, c) = match config {
                "tok" => {
                    tokenizer
                        .set_reader(Box::new(StrReader::new(t.as_str())))
                        .unwrap();
                    tokens(&mut tokenizer)
                }
                _ => {
                    let mut ts = analyzer.token_stream("body", t).unwrap();
                    tokens(&mut ts)
                }
            };
            got = fnv(got, toks.as_bytes());
            got_count += c;
            let line_digest = format!("{:08x}", fnv(FNV_OFFSET, toks.as_bytes()) as u32);
            if line_digest != per_line[k] && failures.len() < 5 {
                failures.push(format!(
                    "{config} {id} line {}: tokens differ from Lucene's; ours, one per line \
                     (term|start,end,posInc,type):\n{}\n  Lucene's: java -cp <lucene-core, \
                     lucene-analysis-common, fixtures classes> GenStandardTokenizerCorpus \
                     --dump {id} {} {config}  (LUCENE_TEST_FRAMEWORK_JAR set)\n  text: {:?}",
                    first + k,
                    toks.replace('\0', "|"),
                    first + k,
                    t
                ));
            }
        }
        if got != digest || got_count != count {
            failures.push(format!(
                "{config} {id} lines {first}..{}: {got_count} tokens (Lucene {count}), \
                 digest {got:016x} (Lucene {digest:016x})",
                first + n
            ));
        }
        chunks += 1;
        total_tokens += got_count;
    }
    eprintln!(
        "standard_tokenizer_corpus: {chunks} chunks, {total_tokens} tokens, {} sources",
        sources.len()
    );
    assert!(
        chunks >= 100 && total_tokens >= 1_000_000,
        "{chunks} chunks"
    );
    assert!(
        failures.is_empty(),
        "{} differences:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
