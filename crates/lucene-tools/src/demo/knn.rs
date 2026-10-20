//! `org.apache.lucene.demo.knn`: the demo's toy word-vector dictionary and
//! the embeddings it computes for a text.
//!
//! A dictionary is built from a GloVe-style text file -- one word per line,
//! followed by its vector's components, separated by single spaces -- into
//! two files of an index directory: `<name>.fst`, an FST from each word to
//! its ordinal, and `<name>.bin`, every vector L2-normalized as
//! little-endian floats, then the dimension as a little-endian `int`. The
//! words must come in sorted (UTF-8 byte) order, as the FST compiler takes
//! them. As in Java, the first line's vector is written to `.bin` but its
//! word is not added to the FST: ordinals start at 1, at the second line.

use std::path::Path;

use lucene_analysis::Analyzer;
use lucene_codecs::fst::{InputType, PositiveIntOutputs};
use lucene_codecs::fst_compiler::FstCompilerBuilder;
use lucene_codecs::fst_util::{self, TypedFst};
use lucene_store::directory::Directory;
use lucene_store::{DataOutput, FsIndexOutput};
use lucene_util::vector_util;

use crate::cli::{exception, Error};

/// A word-vector dictionary written by [`KnnVectorDict::build`]
/// (`KnnVectorDict`).
pub struct KnnVectorDict {
    fst: TypedFst<PositiveIntOutputs>,
    fst_len: u64,
    vectors: Vec<u8>,
    dimension: usize,
}

impl std::fmt::Debug for KnnVectorDict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnnVectorDict")
            .field("dimension", &self.dimension)
            .field("vectors", &self.vectors.len())
            .finish_non_exhaustive()
    }
}

impl KnnVectorDict {
    /// `new KnnVectorDict(directory, dictName)`: reads `<name>.fst` and
    /// `<name>.bin`.
    ///
    /// # Errors
    /// A missing or undecodable file, and Java's `IllegalStateException` for
    /// a `.bin` whose length is not a whole number of vectors.
    pub fn open(directory: &dyn Directory, dict_name: &str) -> Result<Self, Error> {
        let fst_bytes = directory.open(&format!("{dict_name}.fst"))?;
        let fst = TypedFst::<PositiveIntOutputs>::read(&fst_bytes)?;
        let vectors = directory.open(&format!("{dict_name}.bin"))?.to_vec();
        let size = vectors.len();
        let Some(tail) = size.checked_sub(4).map(|at| &vectors[at..]) else {
            return Err(exception(format!(
                "java.io.EOFException: read past EOF: {dict_name}.bin"
            )));
        };
        let dimension = i32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
        let vector_bytes = i64::from(dimension).saturating_mul(4);
        let body = (size - 4) as i64;
        if vector_bytes == 0 || body % vector_bytes != 0 {
            return Err(exception(format!(
                "java.lang.IllegalStateException: vector file size {size} is not consonant with \
                 the vector dimension {dimension}"
            )));
        }
        Ok(KnnVectorDict {
            fst,
            fst_len: fst_bytes.len() as u64,
            vectors,
            dimension: usize::try_from(dimension).unwrap_or(0),
        })
    }

    /// `get(token, output)`: `token`'s vector as little-endian float bytes,
    /// zeros for a word the dictionary does not hold.
    ///
    /// # Errors
    /// Java's `IllegalArgumentException` for an `output` that is not
    /// `dimension * 4` bytes long, and an FST that does not decode.
    pub fn get(&self, token: &[u8], output: &mut [u8]) -> Result<(), Error> {
        let want = self.dimension.saturating_mul(4);
        if output.len() != want {
            return Err(exception(format!(
                "java.lang.IllegalArgumentException: the output array must be of length {want}, \
                 got {}",
                output.len()
            )));
        }
        let input: Vec<i32> = token.iter().map(|&b| i32::from(b)).collect();
        match fst_util::get(&self.fst, &input)? {
            None => output.fill(0),
            Some(ord) => {
                let at = usize::try_from(ord)
                    .ok()
                    .and_then(|o| o.checked_mul(want))
                    .and_then(|at| self.vectors.get(at..at.checked_add(want)?))
                    .ok_or_else(|| {
                        exception(format!(
                            "java.io.EOFException: read past EOF: ordinal {ord}"
                        ))
                    })?;
                output.copy_from_slice(at);
            }
        }
        Ok(())
    }

    /// `getDimension()`.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// `ramBytesUsed()`: what the two files hold. Java adds the FST
    /// object's own estimated overhead, a few dozen bytes; the demo only
    /// compares the sum against a megabyte.
    pub fn ram_bytes_used(&self) -> u64 {
        self.fst_len.saturating_add(self.vectors.len() as u64)
    }

    /// `KnnVectorDict.build(gloveInput, directory, dictName)`.
    ///
    /// # Errors
    /// Reading `glove_input` or writing the two files, and Java's exceptions
    /// for a line with another number of fields than the first
    /// (`IllegalStateException`), a component that is not a float
    /// (`NumberFormatException`), a zero vector (`IllegalArgumentException`)
    /// and words out of order.
    pub fn build(
        glove_input: &Path,
        directory: &dyn Directory,
        dict_name: &str,
    ) -> Result<(), Error> {
        let text = std::fs::read(glove_input)?;
        // `Files.newBufferedReader`'s `MalformedInputException`, an `IOException`
        let text = String::from_utf8(text).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Input length = 1")
        })?;
        // `createOutput` opens with `CREATE_NEW`, `.bin` then `.fst`, before
        // a line is read: an existing file fails the build, and a build that
        // fails later leaves the files it created behind.
        let mut bin_out = create_new(directory, &format!("{dict_name}.bin"))?;
        let mut fst_out = match create_new(directory, &format!("{dict_name}.fst")) {
            Ok(out) => out,
            Err(e) => {
                bin_out.close()?;
                return Err(e);
            }
        };
        let compiled = Self::compile(&text);
        let (fst, bin) = match compiled {
            Ok(files) => files,
            Err(e) => {
                fst_out.close()?;
                bin_out.close()?;
                return Err(e);
            }
        };
        fst_out.write_bytes(&fst);
        bin_out.write_bytes(&bin);
        fst_out.close()?;
        bin_out.close()?;
        Ok(())
    }

    /// `Builder.build`'s loop over the text: the `.fst` and `.bin` bytes.
    fn compile(text: &str) -> Result<(Vec<u8>, Vec<u8>), Error> {
        let mut lines = java_lines(text);
        let mut bin = Vec::new();
        let mut compiler = FstCompilerBuilder::new(InputType::Byte1).build::<PositiveIntOutputs>();
        let mut num_fields = 0usize;
        if let Some(first) = lines.next() {
            let fields = split_fields(first);
            num_fields = fields.len();
            write_vector(&fields, &mut bin)?;
            let mut ordinal: i64 = 1;
            for line in lines {
                let fields = split_fields(line);
                if fields.len() != num_fields {
                    return Err(exception(format!(
                        "java.lang.IllegalStateException: different field count at line \
                         {ordinal} got {} when expecting {num_fields}",
                        fields.len()
                    )));
                }
                let input: Vec<i32> = fields[0].bytes().map(i32::from).collect();
                compiler
                    .add(&input, ordinal)
                    .map_err(|e| exception(format!("java.lang.IllegalArgumentException: {e}")))?;
                ordinal = ordinal.saturating_add(1);
                write_vector(&fields, &mut bin)?;
            }
        }
        // Java hands `compile()`'s null (no word added) to `FST.fromFSTReader`.
        let fst = compiler
            .compile()
            .ok_or_else(|| exception("java.lang.NullPointerException"))?;
        let dimension = i32::try_from(num_fields.saturating_sub(1)).unwrap_or(i32::MAX);
        bin.extend_from_slice(&dimension.to_le_bytes());
        Ok((
            TypedFst::<PositiveIntOutputs>::from_compiled(fst).save(),
            bin,
        ))
    }
}

/// `FileAlreadyExistsException`: `createOutput` over a file that exists.
/// The message is the file's name; [`crate::demo::index_files`] prints the
/// directory's real path before it, as Java's message reads.
#[derive(Debug)]
pub struct FileAlreadyExists(pub String);

impl std::fmt::Display for FileAlreadyExists {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FileAlreadyExists {}

/// `directory.createOutput(name)` under `CREATE_NEW`.
fn create_new(directory: &dyn Directory, name: &str) -> Result<FsIndexOutput, Error> {
    if directory.list_all()?.iter().any(|n| n == name) {
        return Err(Box::new(FileAlreadyExists(name.to_string())));
    }
    Ok(directory.create_output(name)?)
}

/// `BufferedReader.readLine` over `text`: lines end at `\n`, `\r` or
/// `\r\n`, and a final empty line is not one.
fn java_lines(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let end = rest.find(['\n', '\r']).unwrap_or(rest.len());
        let line = &rest[..end];
        let mut next = end;
        if rest[end..].starts_with("\r\n") {
            next += 2;
        } else if end < rest.len() {
            next += 1;
        }
        rest = &rest[next..];
        Some(line)
    })
}

/// `Pattern.compile(" ").split(line, 0)`: split at every space, trailing
/// empty strings dropped (an empty line is one empty field).
fn split_fields(line: &str) -> Vec<&str> {
    let mut fields: Vec<&str> = line.split(' ').collect();
    while fields.len() > 1 && fields.last() == Some(&"") {
        fields.pop();
    }
    fields
}

/// `Float.parseFloat(s)`: decimal or exponent notation, surrounding
/// whitespace and a trailing `f`/`d` type suffix allowed.
fn parse_float(s: &str) -> Result<f32, Error> {
    let trimmed = s.trim_matches(|c: char| c <= ' ');
    let t = trimmed
        .strip_suffix(['f', 'F', 'd', 'D'])
        .filter(|t| !t.ends_with(['e', 'E']) && !t.is_empty())
        .unwrap_or(trimmed);
    let parsed = match t {
        "NaN" | "+NaN" | "-NaN" => Some(f32::NAN),
        "Infinity" | "+Infinity" => Some(f32::INFINITY),
        "-Infinity" => Some(f32::NEG_INFINITY),
        // Rust also reads `inf`/`nan` words Java refuses.
        _ if t
            .chars()
            .any(|c| c.is_ascii_alphabetic() && !matches!(c, 'e' | 'E')) =>
        {
            None
        }
        _ => t.parse::<f32>().ok(),
    };
    parsed.ok_or_else(|| {
        exception(if t.is_empty() {
            "java.lang.NumberFormatException: empty String".to_string()
        } else {
            // `FloatingDecimal` quotes the trimmed string
            format!("java.lang.NumberFormatException: For input string: \"{trimmed}\"")
        })
    })
}

/// `Builder.writeVector`: the line's components, L2-normalized, as
/// little-endian floats.
fn write_vector(fields: &[&str], out: &mut Vec<u8>) -> Result<(), Error> {
    let mut v = fields[1..]
        .iter()
        .map(|f| parse_float(f))
        .collect::<Result<Vec<f32>, Error>>()?;
    vector_util::l2normalize(&mut v, true)
        .map_err(|e| exception(format!("java.lang.IllegalArgumentException: {e}")))?;
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    Ok(())
}

/// `KnnVectorDictFilter`: sums the dictionary vector of every token that
/// passes through it, L2-normalized at `end()`. Here it is fed the tokens
/// rather than wrapping a token stream; [`DemoEmbeddings`] does the
/// wrapping.
#[derive(Debug)]
pub struct KnnVectorDictFilter<'a> {
    dict: &'a KnnVectorDict,
    scratch_bytes: Vec<u8>,
    result: Vec<f32>,
}

impl<'a> KnnVectorDictFilter<'a> {
    /// `new KnnVectorDictFilter(input, dict)`.
    pub fn new(dict: &'a KnnVectorDict) -> Self {
        KnnVectorDictFilter {
            dict,
            scratch_bytes: vec![0; dict.dimension().saturating_mul(4)],
            result: vec![0.0; dict.dimension()],
        }
    }

    /// `reset()`: back to the zero vector.
    pub fn reset(&mut self) {
        self.result.fill(0.0);
    }

    /// `incrementToken()` for one token: adds its vector (`VectorUtil.add`).
    ///
    /// # Errors
    /// The dictionary's.
    pub fn token(&mut self, term: &str) -> Result<(), Error> {
        self.dict.get(term.as_bytes(), &mut self.scratch_bytes)?;
        for (r, b) in self
            .result
            .iter_mut()
            .zip(self.scratch_bytes.chunks_exact(4))
        {
            *r += f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        Ok(())
    }

    /// `end()`: L2-normalizes the sum (a zero sum stays zero).
    pub fn end(&mut self) {
        // `l2normalize(result, false)` cannot fail.
        let _ = vector_util::l2normalize(&mut self.result, false);
    }

    /// `getResult()`.
    pub fn result(&self) -> &[f32] {
        &self.result
    }
}

/// `DemoEmbeddings`: a text's embedding, the normalized sum of its words'
/// dictionary vectors -- `StandardTokenizer`, `LowerCaseFilter`, then
/// [`KnnVectorDictFilter`].
pub struct DemoEmbeddings<'a> {
    dict: &'a KnnVectorDict,
    analyzer: Analyzer,
}

impl std::fmt::Debug for DemoEmbeddings<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DemoEmbeddings")
            .field("dict", self.dict)
            .finish_non_exhaustive()
    }
}

impl<'a> DemoEmbeddings<'a> {
    /// `new DemoEmbeddings(vectorDict)`.
    pub fn new(dict: &'a KnnVectorDict) -> Self {
        DemoEmbeddings {
            dict,
            analyzer: Analyzer::standard(None),
        }
    }

    /// `computeEmbedding(input)`.
    ///
    /// # Errors
    /// The analysis's and the dictionary's.
    pub fn compute_embedding(&self, input: &str) -> Result<Vec<f32>, Error> {
        let mut filter = KnnVectorDictFilter::new(self.dict);
        filter.reset();
        let mut failed = None;
        self.analyzer.try_for_each_token(input, |term, _, _, _| {
            if failed.is_none() {
                failed = filter.token(term).err();
            }
        })?;
        if let Some(e) = failed {
            return Err(e);
        }
        filter.end();
        Ok(filter.result().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::directory::FsDirectory;
    use lucene_util::test_support::TempDir;

    fn dict(text: &str) -> (TempDir, Result<KnnVectorDict, Error>) {
        let tmp = TempDir::new("tools-knn-dict");
        let glove = tmp.path().join("glove.txt");
        std::fs::write(&glove, text).unwrap();
        let dir = FsDirectory::open(tmp.path().join("dict"));
        std::fs::create_dir_all(tmp.path().join("dict")).unwrap();
        let built =
            KnnVectorDict::build(&glove, &dir, "d").and_then(|()| KnnVectorDict::open(&dir, "d"));
        (tmp, built)
    }

    #[test]
    fn the_first_line_is_written_but_not_looked_up() {
        let (_tmp, d) = dict("first 1 0\nalpha 0 2\r\nbeta 3 4 \n");
        let d = d.unwrap();
        assert_eq!(d.dimension(), 2);
        let mut out = vec![0u8; 8];
        d.get(b"alpha", &mut out).unwrap();
        assert_eq!(
            out,
            [0.0f32, 1.0]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>()
        );
        d.get(b"first", &mut out).unwrap();
        assert_eq!(out, vec![0; 8]);
        d.get(b"beta", &mut out).unwrap();
        assert_eq!(
            out,
            [0.6f32, 0.8]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert!(d.ram_bytes_used() > 3 * 8);
        let e = d.get(b"beta", &mut [0u8; 4]).unwrap_err().to_string();
        assert_eq!(
            e,
            "java.lang.IllegalArgumentException: the output array must be of length 8, got 4"
        );
    }

    #[test]
    fn javas_build_errors() {
        let err = |text: &str| dict(text).1.unwrap_err().to_string();
        assert_eq!(
            err("a 1 0\nb 1\n"),
            "java.lang.IllegalStateException: different field count at line 1 got 2 when expecting 3"
        );
        assert_eq!(
            err("a 1 0\nb 1 x\n"),
            "java.lang.NumberFormatException: For input string: \"x\""
        );
        assert_eq!(
            err("a 1 0\nb 0 0\n"),
            "java.lang.IllegalArgumentException: Cannot normalize a zero-length vector"
        );
        assert!(err("a 1 0\nc 1 0\nb 1 0\n").starts_with("java.lang.IllegalArgumentException"));
        assert_eq!(err("a 1 0\n"), "java.lang.NullPointerException");
        assert_eq!(err(""), "java.lang.NullPointerException");
        assert!(err("a \u{0} 1\n").starts_with("java.lang.NumberFormatException"));
        // not UTF-8: an `IOException`, which `IndexFiles` catches
        let tmp = TempDir::new("tools-knn-utf8");
        let glove = tmp.path().join("glove.txt");
        std::fs::write(&glove, b"a 1 0\n\xff 1 0\n").unwrap();
        let dir = FsDirectory::open(tmp.path());
        let e = KnnVectorDict::build(&glove, &dir, "d").unwrap_err();
        assert!(e.is::<std::io::Error>());
        assert_eq!(e.to_string(), "Input length = 1");
    }

    #[test]
    fn outputs_are_created_new_before_the_text_is_read() {
        let tmp = TempDir::new("tools-knn-create-new");
        let glove = tmp.path().join("glove.txt");
        std::fs::write(&glove, "a 1 0\nb 1\n").unwrap();
        let dir = FsDirectory::open(tmp.path().join("dict"));
        std::fs::create_dir_all(tmp.path().join("dict")).unwrap();
        // a failed build leaves both files behind, empty
        assert!(KnnVectorDict::build(&glove, &dir, "d").is_err());
        for ext in ["bin", "fst"] {
            assert_eq!(dir.file_length(&format!("d.{ext}")).unwrap(), 0, "{ext}");
        }
        // and the next build over them fails on the first, `.bin`
        let e = KnnVectorDict::build(&glove, &dir, "d").unwrap_err();
        assert_eq!(e.downcast_ref::<FileAlreadyExists>().unwrap().0, "d.bin");
        // only `.fst` there: `.bin` is created, then `.fst` fails
        dir.delete_file("d.bin").unwrap();
        let e = KnnVectorDict::build(&glove, &dir, "d").unwrap_err();
        assert_eq!(e.to_string(), "d.fst");
        assert_eq!(dir.file_length("d.bin").unwrap(), 0);
    }

    #[test]
    fn floats_parse_as_java_parses_them() {
        assert_eq!(
            parse_float(" x	").unwrap_err().to_string(),
            "java.lang.NumberFormatException: For input string: \"x\""
        );
        assert_eq!(parse_float(" 1.5f ").unwrap(), 1.5);
        assert_eq!(parse_float("2e1").unwrap(), 20.0);
        assert_eq!(parse_float("3D").unwrap(), 3.0);
        assert!(parse_float("NaN").unwrap().is_nan());
        assert_eq!(parse_float("-Infinity").unwrap(), f32::NEG_INFINITY);
        assert_eq!(parse_float("+Infinity").unwrap(), f32::INFINITY);
        assert!(parse_float("inf").is_err());
        assert!(parse_float("1e").is_err());
        assert_eq!(
            parse_float("").unwrap_err().to_string(),
            "java.lang.NumberFormatException: empty String"
        );
    }

    #[test]
    fn a_damaged_bin_is_refused() {
        let tmp = TempDir::new("tools-knn-bad");
        let dir = FsDirectory::open(tmp.path());
        let (_t, d) = dict("first 1 0\nalpha 0 2\n");
        drop(d);
        let glove = tmp.path().join("g.txt");
        std::fs::write(&glove, "first 1 0\nalpha 0 2\n").unwrap();
        KnnVectorDict::build(&glove, &dir, "d").unwrap();
        std::fs::write(tmp.path().join("d.bin"), [0u8, 0, 0, 0, 0, 3, 0, 0, 0]).unwrap();
        assert_eq!(
            KnnVectorDict::open(&dir, "d").unwrap_err().to_string(),
            "java.lang.IllegalStateException: vector file size 9 is not consonant with the \
             vector dimension 3"
        );
        std::fs::write(tmp.path().join("d.bin"), [0u8, 0]).unwrap();
        assert!(KnnVectorDict::open(&dir, "d")
            .unwrap_err()
            .to_string()
            .starts_with("java.io.EOFException"));
        // an ordinal past the vectors
        std::fs::write(tmp.path().join("d.bin"), [2u8, 0, 0, 0]).unwrap();
        let d = KnnVectorDict::open(&dir, "d").unwrap();
        assert!(d.get(b"alpha", &mut [0u8; 8]).is_err());
    }

    #[test]
    fn an_embedding_sums_and_normalizes_the_known_words() {
        let (_tmp, d) = dict("first 1 0\nalpha 0 2\nbeta 3 4\n");
        let d = d.unwrap();
        let e = DemoEmbeddings::new(&d);
        assert_eq!(
            e.compute_embedding("ALPHA unknown").unwrap(),
            vec![0.0, 1.0]
        );
        assert_eq!(
            e.compute_embedding("nothing known").unwrap(),
            vec![0.0, 0.0]
        );
        let both = e.compute_embedding("alpha beta").unwrap();
        assert!((both[0] * both[0] + both[1] * both[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lines_split_as_javas_reader_splits_them() {
        let lines: Vec<&str> = java_lines("a\rb\r\nc\n\nd").collect();
        assert_eq!(lines, ["a", "b", "c", "", "d"]);
        assert_eq!(split_fields("a b  "), ["a", "b"]);
        assert_eq!(split_fields(""), [""]);
        let dict = KnnVectorDict {
            fst: TypedFst::read(
                &TypedFst::<PositiveIntOutputs>::from_compiled({
                    let mut c =
                        FstCompilerBuilder::new(InputType::Byte1).build::<PositiveIntOutputs>();
                    c.add(&[97], 1).unwrap();
                    c.compile().unwrap()
                })
                .save(),
            )
            .unwrap(),
            fst_len: 0,
            vectors: Vec::new(),
            dimension: 1,
        };
        assert!(format!("{dict:?}").contains("dimension: 1"));
    }
}
