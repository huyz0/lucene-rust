//! Hostile dictionaries: truncated and byte-flipped copies of the IPADIC
//! files read or fail, and what reads tokenizes or fails, never panicking
//! and never allocating more than the file can describe. Also the
//! caller-supplied-path loaders, over the vendored files written out.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::reader::StrReader;
use lucene_analysis::{TokenStream, Tokenizer};
use lucene_analysis_kuromoji::dict::{
    CharacterDefinition, ConnectionCosts, TokenInfoDictionary, UnknownDictionary,
};
use lucene_analysis_kuromoji::{JapaneseTokenizer, Mode};

fn inflate(name: &str) -> Vec<u8> {
    let path = format!("{}/src/resources/{name}", env!("CARGO_MANIFEST_DIR"));
    miniz_oxide::inflate::decompress_to_vec_zlib(&std::fs::read(path).unwrap()).unwrap()
}

const TEXT: &str = "関西国際空港で日本語の形態素解析をテストする。ｱｲｳ abc 123 😀";

/// Tokenizes [`TEXT`]; any outcome but a panic is fine.
fn run(t: &mut JapaneseTokenizer) -> Result<usize, lucene_analysis::AnalysisError> {
    t.set_reader(Box::new(StrReader::new(TEXT)))?;
    let r = (|| {
        t.reset()?;
        let mut n = 0;
        while t.increment_token()? {
            n += 1;
        }
        t.end()?;
        Ok(n)
    })();
    t.close()?;
    r
}

/// A cheap deterministic sequence of positions in `0..len`.
fn positions(len: usize, n: usize, seed: u64) -> Vec<usize> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as usize % len.max(1)
        })
        .collect()
}

#[test]
fn unknown_dictionary_survives_every_flip_and_cut() {
    let (map, pos, buf) = (
        inflate("unknown_target_map.dat.z"),
        inflate("unknown_pos_dict.dat.z"),
        inflate("unknown_buffer.dat.z"),
    );
    let known = TokenInfoDictionary::instance();
    let costs = ConnectionCosts::instance();
    let mut tokenized = 0;
    for which in 0..3 {
        let file = [&map, &pos, &buf][which];
        let mut variants: Vec<Vec<u8>> = (0..file.len()).map(|cut| file[..cut].to_vec()).collect();
        for i in 0..file.len() {
            let mut f = file.clone();
            f[i] ^= 0xFF;
            variants.push(f);
        }
        for v in variants {
            let files = match which {
                0 => [&v, &pos, &buf],
                1 => [&map, &v, &buf],
                _ => [&map, &pos, &v],
            };
            if let Ok(unk) = UnknownDictionary::read(files[0], files[1], files[2]) {
                let mut t = JapaneseTokenizer::with_dictionaries(
                    Arc::clone(&known),
                    Arc::new(unk),
                    Arc::clone(&costs),
                    None,
                    true,
                    true,
                    Mode::Search,
                );
                tokenized += usize::from(run(&mut t).is_ok());
            }
        }
    }
    assert!(tokenized > 0);
}

#[test]
fn character_definition_and_costs_survive_cuts_and_flips() {
    let cd = inflate("character_definition.dat.z");
    for cut in (0..cd.len()).step_by(97).chain([cd.len() - 1]) {
        assert!(CharacterDefinition::read(&cd[..cut]).is_err());
    }
    for i in positions(cd.len(), 400, 1).into_iter().chain(0..40) {
        let mut f = cd.clone();
        f[i] ^= 0xFF;
        let _ = CharacterDefinition::read(&f);
    }
    let cc = inflate("connection_costs.dat.z");
    for cut in [0, 5, 20, 30, 1000, cc.len() / 2, cc.len() - 1] {
        assert!(ConnectionCosts::read(&cc[..cut]).is_err());
    }
    for i in (0..40).chain(positions(cc.len(), 30, 2)) {
        let mut f = cc.clone();
        f[i] ^= 0xFF;
        let _ = ConnectionCosts::read(&f);
    }
}

#[test]
fn system_dictionary_survives_sampled_cuts_and_flips() {
    let files = [
        inflate("token_info_target_map.dat.z"),
        inflate("token_info_pos_dict.dat.z"),
        inflate("token_info_buffer.dat.z"),
        inflate("token_info_fst.dat.z"),
    ];
    let unk = UnknownDictionary::instance();
    let costs = ConnectionCosts::instance();
    let mut tokenized = 0;
    for which in 0..4 {
        let file = &files[which];
        let mut variants: Vec<Vec<u8>> = [0, 30, file.len() / 3, file.len() - 1]
            .iter()
            .map(|&c| file[..c.min(file.len())].to_vec())
            .collect();
        for i in (0..12).chain(positions(file.len(), 5, which as u64 + 10)) {
            let mut f = file.clone();
            f[i] ^= 0xFF;
            variants.push(f);
        }
        for v in variants {
            let mut set: Vec<&[u8]> = files.iter().map(Vec::as_slice).collect();
            set[which] = &v;
            if let Ok(d) = TokenInfoDictionary::read(set[0], set[1], set[2], set[3]) {
                let mut t = JapaneseTokenizer::with_dictionaries(
                    Arc::new(d),
                    Arc::clone(&unk),
                    Arc::clone(&costs),
                    None,
                    false,
                    false,
                    Mode::Extended,
                );
                t.set_n_best_cost(3000);
                tokenized += usize::from(run(&mut t).is_ok());
            }
        }
    }
    assert!(tokenized > 0);
}

#[test]
fn dictionaries_load_from_caller_paths() {
    let dir: PathBuf = std::env::temp_dir().join(format!("kuromoji-dict-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, z) in [
        (
            "TokenInfoDictionary$targetMap.dat",
            "token_info_target_map.dat.z",
        ),
        (
            "TokenInfoDictionary$posDict.dat",
            "token_info_pos_dict.dat.z",
        ),
        ("TokenInfoDictionary$buffer.dat", "token_info_buffer.dat.z"),
        ("TokenInfoDictionary$fst.dat", "token_info_fst.dat.z"),
        (
            "UnknownDictionary$targetMap.dat",
            "unknown_target_map.dat.z",
        ),
        ("UnknownDictionary$posDict.dat", "unknown_pos_dict.dat.z"),
        ("UnknownDictionary$buffer.dat", "unknown_buffer.dat.z"),
        ("ConnectionCosts.dat", "connection_costs.dat.z"),
    ] {
        std::fs::write(dir.join(name), inflate(z)).unwrap();
    }
    let known = TokenInfoDictionary::from_dir(&dir, "TokenInfoDictionary").unwrap();
    let unk = UnknownDictionary::from_paths(
        &dir.join("UnknownDictionary$targetMap.dat"),
        &dir.join("UnknownDictionary$posDict.dat"),
        &dir.join("UnknownDictionary$buffer.dat"),
    )
    .unwrap();
    let costs = ConnectionCosts::from_path(&dir.join("ConnectionCosts.dat")).unwrap();
    let mut own = JapaneseTokenizer::with_dictionaries(
        Arc::new(known),
        Arc::new(unk),
        Arc::new(costs),
        None,
        true,
        true,
        Mode::Search,
    );
    let mut default = JapaneseTokenizer::with_options(None, true, true, Mode::Search);
    assert_eq!(run(&mut own).unwrap(), run(&mut default).unwrap());
    let missing = TokenInfoDictionary::from_dir(&dir, "Nope")
        .unwrap_err()
        .to_string();
    assert!(missing.contains("NoSuchFileException"), "{missing}");
    assert!(ConnectionCosts::from_path(&dir).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
