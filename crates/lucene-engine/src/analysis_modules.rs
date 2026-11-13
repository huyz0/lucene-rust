//! The Lucene analysis modules a spec ([`crate::analysis`]) can name,
//! registered with `lucene-analysis`' SPI registry once, before the first
//! spec is built: Java's `ServiceLoader` sees every module jar on the
//! plugin's classpath, and this is the port's classpath. Each module is a
//! cargo feature of this crate (all on by default); a spec naming a factory
//! of a module left out fails to build, and the plugin keeps that analyzer
//! in Java, as for any spec Rust refuses.
//!
//! The modules' dictionaries are data packs ([`lucene_analysis::data_pack`]):
//! a spec naming a module whose pack is in no pack directory
//! (`ffi_add_data_pack_dir`) fails to build the same way, unless the
//! `embedded-data` feature compiled the packs in.

use std::sync::Once;

static REGISTER: Once = Once::new();

/// Registers every enabled module's factories, and the `thai` tokenizer's
/// word iterator (ICU's Thai dictionary); idempotent.
pub fn register() {
    REGISTER.call_once(|| {
        lucene_analysis_icu::install_thai_word_breaker();
        // A module whose names are taken already keeps the first class, as
        // Java's lookup does; registration itself only fails on an invalid
        // name, which the modules' own tests rule out.
        #[cfg(feature = "analysis-icu")]
        let _ = lucene_analysis_icu::register_factories();
        #[cfg(feature = "analysis-kuromoji")]
        let _ = lucene_analysis_kuromoji::register_factories();
        #[cfg(feature = "analysis-nori")]
        let _ = lucene_analysis_nori::register_factories();
        #[cfg(feature = "analysis-smartcn")]
        let _ = lucene_analysis_smartcn::register_factories();
        #[cfg(feature = "analysis-stempel")]
        let _ = lucene_analysis_stempel::register_factories();
        #[cfg(feature = "analysis-morfologik")]
        let _ = lucene_analysis_morfologik::register_factories();
        #[cfg(feature = "analysis-phonetic")]
        let _ = lucene_analysis_phonetic::register_factories();
    });
}

#[cfg(test)]
mod tests {
    use lucene_analysis::factory::spi;

    #[test]
    fn enabled_modules_are_registered() {
        super::register();
        super::register();
        let tokenizers = spi::available_tokenizers();
        let filters = spi::available_token_filters();
        assert!(tokenizers.contains(&"thai"));
        #[cfg(feature = "analysis-icu")]
        assert!(tokenizers.contains(&"icu"));
        #[cfg(feature = "analysis-kuromoji")]
        assert!(tokenizers.contains(&"japanese"));
        #[cfg(feature = "analysis-nori")]
        assert!(tokenizers.contains(&"korean"));
        #[cfg(feature = "analysis-smartcn")]
        assert!(tokenizers.contains(&"hmmChinese"));
        #[cfg(feature = "analysis-stempel")]
        assert!(filters.contains(&"stempelPolishStem"));
        #[cfg(feature = "analysis-morfologik")]
        assert!(filters.contains(&"morfologik"));
        #[cfg(feature = "analysis-phonetic")]
        assert!(filters.contains(&"phonetic"));
        let _ = filters;
    }
}
