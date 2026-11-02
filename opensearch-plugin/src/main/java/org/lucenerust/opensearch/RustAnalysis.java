/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.DecimalDigitFilter;
import org.apache.lucene.analysis.th.ThaiAnalyzer;
import org.opensearch.common.settings.Settings;
import org.opensearch.env.Environment;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.analysis.AbstractIndexAnalyzerProvider;
import org.opensearch.index.analysis.AbstractTokenizerFactory;
import org.opensearch.index.analysis.Analysis;
import org.opensearch.index.analysis.AnalyzerProvider;
import org.opensearch.index.analysis.TokenizerFactory;
import org.opensearch.indices.analysis.AnalysisModule;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The plugin's {@code rust_*} tokenizers and analyzers: Rust tokenizers ({@link RustTokenizer})
 * under an OpenSearch name, each with an optional analyzer chain around it.
 *
 * <p>Adding one is a row in {@link #TOKENIZERS}: the OpenSearch name, the Rust factory's SPI name
 * and its arguments, and, for an analyzer of the same name, how its chain wraps the tokenizer
 * (the stop word default comes from the analyzer's {@code stopwords} setting). The Rust factory
 * must be registered in {@code lucene-ffi}'s build; {@link AnalyzerSpec} maps every
 * {@link RustTokenizer} by its SPI name and arguments, so nothing else changes on either side.
 *
 * <p>The language modules' tokenizers ({@code rust_japanese}, {@code rust_korean},
 * {@code rust_chinese}) read their dictionaries from data packs, which a node opts into by
 * putting the pack directories (from {@code scripts/make-data-packs.sh}) under
 * {@code <config>/lucene-rust/packs/}. An index naming one of them without its pack is refused
 * when the index is created, with the missing file in the message.
 */
final class RustAnalysis {
    private RustAnalysis() {}

    /** One {@code rust_*} tokenizer, and the analyzer of the same name if {@code chain} is set. */
    static final class Entry {
        final String name;
        final String spi;
        final Map<String, String> args;
        final CharArraySet defaultStopWords;
        final Chain chain;
        private final byte[] spec;

        Entry(String name, String spi, Map<String, String> args, CharArraySet defaultStopWords, Chain chain) {
            this.name = name;
            this.spi = spi;
            this.args = RustTokenizer.args(args);
            this.defaultStopWords = defaultStopWords;
            this.chain = chain;
            this.spec = RustTokenizer.spec(spi, this.args);
        }

        RustTokenizer tokenizer() {
            return new RustTokenizer(name, spi, args, spec);
        }

        /** Builds the Rust tokenizer once, so a missing data pack refuses the index, not a document. */
        void check() {
            if (AnalyzerSpec.rust(spec, "", "") == null) {
                throw new IllegalArgumentException("[" + name + "]: the Rust tokenizer is unavailable: " + NativeBridge.lastError());
            }
        }
    }

    /** An analyzer's chain over its tokenizer: the filters, then {@code normalize}'s. */
    interface Chain {
        TokenStream filters(Tokenizer source, CharArraySet stopWords);

        TokenStream normalize(TokenStream in);
    }

    /** {@code rust_thai}: Lucene's {@code ThaiAnalyzer} chain over ICU's Thai dictionary in Rust. */
    static final Entry THAI = new Entry("rust_thai", "thai", Map.of(), ThaiAnalyzer.getDefaultStopSet(), new Chain() {
        @Override
        public TokenStream filters(Tokenizer source, CharArraySet stopWords) {
            TokenStream result = new LowerCaseFilter(source);
            result = new DecimalDigitFilter(result);
            return new StopFilter(result, stopWords);
        }

        @Override
        public TokenStream normalize(TokenStream in) {
            return new DecimalDigitFilter(new LowerCaseFilter(in));
        }
    });

    /** {@code rust_japanese}: Lucene's {@code JapaneseTokenizer} (IPADIC; the {@code kuromoji} pack). */
    static final Entry JAPANESE = new Entry("rust_japanese", "japanese", Map.of(), null, null);
    /** {@code rust_korean}: Lucene's {@code KoreanTokenizer} (mecab-ko-dic; the {@code nori} pack). */
    static final Entry KOREAN = new Entry("rust_korean", "korean", Map.of(), null, null);
    /** {@code rust_chinese}: Lucene's {@code HMMChineseTokenizer} (the {@code smartcn} pack). */
    static final Entry CHINESE = new Entry("rust_chinese", "hmmChinese", Map.of(), null, null);

    /** Every {@code rust_*} tokenizer the plugin registers. */
    static final List<Entry> TOKENIZERS = List.of(THAI, JAPANESE, KOREAN, CHINESE);

    static Map<String, AnalysisModule.AnalysisProvider<TokenizerFactory>> tokenizers() {
        Map<String, AnalysisModule.AnalysisProvider<TokenizerFactory>> m = new LinkedHashMap<>();
        for (Entry e : TOKENIZERS) {
            m.put(e.name, (indexSettings, env, name, settings) -> new Factory(indexSettings, name, settings, e));
        }
        return m;
    }

    static Map<String, AnalysisModule.AnalysisProvider<AnalyzerProvider<? extends Analyzer>>> analyzers() {
        Map<String, AnalysisModule.AnalysisProvider<AnalyzerProvider<? extends Analyzer>>> m = new LinkedHashMap<>();
        for (Entry e : TOKENIZERS) {
            if (e.chain != null) {
                m.put(e.name, (indexSettings, env, name, settings) -> new Provider(indexSettings, env, name, settings, e));
            }
        }
        return m;
    }

    static final class Factory extends AbstractTokenizerFactory {
        private final Entry entry;

        Factory(IndexSettings indexSettings, String name, Settings settings, Entry entry) {
            super(indexSettings, settings, name);
            this.entry = entry;
            entry.check();
        }

        @Override
        public Tokenizer create() {
            return entry.tokenizer();
        }
    }

    /** An entry's analyzer: its chain over its tokenizer, with the given stop words. */
    static final class RustAnalyzer extends Analyzer {
        private final Entry entry;
        private final CharArraySet stopWords;

        RustAnalyzer(Entry entry, CharArraySet stopWords) {
            this.entry = entry;
            this.stopWords = stopWords;
        }

        @Override
        protected TokenStreamComponents createComponents(String fieldName) {
            Tokenizer source = entry.tokenizer();
            return new TokenStreamComponents(source, entry.chain.filters(source, stopWords));
        }

        @Override
        protected TokenStream normalize(String fieldName, TokenStream in) {
            return entry.chain.normalize(in);
        }
    }

    static final class Provider extends AbstractIndexAnalyzerProvider<RustAnalyzer> {
        private final RustAnalyzer analyzer;

        Provider(IndexSettings indexSettings, Environment env, String name, Settings settings, Entry entry) {
            super(indexSettings, name, settings);
            entry.check();
            analyzer = new RustAnalyzer(entry, Analysis.parseStopWords(env, settings, entry.defaultStopWords));
        }

        @Override
        public RustAnalyzer get() {
            return analyzer;
        }
    }
}
