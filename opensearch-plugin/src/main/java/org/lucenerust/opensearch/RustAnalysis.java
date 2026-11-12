/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
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

import java.util.ArrayList;
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
 * <p>The language modules' tokenizers ({@code japanese}, {@code korean},
 * {@code chinese}) read their dictionaries from data packs, which a node opts into by
 * putting the pack directories (from {@code scripts/make-data-packs.sh}) under
 * {@code <config>/lucene-rust/packs/}. OpenSearch builds every registered tokenizer for every
 * index, so a tokenizer whose pack is missing is not registered at all (the node logs which pack it
 * lacks), and an index naming it is refused as naming an unknown tokenizer.
 */
final class RustAnalysis {
    private RustAnalysis() {}

    private static final Logger logger = LogManager.getLogger(RustAnalysis.class);

    /**
     * One {@code rust_*} tokenizer, and the analyzer of the same name if {@code chain} is set;
     * {@code pack} names the data pack it reads, if any.
     */
    static final class Entry {
        final String name;
        final String spi;
        final String pack;
        final Map<String, String> args;
        final CharArraySet defaultStopWords;
        final Chain chain;
        private final byte[] spec;

        Entry(String name, String spi, String pack, Map<String, String> args, CharArraySet defaultStopWords, Chain chain) {
            this.name = name;
            this.spi = spi;
            this.pack = pack;
            this.args = RustTokenizer.args(args);
            this.defaultStopWords = defaultStopWords;
            this.chain = chain;
            this.spec = RustTokenizer.spec(spi, this.args);
        }

        RustTokenizer tokenizer() {
            return new RustTokenizer(name, spi, args, spec);
        }

        /** Builds the Rust tokenizer once: a missing data pack throws, naming the pack's file. */
        void check() {
            if (AnalyzerSpec.rust(spec, "", "") == null) {
                throw new IllegalArgumentException("[" + name + "]: the accelerated tokenizer is unavailable: " + NativeBridge.lastError());
            }
        }
    }

    /** An analyzer's chain over its tokenizer: the filters, then {@code normalize}'s. */
    interface Chain {
        TokenStream filters(Tokenizer source, CharArraySet stopWords);

        TokenStream normalize(TokenStream in);
    }

    /** {@code thai_dictionary}: Lucene's {@code ThaiAnalyzer} chain over ICU's Thai dictionary in Rust. */
    static final Entry THAI = new Entry("thai_dictionary", "thai", null, Map.of(), ThaiAnalyzer.getDefaultStopSet(), new Chain() {
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

    /** {@code japanese}: Lucene's {@code JapaneseTokenizer} (IPADIC; the {@code kuromoji} pack). */
    static final Entry JAPANESE = new Entry("japanese", "japanese", "kuromoji", Map.of(), null, null);
    /** {@code korean}: Lucene's {@code KoreanTokenizer} (mecab-ko-dic; the {@code nori} pack). */
    static final Entry KOREAN = new Entry("korean", "korean", "nori", Map.of(), null, null);
    /** {@code chinese}: Lucene's {@code HMMChineseTokenizer} (the {@code smartcn} pack). */
    static final Entry CHINESE = new Entry("chinese", "hmmChinese", "smartcn", Map.of(), null, null);

    /** Every {@code rust_*} tokenizer the plugin registers. */
    static final List<Entry> TOKENIZERS = List.of(THAI, JAPANESE, KOREAN, CHINESE);

    /** The entries this node registers: those reading no data pack, and those whose pack is there. */
    static List<Entry> available() {
        List<Entry> out = new ArrayList<>();
        for (Entry e : TOKENIZERS) {
            if (e.pack == null) {
                out.add(e);
                continue;
            }
            try {
                e.check();
                out.add(e);
            } catch (IllegalArgumentException ex) {
                logger.info("lucene-rust: [{}] not registered, its data pack [{}] is not installed: {}", e.name, e.pack, ex.getMessage());
            }
        }
        return out;
    }

    static Map<String, AnalysisModule.AnalysisProvider<TokenizerFactory>> tokenizers() {
        Map<String, AnalysisModule.AnalysisProvider<TokenizerFactory>> m = new LinkedHashMap<>();
        for (Entry e : available()) {
            m.put(e.name, (indexSettings, env, name, settings) -> new Factory(indexSettings, name, settings, e));
        }
        return m;
    }

    static Map<String, AnalysisModule.AnalysisProvider<AnalyzerProvider<? extends Analyzer>>> analyzers() {
        Map<String, AnalysisModule.AnalysisProvider<AnalyzerProvider<? extends Analyzer>>> m = new LinkedHashMap<>();
        for (Entry e : available()) {
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
            analyzer = new RustAnalyzer(entry, Analysis.parseStopWords(env, settings, entry.defaultStopWords));
        }

        @Override
        public RustAnalyzer get() {
            return analyzer;
        }
    }
}
