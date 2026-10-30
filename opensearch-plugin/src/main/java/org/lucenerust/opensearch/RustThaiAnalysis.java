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

/**
 * The {@code rust_thai} tokenizer and analyzer: {@link RustThaiTokenizer}, and OpenSearch's
 * {@code thai} analyzer chain around it (lower case, decimal digits, the {@code stopwords} setting
 * defaulting to Lucene's Thai stop set).
 */
final class RustThaiAnalysis {
    static final String NAME = "rust_thai";

    private RustThaiAnalysis() {}

    static final class TokenizerFactory extends AbstractTokenizerFactory {
        TokenizerFactory(IndexSettings indexSettings, Environment env, String name, Settings settings) {
            super(indexSettings, settings, name);
        }

        @Override
        public Tokenizer create() {
            return new RustThaiTokenizer();
        }
    }

    /** Lucene's {@code ThaiAnalyzer} with {@link RustThaiTokenizer}. */
    static final class RustThaiAnalyzer extends Analyzer {
        private final CharArraySet stopwords;

        RustThaiAnalyzer(CharArraySet stopwords) {
            this.stopwords = stopwords;
        }

        @Override
        protected TokenStreamComponents createComponents(String fieldName) {
            Tokenizer source = new RustThaiTokenizer();
            TokenStream result = new LowerCaseFilter(source);
            result = new DecimalDigitFilter(result);
            result = new StopFilter(result, stopwords);
            return new TokenStreamComponents(source, result);
        }

        @Override
        protected TokenStream normalize(String fieldName, TokenStream in) {
            return new DecimalDigitFilter(new LowerCaseFilter(in));
        }
    }

    static final class AnalyzerProvider extends AbstractIndexAnalyzerProvider<RustThaiAnalyzer> {
        private final RustThaiAnalyzer analyzer;

        AnalyzerProvider(IndexSettings indexSettings, Environment env, String name, Settings settings) {
            super(indexSettings, name, settings);
            analyzer = new RustThaiAnalyzer(Analysis.parseStopWords(env, settings, ThaiAnalyzer.getDefaultStopSet()));
        }

        @Override
        public RustThaiAnalyzer get() {
            return analyzer;
        }
    }
}
