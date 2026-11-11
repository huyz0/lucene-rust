/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.ml;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;
import org.apache.lucene.util.BytesRef;
import org.lucenerust.opensearch.NativeBridge;
import org.lucenerust.opensearch.neural.ml.SparseEmbeddingFormat;
import org.opensearch.common.settings.Settings;
import org.opensearch.env.Environment;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.analysis.AbstractIndexAnalyzerProvider;
import org.opensearch.index.analysis.AbstractTokenizerFactory;
import org.opensearch.index.analysis.AnalyzerProvider;
import org.opensearch.index.analysis.PreBuiltAnalyzerProviderFactory;
import org.opensearch.index.analysis.PreConfiguredTokenizer;
import org.opensearch.index.analysis.TokenizerFactory;
import org.opensearch.indices.analysis.AnalysisModule;
import org.opensearch.indices.analysis.PreBuiltCacheFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * ml-commons' Hugging Face analysis: the {@code hf_model_tokenizer} tokenizer and analyzer types
 * and the pre-built {@code bert-uncased} and {@code mbert-uncased} tokenizers and analyzers. The
 * word pieces come from the native tokenizer; the vocabularies and idf weights are ml-commons'
 * own, bundled with the plugin.
 */
public final class HfAnalysis {
    public static final String NAME = "hf_model_tokenizer";
    public static final String DEFAULT_TOKENIZER_NAME = "bert-uncased";
    public static final String DEFAULT_MULTILINGUAL_TOKENIZER_NAME = "mbert-uncased";

    private HfAnalysis() {}

    public static Map<String, AnalysisModule.AnalysisProvider<TokenizerFactory>> tokenizers() {
        return Map.of(NAME, Factory::new);
    }

    public static Map<String, AnalysisModule.AnalysisProvider<AnalyzerProvider<? extends Analyzer>>> analyzers() {
        return Map.of(NAME, Provider::new);
    }

    public static List<PreConfiguredTokenizer> preConfiguredTokenizers() {
        return List.of(
            PreConfiguredTokenizer.singleton(DEFAULT_TOKENIZER_NAME, HfAnalysis::createDefault),
            PreConfiguredTokenizer.singleton(DEFAULT_MULTILINGUAL_TOKENIZER_NAME, HfAnalysis::createDefaultMultilingual)
        );
    }

    public static List<PreBuiltAnalyzerProviderFactory> preBuiltAnalyzers() {
        return List.of(
            new PreBuiltAnalyzerProviderFactory(
                DEFAULT_TOKENIZER_NAME,
                PreBuiltCacheFactory.CachingStrategy.ONE,
                () -> new HfAnalyzer(HfAnalysis::createDefault)
            ),
            new PreBuiltAnalyzerProviderFactory(
                DEFAULT_MULTILINGUAL_TOKENIZER_NAME,
                PreBuiltCacheFactory.CachingStrategy.ONE,
                () -> new HfAnalyzer(HfAnalysis::createDefaultMultilingual)
            )
        );
    }

    /** A {@code bert-uncased} tokenizer. */
    public static Tokenizer createDefault() {
        return new HfTokenizer(Bert.HANDLE);
    }

    /** An {@code mbert-uncased} tokenizer. */
    public static Tokenizer createDefaultMultilingual() {
        return new HfTokenizer(MultilingualBert.HANDLE);
    }

    /** The native analyzer handles, each opened on first use and kept for the node's life. */
    private static final class Bert {
        static final Handle HANDLE = new Handle("/analysis/bert-uncased.zip");
    }

    private static final class MultilingualBert {
        static final Handle HANDLE = new Handle("/analysis/mbert-uncased.zip");
    }

    static final class Handle {
        private final String resource;
        private volatile long handle;

        Handle(String resource) {
            this.resource = resource;
        }

        long get() {
            long h = handle;
            if (h == 0) {
                synchronized (this) {
                    h = handle;
                    if (h == 0) {
                        h = open(resource);
                        handle = h;
                    }
                }
            }
            return h;
        }
    }

    private static long open(String resource) {
        byte[] tokenizer = null;
        byte[] idf = null;
        try (InputStream in = HfAnalysis.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException("Invalid resource path " + resource);
            }
            try (ZipInputStream zip = new ZipInputStream(in)) {
                for (ZipEntry e = zip.getNextEntry(); e != null; e = zip.getNextEntry()) {
                    String name = e.getName().substring(e.getName().lastIndexOf('/') + 1);
                    if (name.equals("tokenizer.json")) {
                        tokenizer = zip.readAllBytes();
                    } else if (name.equals("idf.json")) {
                        idf = zip.readAllBytes();
                    }
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read analyzer " + resource, e);
        }
        if (tokenizer == null) {
            throw new IllegalStateException("Analyzer " + resource + " has no tokenizer.json");
        }
        long[] out = new long[1];
        if (NativeBridge.mlHfAnalyzerOpen(tokenizer, idf, out) != NativeBridge.OK) {
            throw new IllegalStateException("Failed to initialize tokenizer: " + resource + ": " + NativeBridge.lastError());
        }
        return out[0];
    }

    /** ml-commons' {@code HFModelTokenizer}: the word pieces of the input, each weighted in its payload. */
    static final class HfTokenizer extends Tokenizer {
        private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
        private final OffsetAttribute offsetAtt = addAttribute(OffsetAttribute.class);
        private final TypeAttribute typeAtt = addAttribute(TypeAttribute.class);
        private final PayloadAttribute payloadAtt = addAttribute(PayloadAttribute.class);
        private final Handle handle;
        private ByteBuffer tokens;
        private int remaining;

        HfTokenizer(Handle handle) {
            this.handle = handle;
        }

        @Override
        public void reset() throws IOException {
            super.reset();
            String text = readAll(input);
            tokens = null;
            remaining = 0;
            if (text.isEmpty()) {
                return;
            }
            byte[] out = NativeBridge.mlHfAnalyze(handle.get(), text.getBytes(StandardCharsets.UTF_8));
            if (out == null) {
                throw new IllegalStateException("Failed to tokenize: " + NativeBridge.lastError());
            }
            tokens = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
            remaining = tokens.getInt();
        }

        @Override
        public boolean incrementToken() {
            String type = typeAtt.type();
            clearAttributes();
            typeAtt.setType(type);
            if (remaining == 0) {
                return false;
            }
            remaining--;
            int id = tokens.getInt();
            int start = tokens.getInt();
            int end = tokens.getInt();
            int length = tokens.getInt();
            String token = new String(tokens.array(), tokens.position(), length, StandardCharsets.UTF_8);
            tokens.position(tokens.position() + length);
            float weight = tokens.getFloat();
            SparseEmbeddingFormat format = SparseEmbeddingFormat.valueOf(type.toUpperCase(Locale.ROOT));
            termAtt.append(format == SparseEmbeddingFormat.WORD ? token : String.valueOf(id));
            offsetAtt.setOffset(start, end);
            payloadAtt.setPayload(new BytesRef(ByteBuffer.allocate(Float.BYTES).putFloat(weight).array()));
            return true;
        }

        private static String readAll(Reader reader) throws IOException {
            StringBuilder sb = new StringBuilder();
            char[] buf = new char[1024];
            for (int n = reader.read(buf); n != -1; n = reader.read(buf)) {
                sb.append(buf, 0, n);
            }
            return sb.toString();
        }
    }

    /** ml-commons' {@code HFModelAnalyzer}. */
    static final class HfAnalyzer extends Analyzer {
        private final java.util.function.Supplier<Tokenizer> tokenizer;

        HfAnalyzer(java.util.function.Supplier<Tokenizer> tokenizer) {
            this.tokenizer = tokenizer;
        }

        @Override
        protected TokenStreamComponents createComponents(String fieldName) {
            Tokenizer src = tokenizer.get();
            return new TokenStreamComponents(src, src);
        }
    }

    /** ml-commons' {@code HFModelTokenizerFactory}: a custom {@code hf_model_tokenizer} is the {@code bert-uncased} one. */
    static final class Factory extends AbstractTokenizerFactory {
        Factory(IndexSettings indexSettings, Environment environment, String name, Settings settings) {
            super(indexSettings, settings, name);
        }

        @Override
        public Tokenizer create() {
            return createDefault();
        }
    }

    /** ml-commons' {@code HFModelAnalyzerProvider}. */
    static final class Provider extends AbstractIndexAnalyzerProvider<HfAnalyzer> {
        private final HfAnalyzer analyzer = new HfAnalyzer(HfAnalysis::createDefault);

        Provider(IndexSettings indexSettings, Environment environment, String name, Settings settings) {
            super(indexSettings, name, settings);
        }

        @Override
        public HfAnalyzer get() {
            return analyzer;
        }
    }
}
