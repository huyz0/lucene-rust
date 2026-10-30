/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.similarities.AfterEffect;
import org.apache.lucene.search.similarities.AfterEffectB;
import org.apache.lucene.search.similarities.AfterEffectL;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.BasicModel;
import org.apache.lucene.search.similarities.BasicModelG;
import org.apache.lucene.search.similarities.BasicModelIF;
import org.apache.lucene.search.similarities.BasicModelIn;
import org.apache.lucene.search.similarities.BasicModelIne;
import org.apache.lucene.search.similarities.BooleanSimilarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.DFISimilarity;
import org.apache.lucene.search.similarities.DFRSimilarity;
import org.apache.lucene.search.similarities.Distribution;
import org.apache.lucene.search.similarities.DistributionLL;
import org.apache.lucene.search.similarities.DistributionSPL;
import org.apache.lucene.search.similarities.IBSimilarity;
import org.apache.lucene.search.similarities.Independence;
import org.apache.lucene.search.similarities.IndependenceChiSquared;
import org.apache.lucene.search.similarities.IndependenceSaturated;
import org.apache.lucene.search.similarities.IndependenceStandardized;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.LMJelinekMercerSimilarity;
import org.apache.lucene.search.similarities.LMSimilarity;
import org.apache.lucene.search.similarities.Lambda;
import org.apache.lucene.search.similarities.LambdaDF;
import org.apache.lucene.search.similarities.LambdaTTF;
import org.apache.lucene.search.similarities.Normalization;
import org.apache.lucene.search.similarities.NormalizationH1;
import org.apache.lucene.search.similarities.NormalizationH2;
import org.apache.lucene.search.similarities.NormalizationH3;
import org.apache.lucene.search.similarities.NormalizationZ;
import org.apache.lucene.search.similarities.PerFieldSimilarityWrapper;
import org.apache.lucene.search.similarities.Similarity;
import org.opensearch.lucene.similarity.LegacyBM25Similarity;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Predicate;

/**
 * The searcher's similarity for each field a query scores, as the table {@code
 * crates/lucene-ffi/src/jvm_similarity.rs} reads in front of the query blob.
 *
 * <p>As the {@link QueryEncoder}'s field predicate it accepts a field whose similarity is
 * Lucene's default {@code BM25Similarity} (nothing to send) or one of the classes OpenSearch's
 * {@code SimilarityProviders} builds -- {@code BM25} with any parameters, {@code boolean}, {@code
 * DFR}, {@code IB}, {@code DFI}, {@code LMDirichlet}, {@code LMJelinekMercer}, {@code LegacyBM25},
 * {@code classic} -- and records it. It refuses anything else, which falls back as {@code
 * field_similarity}: {@code scripted} runs the user's script and a plugin's similarity the
 * plugin's code. Matched by exact class: a subclass may score differently.
 */
final class SimilarityEncoder implements Predicate<String> {
    static final byte QUERY_SIMILARITY = 4;

    private final Similarity searcher;
    /** The recorded fields' encoded similarities, in the order first seen. */
    private final Map<String, byte[]> fields = new LinkedHashMap<>();

    SimilarityEncoder(Similarity searcher) {
        this.searcher = searcher;
    }

    /** {@code field}'s similarity: the per-field wrapper's choice, or the searcher's own. */
    Similarity of(String field) {
        return searcher instanceof PerFieldSimilarityWrapper w ? w.get(field) : searcher;
    }

    /** Whether {@code field} scores with Lucene's default {@code BM25Similarity}. */
    boolean isDefault(String field) {
        return isDefault(of(field));
    }

    static boolean isDefault(Similarity sim) {
        return sim != null
            && sim.getClass() == BM25Similarity.class
            && ((BM25Similarity) sim).getK1() == 1.2f
            && ((BM25Similarity) sim).getB() == 0.75f
            && sim.getDiscountOverlaps();
    }

    @Override
    public boolean test(String field) {
        Similarity sim = of(field);
        if (isDefault(sim) || fields.containsKey(field)) {
            return true;
        }
        byte[] spec = spec(sim);
        if (spec == null) {
            return false;
        }
        fields.put(field, spec);
        return true;
    }

    /** {@code blob} behind the recorded table, or as it is when every field is the default. */
    byte[] prefix(byte[] blob) {
        if (blob == null || fields.isEmpty()) {
            return blob;
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(QUERY_SIMILARITY);
        writeInt(out, fields.size());
        for (Map.Entry<String, byte[]> e : fields.entrySet()) {
            byte[] name = e.getKey().getBytes(StandardCharsets.UTF_8);
            writeInt(out, name.length);
            out.writeBytes(name);
            out.writeBytes(e.getValue());
        }
        out.writeBytes(blob);
        return out.toByteArray();
    }

    /** One similarity's encoding, or null when it is not one the native side computes. */
    @SuppressWarnings("deprecation") // LegacyBM25Similarity: OpenSearch still builds it for LegacyBM25
    static byte[] spec(Similarity sim) {
        if (sim == null) {
            return null;
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        Class<?> c = sim.getClass();
        if (c == BM25Similarity.class) {
            BM25Similarity b = (BM25Similarity) sim;
            out.write(0);
            writeFloat(out, b.getK1());
            writeFloat(out, b.getB());
        } else if (c == BooleanSimilarity.class) {
            out.write(1);
            return out.toByteArray();
        } else if (c == DFRSimilarity.class) {
            DFRSimilarity d = (DFRSimilarity) sim;
            int basic = basicModel(d.getBasicModel());
            int after = afterEffect(d.getAfterEffect());
            if (basic < 0 || after < 0) {
                return null;
            }
            out.write(2);
            out.write(basic);
            out.write(after);
            if (normalization(d.getNormalization(), out) == false) {
                return null;
            }
        } else if (c == IBSimilarity.class) {
            IBSimilarity ib = (IBSimilarity) sim;
            int distribution = distribution(ib.getDistribution());
            int lambda = lambda(ib.getLambda());
            if (distribution < 0 || lambda < 0) {
                return null;
            }
            out.write(3);
            out.write(distribution);
            out.write(lambda);
            if (normalization(ib.getNormalization(), out) == false) {
                return null;
            }
        } else if (c == DFISimilarity.class) {
            int independence = independence(((DFISimilarity) sim).getIndependence());
            if (independence < 0) {
                return null;
            }
            out.write(4);
            out.write(independence);
        } else if (c == LMDirichletSimilarity.class && defaultCollectionModel((LMSimilarity) sim)) {
            out.write(5);
            writeFloat(out, ((LMDirichletSimilarity) sim).getMu());
        } else if (c == LMJelinekMercerSimilarity.class && defaultCollectionModel((LMSimilarity) sim)) {
            out.write(6);
            writeFloat(out, ((LMJelinekMercerSimilarity) sim).getLambda());
        } else if (c == ClassicSimilarity.class) {
            out.write(7);
        } else if (c == LegacyBM25Similarity.class) {
            // Its own discountOverlaps is Similarity's default; the wrapped BM25's computes the norms.
            BM25Similarity bm25 = legacyBm25(sim);
            if (bm25 == null) {
                return null;
            }
            out.write(8);
            writeFloat(out, bm25.getK1());
            writeFloat(out, bm25.getB());
            out.write(bm25.getDiscountOverlaps() ? 1 : 0);
            return out.toByteArray();
        } else {
            return null;
        }
        out.write(sim.getDiscountOverlaps() ? 1 : 0);
        return out.toByteArray();
    }

    private static int basicModel(BasicModel m) {
        Class<?> c = m.getClass();
        return c == BasicModelG.class ? 0 : c == BasicModelIF.class ? 1 : c == BasicModelIn.class ? 2 : c == BasicModelIne.class ? 3 : -1;
    }

    private static int afterEffect(AfterEffect a) {
        Class<?> c = a.getClass();
        return c == AfterEffectB.class ? 0 : c == AfterEffectL.class ? 1 : -1;
    }

    private static int distribution(Distribution d) {
        Class<?> c = d.getClass();
        return c == DistributionLL.class ? 0 : c == DistributionSPL.class ? 1 : -1;
    }

    private static int lambda(Lambda l) {
        Class<?> c = l.getClass();
        return c == LambdaDF.class ? 0 : c == LambdaTTF.class ? 1 : -1;
    }

    private static int independence(Independence i) {
        Class<?> c = i.getClass();
        return c == IndependenceStandardized.class ? 0
            : c == IndependenceSaturated.class ? 1
            : c == IndependenceChiSquared.class ? 2
            : -1;
    }

    private static boolean normalization(Normalization n, ByteArrayOutputStream out) {
        Class<?> c = n.getClass();
        if (c == Normalization.NoNormalization.class) {
            out.write(0);
        } else if (c == NormalizationH1.class) {
            out.write(1);
            writeFloat(out, ((NormalizationH1) n).getC());
        } else if (c == NormalizationH2.class) {
            out.write(2);
            writeFloat(out, ((NormalizationH2) n).getC());
        } else if (c == NormalizationH3.class) {
            out.write(3);
            writeFloat(out, ((NormalizationH3) n).getMu());
        } else if (c == NormalizationZ.class) {
            out.write(4);
            writeFloat(out, ((NormalizationZ) n).getZ());
        } else {
            return false;
        }
        return true;
    }

    /** {@code LMSimilarity.DefaultCollectionModel}, the only one OpenSearch builds. */
    private static boolean defaultCollectionModel(LMSimilarity sim) {
        try {
            Object model = Reflect.inHierarchy(LMSimilarity.class, "collectionModel").get(sim);
            return model != null && model.getClass() == LMSimilarity.DefaultCollectionModel.class;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return false;
        }
    }

    /** {@code LegacyBM25Similarity}'s wrapped {@code BM25Similarity}, or null when unreadable. */
    @SuppressWarnings("deprecation")
    private static BM25Similarity legacyBm25(Similarity sim) {
        try {
            Object bm25 = Reflect.inHierarchy(LegacyBM25Similarity.class, "bm25Similarity").get(sim);
            return bm25 != null && bm25.getClass() == BM25Similarity.class ? (BM25Similarity) bm25 : null;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    private static void writeFloat(ByteArrayOutputStream out, float f) {
        writeInt(out, Float.floatToIntBits(f));
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }
}
