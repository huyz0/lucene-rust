/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LateInteractionField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.lucenerust.opensearch.tensor.MaxSimQuery;
import org.lucenerust.opensearch.tensor.MultiVectorEncoding;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.stream.Stream;

/**
 * The {@code max_sim} shapes in {@link NativeSelfTest}: a {@code multi_vector} field in each
 * encoding (written by {@link MultiVectorEncoding}, the {@code float} one also by Lucene's {@code
 * LateInteractionField}) over an NRT index with deletions and documents without a value, each
 * {@link MaxSimQuery} encoded and compared by {@code NativeSelfTest.compare} (the native kernels
 * against the Java scorer, hits and scores).
 */
final class NativeSelfTestTensor {
    private NativeSelfTestTensor() {}

    static final int DIM = 24;
    static final String[] FIELDS = { "mv_float", "mv_float16", "mv_int8", "mv_lucene" };
    private static final MultiVectorEncoding[] ENCODINGS = { MultiVectorEncoding.FLOAT, MultiVectorEncoding.FLOAT16, MultiVectorEncoding.INT8 };

    /** {@code golden.rs}'s {@code FLOAT}, {@code FLOAT16}, {@code INT8}. */
    private static final String[] GOLDEN = {
        "030000000000fe4200002040000020c000000080000000000000000005000000030000800000000000e07f4700e07fc70000803f000080bf0000003f000020bf000040bf00000000000000be000000bf0000003e000000bf000080be0000803e00000080000000000000c03e0000c0be0000803e0000003f000040bf0000003f00000000000080be0000403f0000003e000020bf0000803f0000803e000000be0000a03f0000c03e000000bf0000c03f0000003f000000800000e03f000000000000c0be000000400000003e000040bf",
        "ffffffff0300000011000000f05700800000ff7b00bc00ba00b800b4000000340038003a003c003d003e003f004100000080fffb00380000003000340036003800000030003400360038000000c100000000003c00b900b000b8008000b600ba00b400b900b000b8008000b60040000000000000000000000000000000000000000000000000000000000000003000000000000000000000000000000000000000000000000000000000000000ba000000000000000000000000000000000000000000000000000000000000",
        "feffffff03000000110000000000803f0000803f01000000e4f100440402013c0683c13b0402813b0402013b0683413b0683c13b0402813b0683c13b0402013c8542213c0683413c87c3613c0402813c7f00057f81818181002a7f7f7f7f7f7f0300fd814000207f7f55001520262a00fd000000b1eb81008181c096f0cd00e57f00000000000000000000000000000008000000000000000000000000000000d0000000000000000000000000000000"
    };

    /** {@code golden.rs}'s tokens. */
    private static float[][] golden() {
        float[][] t = new float[17][];
        t[0] = new float[] { 127f, 2.5f, -2.5f };
        t[1] = new float[] { -0f, 0f, 0f };
        t[2] = new float[] { Float.intBitsToFloat(5), -Float.intBitsToFloat(3), 0f };
        t[3] = new float[] { 65504f, -65504f, 1f };
        for (int i = 4; i < 17; i++) {
            t[i] = new float[] { i * 0.25f - 2f, (i % 5) * 0.125f, -(((i * 3) % 7) / 8f) };
        }
        return t;
    }

    /** Queries encoded and compared. */
    static int encoded;

    private static float[][] tokens(Random r, int n) {
        float[][] t = new float[n][DIM];
        for (float[] v : t) {
            double norm = 0;
            for (int k = 0; k < DIM; k++) {
                v[k] = (float) r.nextGaussian();
                norm += v[k] * v[k];
            }
            for (int k = 0; k < DIM; k++) {
                v[k] = (float) (v[k] / Math.sqrt(norm));
            }
        }
        return t;
    }

    static void run(Random r, BiConsumer<Boolean, String> check, NativeSelfTestExtended.Compare compare) throws Exception {
        // The Java encoder writes Lucene's bytes for float, and decodes every encoding back.
        float[][] sample = tokens(r, 21);
        check.accept(
            Arrays.equals(MultiVectorEncoding.FLOAT.encode(sample, DIM), LateInteractionField.encode(sample).bytes),
            "multi_vector float is Lucene's LateInteractionField layout"
        );
        for (MultiVectorEncoding e : ENCODINGS) {
            byte[] bytes = e.encode(sample, DIM);
            MultiVectorEncoding.Blocks blocks = MultiVectorEncoding.decode(bytes, 0, bytes.length);
            boolean same = blocks.dim() == DIM && blocks.tokens() == sample.length;
            for (int i = 0; same && i < sample.length; i++) {
                for (int k = 0; k < DIM; k++) {
                    float got = blocks.values()[(i / 16) * DIM * 16 + k * 16 + i % 16];
                    same &= Math.abs(got - sample[i][k]) <= (e == MultiVectorEncoding.INT8 ? 0.01f : e == MultiVectorEncoding.FLOAT16 ? 1e-3f : 0f);
                }
            }
            check.accept(same, "multi_vector " + e.getName() + " decodes back to its tokens");
            for (int cut : new int[] { 1, 3, bytes.length - 1 }) {
                boolean refused;
                try {
                    MultiVectorEncoding.decode(bytes, 0, cut);
                    refused = false;
                } catch (IllegalArgumentException ex) {
                    refused = true;
                }
                check.accept(refused, "multi_vector " + e.getName() + " refuses a value cut to " + cut + " bytes");
            }
        }
        // The same bytes as lucene-tensor's encode (crates/lucene-tensor/tests/golden.rs).
        for (int e = 0; e < ENCODINGS.length; e++) {
            String hex = HexFormat.of().formatHex(ENCODINGS[e].encode(golden(), 3));
            check.accept(hex.equals(GOLDEN[e]), "multi_vector " + ENCODINGS[e].getName() + " writes lucene-tensor's bytes: " + hex);
        }
        Path dir = Files.createTempDirectory("lucene-rust-selftest-tensor");
        NativeReaders readers = new NativeReaders();
        try (FSDirectory d = FSDirectory.open(dir); IndexWriter w = new IndexWriter(d, new IndexWriterConfig(new StandardAnalyzer()))) {
            DirectoryReader reader = null;
            int id = 0;
            for (int round = 0; round < 3; round++) {
                for (int i = 0; i < 150 + r.nextInt(100); i++, id++) {
                    Document doc = new Document();
                    doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
                    doc.add(new StringField("tag", id % 3 == 0 ? "a" : "b", Field.Store.NO));
                    if (r.nextInt(8) != 0) {
                        float[][] t = tokens(r, 1 + r.nextInt(40));
                        for (int f = 0; f < ENCODINGS.length; f++) {
                            doc.add(new BinaryDocValuesField(FIELDS[f], new BytesRef(ENCODINGS[f].encode(t, DIM))));
                        }
                        doc.add(new LateInteractionField("mv_lucene", t));
                    }
                    w.addDocument(doc);
                }
                for (int i = 0; i < 15; i++) {
                    w.deleteDocuments(new Term("id", Integer.toString(r.nextInt(id))));
                }
                w.commit();
                DirectoryReader next = reader == null ? DirectoryReader.open(w) : DirectoryReader.openIfChanged(reader, w);
                if (next != null) {
                    if (reader != null) {
                        reader.close();
                    }
                    reader = next;
                }
                Query tagA = new TermQuery(new Term("tag", "a"));
                List<Query> queries = new ArrayList<>();
                for (String field : FIELDS) {
                    for (int q = 0; q < 4; q++) {
                        float[][] t = tokens(r, 1 + r.nextInt(32));
                        Query exists = new FieldExistsQuery(field);
                        queries.add(new MaxSimQuery(exists, field, t));
                        queries.add(new MaxSimQuery(MatchAllDocsQuery.INSTANCE, field, t));
                        queries.add(
                            new MaxSimQuery(new BooleanQuery.Builder().add(exists, Occur.FILTER).add(tagA, Occur.FILTER).build(), field, t)
                        );
                        queries.add(new BooleanQuery.Builder().add(new BoostQuery(new MaxSimQuery(exists, field, t), 0.5f), Occur.MUST).add(tagA, Occur.SHOULD).build());
                    }
                }
                for (Query q : queries) {
                    QueryEncoder.Encoded enc = QueryEncoder.encode(q, f -> true);
                    check.accept(enc.blob() != null, "max_sim query does not encode: " + q + " (" + enc.fallbackReason() + ")");
                    encoded += enc.blob() != null ? 1 : 0;
                }
                compare.run("max_sim round " + round + " (" + reader.leaves().size() + " segments, " + reader.numDeletedDocs() + " deleted)", reader, readers, queries);
            }
            reader.close();
            check.accept(readers.openCount() == 0, "max_sim: no native readers left open");
        }
        try (Stream<Path> s = Files.walk(dir)) {
            s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }
}
