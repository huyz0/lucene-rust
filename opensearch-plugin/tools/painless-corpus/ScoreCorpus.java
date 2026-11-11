/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.sandbox.document.HalfFloatPoint;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.NumericUtils;
import org.lucenerust.opensearch.painless.PainlessIr;
import org.lucenerust.opensearch.painless.PainlessScore;
import org.opensearch.Version;
import org.opensearch.common.lucene.search.function.ScriptScoreQuery;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.indices.breaker.NoneCircuitBreakerService;
import org.opensearch.index.fielddata.IndexFieldDataCache;
import org.opensearch.index.mapper.BooleanFieldMapper;
import org.opensearch.index.mapper.DateFieldMapper;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.painless.PainlessScriptEngine;
import org.opensearch.painless.spi.Allowlist;
import org.opensearch.painless.spi.AllowlistLoader;
import org.opensearch.script.FilterScript;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptException;
import org.opensearch.script.Script;
import org.opensearch.script.ScriptType;
import org.opensearch.search.lookup.SearchLookup;

/**
 * Writes the fixture the native {@code script_score} and {@code script} query nodes are tested
 * against: an index, and for each script of a set the hits and scores OpenSearch's own {@code
 * ScriptScoreQuery} (or {@code ScriptQuery}) gives, or the exception it throws, with the IR,
 * params and doc fields the plugin would send.
 *
 * <p>Layout (little-endian): {@code PSC2}, count, then per script: source, sub-query kind
 * ({@code u8}: 0 {@code body:alpha}, 1 match all, 2 none: a filter script's {@code script}
 * query), boost {@code f32}, native {@code u8}; when native, the IR, params, the context {@code
 * u8} (0 score, 1 score reading {@code _score}, 2 filter) and the fields (count, then name and
 * kind);
 * then the outcome {@code u8}: 0 the hits (count, then doc {@code i32} and score {@code f32}, by
 * doc), 1 the script's exception, 2 another exception: the class (the one Painless's {@code
 * ScriptException} wraps), its message ({@code
 * u8} 1 then the string, or 0 for none) and the script offset it points at ({@code i32}, -1 none).
 *
 * <pre>
 *   java -cp ... ScoreCorpus fixtures/data/painless-score
 * </pre>
 */
public class ScoreCorpus {

    static final String[] BODIES = { "alpha beta gamma", "beta alpha delta alpha", "gamma gamma alpha", "delta beta", "alpha delta beta gamma alpha", "epsilon" };

    public static void main(String[] args) throws Exception {
        Path out = Path.of(args[0]);
        Path indexDir = out.resolve("index");
        if (Files.exists(indexDir)) {
            try (var s = Files.list(indexDir)) {
                for (Path p : (Iterable<Path>) s::iterator) {
                    Files.delete(p);
                }
            }
        }
        Files.createDirectories(indexDir);
        try (FSDirectory dir = FSDirectory.open(indexDir)) {
            IndexWriterConfig cfg = new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE).setUseCompoundFile(false);
            try (IndexWriter w = new IndexWriter(dir, cfg)) {
                for (int i = 0; i < 300; i++) {
                    w.addDocument(doc(i));
                    if (i == 149) {
                        w.commit();
                    }
                }
                w.commit();
            }
            write(dir, out.resolve("expected.bin"));
        }
    }

    static Document doc(int i) {
        Document d = new Document();
        d.add(new TextField("body", BODIES[i % 6], Field.Store.NO));
        d.add(new SortedNumericDocValuesField("likes", (i * 7919L) % 1000));
        if (i % 7 != 0) {
            d.add(new SortedNumericDocValuesField("price", NumericUtils.doubleToSortableLong((i % 50) * 1.75 + 0.5)));
        }
        d.add(new SortedNumericDocValuesField("ratio", NumericUtils.floatToSortableInt((i % 13) * 0.37f - 1.5f)));
        d.add(new SortedNumericDocValuesField("hf", HalfFloatPoint.halfFloatToSortableShort((i % 11) * 0.3f)));
        d.add(new SortedNumericDocValuesField("flag", i % 3 == 0 ? 1 : 0));
        d.add(new SortedSetDocValuesField("tag", new BytesRef("tag" + (i % 5))));
        if (i % 4 == 0) {
            d.add(new SortedSetDocValuesField("tag", new BytesRef("tag" + ((i + 2) % 5))));
        }
        d.add(new SortedNumericDocValuesField("vals", i % 10));
        d.add(new SortedNumericDocValuesField("vals", (i % 7) * 3L));
        d.add(new SortedNumericDocValuesField("vals", 100 - i % 9));
        d.add(new SortedNumericDocValuesField("neg", i - 150));
        d.add(new SortedNumericDocValuesField("d", 1704067200000L + i * 86_400_000L));
        return d;
    }

    record Case(String source, Map<String, Object> params, int sub, float boost) {}

    static List<Case> cases() {
        Map<String, Object> p = new HashMap<>();
        p.put("w", 2.5);
        p.put("t", "tag3");
        p.put("offset", 100);
        p.put("weights", new ArrayList<>(List.of(1, 2.5, 4L)));
        Map<String, Object> m = new HashMap<>();
        m.put("tag0", 0.25);
        m.put("tag2", 3);
        p.put("m", m);
        p.put("origin", 500);
        p.put("scale", 200);
        p.put("decay", 0.5);
        List<Case> c = new ArrayList<>();
        String[] scripts = {
            "doc['likes'].value * params.w + _score",
            "Math.log(2 + doc['likes'].value) * params.w",
            "doc['price'].size() == 0 ? 0 : doc['price'].value / (1 + doc['likes'].value)",
            "double s = 0; for (int i = 0; i < doc['vals'].length; ++i) { s += doc['vals'][i]; } return s;",
            "doc['tag'].value == params.t ? 2 : 1",
            "def x = doc['likes'].value; if (x > 500) { return x / 2; } return x * 1.5 + params.w;",
            "_score * (doc['flag'].value ? 2 : 1)",
            "doc['ratio'].value + doc['hf'].value + 2",
            "double s = 0; for (def v : doc['vals']) { s += v; } return s / doc['vals'].size();",
            "doc['tag'].contains(params.t) ? _score * 3 : _score",
            "params.weights[(int) (doc['likes'].value % 3)] * _score",
            "params.m.getOrDefault(doc['tag'].value, 0.5)",
            "def v = params.m[doc['tag'].value]; return v == null ? 1 : v;",
            "doc['price'].value",
            "doc['neg'].value",
            "doc['d'].value.toInstant().toEpochMilli() / 1e12",
            "doc.containsKey('likes') && !doc['likes'].empty ? doc['likes'].value : 0",
            "String t = doc['tag'].value; return t.length() + t.charAt(3);",
            "Math.max(0, doc['likes'].value - params.offset) / (1.0 + Math.abs(doc['ratio'].value))",
            "_score + doc['likes'].value / 1000.0",
            "doc['nosuch'].value",
            "params._source.size()",
            "int total = 0; for (String t : doc['tag']) { total += t.hashCode() % 7; } return Math.abs(total);",
            "doc['tag'].value =~ /tag[12]/ ? 2 : 1",
            "decayNumericLinear(params.origin, params.scale, 0, params.decay, doc['likes'].value)",
            "decayNumericExp(params.origin, params.scale, 10, params.decay, doc['likes'].value)",
            "decayNumericGauss(params.origin, params.scale, 10, params.decay, doc['likes'].value)",
            "saturation(doc['likes'].value, 10)",
            "sigmoid(doc['likes'].value + 1, 10, 0.6)",
            "randomScore(100)",
            "randomScore(7, 'likes')",
            "randomScore(7, 'tag')",
            "params.doc['likes'].value + params._doc['flag'].value.compareTo(true) + 3",
            "List l = new ArrayList(doc['vals']); Collections.sort(l); return l[1];",
            "doc['vals'].stream().mapToLong(x -> x).sum()",
            "long m = 0; for (long v : doc['vals']) { if (v > m) { m = v; } } return m * _score;",
            "doc['likes'].get(0) + doc['likes'][0]",
            "doc['likes'].value > 500 ? 1 : -1",
            "throw new IllegalArgumentException('nope')",
            "Double.isNaN(_score) ? 0 : (float) (doc['ratio'].value * 1.1f) + 3",
            // Counted in a batch: a keyword's and a boolean's value counts.
            "doc['tag'].size() + doc['flag'].size() + _score",
            // Refused: a params collection changed (Painless shares it across segments), and
            // params changed through a second variable.
            "params.weights.add(1); return params.weights.size()",
            "def p = params; def q = p; p['w'] = 5; return q.w",
            // Past a document's values: ScriptDocValues's reused array (a stale value, a never
            // written slot, or an index error).
            "doc['vals'].size() == 0 ? 0 : doc['vals'][1] + doc['vals'].get(2)",
            "doc['tag'][2].length() + 1",
            "doc['flag'][3] ? 2 : 1",
            // Refused: a map's key set changed (Java's is a live view).
            "Map m = ['a': 1, 'b': 2]; m.keySet().remove('a'); return m.size()",
            // A field some documents lack, iterated: no values, so nothing to read.
            "double s = 1; for (def v : doc['price']) { s += v; } return s;",
        };
        for (String s : scripts) {
            c.add(new Case(s, p, 0, 1f));
        }
        c.add(new Case("doc['likes'].value * params.w + _score", p, 1, 1f));
        c.add(new Case("_score + doc['likes'].value / 1000.0", p, 0, 2f));
        String[] filters = {
            "doc['likes'].value > params.offset * 5",
            "doc['tag'].value == params.t",
            "doc['price'].size() > 0 && doc['price'].value < 20",
            "def s = 0; for (def v : doc['vals']) { s += v; } return s > 120;",
            "doc['flag'].value",
            "doc['ratio'].value > 0 || doc['hf'].value > 1.5",
            "params.m.containsKey(doc['tag'].value)",
            "return doc['likes'].value % 3 == 0",
            "def x = doc['likes'].value; return x > 990 ? true : false;",
            // A filter's params are the script's alone.
            "params.doc == null && doc['likes'].value < 100",
            "doc['tag'].contains(params.t) && doc['neg'].value > 0",
            "for (long v : doc['vals']) { if (v > 98) { return true; } } return false;",
            // Errors: a missing value, an unmapped field, the script's own.
            "doc['price'].value > 10",
            "doc['nosuch'].value > 0",
            "throw new IllegalArgumentException('nope')",
            // Refused: _source.
            "params._source.size() > 0",
        };
        for (String s : filters) {
            c.add(new Case(s, p, 2, 1f));
        }
        c.add(new Case("doc['likes'].value < 500", p, 2, 3f));
        return c;
    }

    static void write(FSDirectory dir, Path expected) throws Exception {
        List<Allowlist> allowlists = new ArrayList<>(Allowlist.BASE_ALLOWLISTS);
        allowlists.add(AllowlistLoader.loadFromResourceFiles(Allowlist.class, "org.opensearch.score.txt"));
        PainlessScriptEngine engine = new PainlessScriptEngine(
            Settings.EMPTY,
            Map.of(ScoreScript.CONTEXT, allowlists, FilterScript.CONTEXT, Allowlist.BASE_ALLOWLISTS)
        );
        var scriptQuery = Class.forName("org.opensearch.index.query.ScriptQueryBuilder$ScriptQuery")
            .getDeclaredConstructor(Script.class, FilterScript.LeafFactory.class, String.class);
        scriptQuery.setAccessible(true);
        PainlessScore.initialize(engine);
        Map<String, MappedFieldType> fts = new LinkedHashMap<>();
        fts.put("likes", new NumberFieldMapper.NumberFieldType("likes", NumberFieldMapper.NumberType.LONG));
        fts.put("vals", new NumberFieldMapper.NumberFieldType("vals", NumberFieldMapper.NumberType.INTEGER));
        fts.put("neg", new NumberFieldMapper.NumberFieldType("neg", NumberFieldMapper.NumberType.LONG));
        fts.put("price", new NumberFieldMapper.NumberFieldType("price", NumberFieldMapper.NumberType.DOUBLE));
        fts.put("ratio", new NumberFieldMapper.NumberFieldType("ratio", NumberFieldMapper.NumberType.FLOAT));
        fts.put("hf", new NumberFieldMapper.NumberFieldType("hf", NumberFieldMapper.NumberType.HALF_FLOAT));
        fts.put("flag", new BooleanFieldMapper.BooleanFieldType("flag"));
        fts.put("tag", new KeywordFieldMapper.KeywordFieldType("tag"));
        fts.put("d", new DateFieldMapper.DateFieldType("d"));
        MapperService ms = mock(MapperService.class);
        for (var e : fts.entrySet()) {
            when(ms.fieldType(e.getKey())).thenReturn(e.getValue());
        }
        SearchLookup lookup = new SearchLookup(
            ms,
            (ft, sup) -> ft.fielddataBuilder("idx", sup).build(new IndexFieldDataCache.None(), new NoneCircuitBreakerService())
        );
        int matched = 0, threw = 0, refused = 0;
        Rec rec = new Rec();
        List<Case> cases = cases();
        rec.out.writeBytes("PSC2".getBytes(StandardCharsets.US_ASCII));
        rec.i32(cases.size());
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setQueryCache(null);
            for (Case c : cases) {
                rec.str(c.source());
                rec.u8(c.sub());
                rec.i32(Float.floatToIntBits(c.boost()));
                boolean filter = c.sub() == 2;
                Object compiled = filter ? PainlessScore.filter(c.source()) : PainlessScore.score(c.source());
                Script script = new Script(ScriptType.INLINE, "painless", c.source(), c.params());
                Query q;
                int context;
                if (filter) {
                    FilterScript.Factory factory = engine.compile("fixture", c.source(), FilterScript.CONTEXT, Map.of());
                    q = (Query) scriptQuery.newInstance(script, factory.newFactory(script.getParams(), lookup), null);
                    context = PainlessScore.FILTER;
                } else {
                    ScoreScript.Factory factory = engine.compile("fixture", c.source(), ScoreScript.CONTEXT, Map.of());
                    ScoreScript.LeafFactory leaf = factory.newFactory(c.params(), lookup, searcher);
                    Query sub = c.sub() == 0 ? new TermQuery(new Term("body", "alpha")) : new MatchAllDocsQuery();
                    q = new ScriptScoreQuery(sub, script, leaf, null, "idx", 3, Version.CURRENT);
                    context = leaf.needs_score() ? PainlessScore.SCORE_WITH_SCORE : PainlessScore.SCORE;
                }
                if (compiled instanceof PainlessIr.Compiled ir) {
                    rec.u8(1);
                    rec.bytes(ir.ir());
                    rec.bytes(PainlessIr.constant(c.params()));
                    rec.u8(context);
                    rec.i32(ir.docFields().size());
                    for (String f : ir.docFields()) {
                        rec.str(f);
                        rec.u8(PainlessScore.fieldKind(fts.get(f)));
                    }
                } else {
                    rec.u8(0);
                    refused++;
                    System.err.println("refused: " + c.source() + ": " + compiled);
                }
                if (c.boost() != 1f) {
                    q = new BoostQuery(q, c.boost());
                }
                try {
                    TopDocs td = searcher.search(q, 1000);
                    ScoreDoc[] hits = td.scoreDocs.clone();
                    Arrays.sort(hits, (a, b) -> Integer.compare(a.doc, b.doc));
                    rec.u8(0);
                    rec.i32(hits.length);
                    for (ScoreDoc h : hits) {
                        rec.i32(h.doc);
                        rec.i32(Float.floatToIntBits(h.score));
                    }
                    matched++;
                } catch (Exception e) {
                    Throwable t = e;
                    ScriptException.Position pos = null;
                    boolean wrapped = false;
                    while (t.getCause() != null && t.getClass().getName().startsWith("org.opensearch.script")) {
                        if (t instanceof ScriptException se) {
                            wrapped = true;
                            pos = se.getPos();
                        }
                        t = t.getCause();
                    }
                    rec.u8(wrapped ? 1 : 2);
                    rec.str(t.getClass().getName());
                    if (t.getMessage() == null) {
                        rec.u8(0);
                    } else {
                        rec.u8(1);
                        rec.str(t.getMessage());
                    }
                    rec.i32(pos == null ? -1 : pos.offset);
                    threw++;
                }
            }
        }
        try (FileOutputStream f = new FileOutputStream(expected.toFile())) {
            rec.out.writeTo(f);
        }
        System.err.printf("%d scored, %d threw, %d not native%n", matched, threw, refused);
    }

    static final class Rec {
        final ByteArrayOutputStream out = new ByteArrayOutputStream();

        void u8(int v) {
            out.write(v);
        }

        void i32(int v) {
            out.write(v);
            out.write(v >>> 8);
            out.write(v >>> 16);
            out.write(v >>> 24);
        }

        void bytes(byte[] b) {
            i32(b.length);
            out.writeBytes(b);
        }

        void str(String s) {
            bytes(s.getBytes(StandardCharsets.UTF_8));
        }
    }
}
