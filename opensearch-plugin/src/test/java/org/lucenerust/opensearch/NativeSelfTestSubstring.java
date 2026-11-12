/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.stream.Stream;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.MultiTermQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.automaton.RegExp;
import org.lucenerust.opensearch.engine.SubstringTestIndex;
import org.lucenerust.opensearch.substring.RustWildcardFieldMapper;
import org.lucenerust.opensearch.substring.RustWildcardFieldMapper.RustWildcardFieldType;
import org.lucenerust.opensearch.substring.RustWildcardQuery;

/**
 * The {@code compact_wildcard} shapes in {@link NativeSelfTest}: an index the Rust engine's writer wrote
 * ({@code Substring90} blocks, flushed, then merged by a writer reopened on it) and one Lucene
 * wrote (plain BINARY, as under OpenSearch's engine). Over each, the field reads back value for value
 * through Lucene, every query of the field type matches the documents whose value its test accepts,
 * and each encoded query's native hits and scores equal Lucene's ({@code NativeSelfTest.compare}).
 */
final class NativeSelfTestSubstring {
    private NativeSelfTestSubstring() {}

    /** Queries encoded and compared. */
    static int encoded;

    private static final String[] WORDS = {
        "Connection", "timeout", "java.net.SocketTimeoutException:", "read", "timed", "out", "ERROR", "error",
        "WARN", "disk", "full", "blk_-16089", "10.250.19.102", "postdrop", "İstanbul", "Kelvin", "parity",
        "corrected", "a*b", "what?", "back\\slash", "/path/", "[x]", "a|b", "", "ab", "Ünïcødé" };

    static List<String> values(Random r, int n) {
        List<String> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (i % 11 == 3) {
                out.add(null);
                continue;
            }
            if (i % 37 == 5) {
                out.add("");
                continue;
            }
            int words = r.nextInt(8);
            StringBuilder b = new StringBuilder(Integer.toString(i % 97));
            for (int k = 0; k < words; k++) {
                b.append(' ').append(WORDS[r.nextInt(WORDS.length)]);
            }
            out.add(b.toString());
        }
        return out;
    }

    @SuppressWarnings("deprecation")
    static List<Query> queries(RustWildcardFieldType ft) {
        MultiTermQuery.RewriteMethod m = MultiTermQuery.CONSTANT_SCORE_REWRITE;
        List<Query> q = new ArrayList<>();
        for (boolean ci : new boolean[] { false, true }) {
            for (String p : new String[] {
                "*timeout*", "*TIMEOUT*", "1? *", "*corrected", "*a\\*b*", "*what\\?*", "*back\\\\slash*", "*", "?", "??",
                "*istanbul*", "*kelvin*", "*conn*time*", "4? *out*", "*/path/*", "*[x]*", "*a|b*", "", "5", "ünï*", "*Ü*" }) {
                q.add(ft.wildcardQuery(p, m, ci, null));
            }
            for (String p : new String[] { "1", "Connection", "conn", "9 ERROR" }) {
                q.add(ft.prefixQuery(p, m, ci, null));
            }
        }
        for (String t : new String[] { "5", "5 timeout", "", "a*b", "1 ERROR" }) {
            q.add(ft.termQuery(t, null));
            q.add(ft.termQueryCaseInsensitive(t, null));
        }
        q.add(ft.termsQuery(List.of("5", "6", "", "7 error", "5"), null));
        for (String re : new String[] {
            ".*[Tt]ime(d|out).*", "1[0-9] .*error.*", ".*blk_-?[0-9]+.*", "5", ".*", "", ".*a\\|b.*", "[0-9]+ (WARN|ERROR).*",
            ".*~(x).*", ".*<1-20> .*" }) {
            q.add(ft.regexpQuery(re, RegExp.ALL, 0, 10000, m, null));
            q.add(ft.regexpQuery(re, RegExp.ALL, RegExp.ASCII_CASE_INSENSITIVE, 10000, m, null));
        }
        // Not RegExp.ALL: Lucene only.
        q.add(ft.regexpQuery(".*time.*", RegExp.NONE, 0, 10000, m, null));
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        b.add(ft.wildcardQuery("*error*", m, true, null), BooleanClause.Occur.MUST);
        b.add(ft.wildcardQuery("*timeout*", m, false, null), BooleanClause.Occur.MUST_NOT);
        q.add(b.build());
        return q;
    }

    /** Whether a document with {@code value} matches {@code q}, by the field type's own tests. */
    static boolean expected(Query q, String value) {
        if (q instanceof RustWildcardQuery w) {
            return value != null && w.matches(value);
        }
        if (q instanceof FieldExistsQuery) {
            return value != null;
        }
        if (q instanceof MatchNoDocsQuery) {
            return false;
        }
        if (q instanceof ConstantScoreQuery cs) {
            return expected(cs.getQuery(), value);
        }
        if (q instanceof BooleanQuery bq) {
            boolean required = true;
            boolean hasRequired = false;
            boolean hasShould = false;
            boolean anyShould = false;
            for (BooleanClause c : bq.clauses()) {
                boolean hit = expected(c.query(), value);
                switch (c.occur()) {
                    case MUST, FILTER -> {
                        hasRequired = true;
                        required &= hit;
                    }
                    case MUST_NOT -> required &= hit == false;
                    case SHOULD -> {
                        hasShould = true;
                        anyShould |= hit;
                    }
                }
            }
            return required && (hasShould == false || hasRequired || anyShould);
        }
        throw new IllegalArgumentException("no expectation for " + q);
    }

    static void run(Random r, BiConsumer<Boolean, String> check, NativeSelfTestExtended.Compare compare) throws Exception {
        List<String> values = values(r, 2600);
        RustWildcardFieldType ft = new RustWildcardFieldType("msg");
        List<Query> queries = queries(ft);
        for (String which : new String[] { "rust", "lucene" }) {
            Path dir = Files.createTempDirectory("lucene-rust-selftest-substring-" + which);
            try {
                if (which.equals("rust")) {
                    SubstringTestIndex.write(dir, "msg", values, 700, 2);
                } else {
                    try (FSDirectory d = FSDirectory.open(dir); IndexWriter w = new IndexWriter(d, new IndexWriterConfig())) {
                        for (int i = 0; i < values.size(); i++) {
                            Document doc = new Document();
                            doc.add(new StringField("id", Integer.toString(i), Field.Store.YES));
                            if (values.get(i) != null) {
                                doc.add(RustWildcardFieldMapper.valueField("msg", values.get(i)));
                            }
                            w.addDocument(doc);
                            if ((i + 1) % 700 == 0) {
                                w.commit();
                            }
                        }
                        w.commit();
                    }
                }
                NativeReaders readers = new NativeReaders();
                try (FSDirectory d = FSDirectory.open(dir); DirectoryReader reader = DirectoryReader.open(d)) {
                    // The documents' values, by doc id across the reader, read back through Lucene.
                    String[] byDoc = new String[reader.maxDoc()];
                    boolean same = true;
                    for (LeafReaderContext leaf : reader.leaves()) {
                        FieldInfo fi = leaf.reader().getFieldInfos().fieldInfo("msg");
                        String format = fi == null ? null : fi.getAttribute("PerFieldDocValuesFormat.format");
                        String want = which.equals("rust") ? "Substring90" : "Lucene90";
                        check.accept(want.equals(format), "substring " + which + ": a segment's msg is in " + format + ", not " + want);
                        StoredFields stored = leaf.reader().storedFields();
                        BinaryDocValues dv = leaf.reader().getBinaryDocValues("msg");
                        for (int doc = 0; doc < leaf.reader().maxDoc(); doc++) {
                            int id = Integer.parseInt(stored.document(doc).get("id"));
                            String got = null;
                            if (dv != null && dv.advanceExact(doc)) {
                                got = dv.binaryValue().utf8ToString();
                            }
                            same &= java.util.Objects.equals(got, values.get(id));
                            byDoc[leaf.docBase + doc] = got;
                        }
                        if (dv != null) {
                            // And iterated: every value once, in doc order.
                            BinaryDocValues it = leaf.reader().getBinaryDocValues("msg");
                            int count = 0;
                            for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
                                same &= it.binaryValue().utf8ToString().equals(byDoc[leaf.docBase + doc]);
                                count++;
                            }
                            int withValue = 0;
                            for (int doc = 0; doc < leaf.reader().maxDoc(); doc++) {
                                withValue += byDoc[leaf.docBase + doc] != null ? 1 : 0;
                            }
                            same &= count == withValue;
                        }
                    }
                    check.accept(same, "substring " + which + ": every value reads back");
                    if (which.equals("rust")) {
                        check.accept(reader.leaves().size() == 2, "substring rust: merged to 2 segments, got " + reader.leaves().size());
                    }
                    IndexSearcher searcher = new IndexSearcher(reader);
                    List<Query> native_ = new ArrayList<>();
                    for (Query q : queries) {
                        int want = 0;
                        for (String v : byDoc) {
                            want += expected(q, v) ? 1 : 0;
                        }
                        int got = searcher.count(q);
                        check.accept(got == want, "substring " + which + " " + q + ": " + got + " hits, " + want + " by its test");
                        QueryEncoder.Encoded enc = QueryEncoder.encode(q, f -> true);
                        boolean javaOnly = q.toString().contains(".*time.*");
                        check.accept((enc.blob() != null) != javaOnly, "substring " + q + " encodes: " + enc.fallbackReason());
                        if (enc.blob() != null) {
                            native_.add(q);
                            encoded++;
                        }
                    }
                    compare.run("substring " + which + " (" + reader.leaves().size() + " segments)", reader, readers, native_);
                }
                check.accept(readers.openCount() == 0, "substring: no native readers left open");
            } finally {
                try (Stream<Path> s = Files.walk(dir)) {
                    s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
                }
            }
        }
    }
}
