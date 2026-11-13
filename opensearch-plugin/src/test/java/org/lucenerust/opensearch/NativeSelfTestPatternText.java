/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.core.KeywordAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.lucenerust.opensearch.patterntext.PatternPhraseQuery;
import org.lucenerust.opensearch.patterntext.PatternTextAnalysis;
import org.lucenerust.opensearch.patterntext.PatternTextFieldMapper;
import org.lucenerust.opensearch.patterntext.PatternTextFormat;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.stream.Stream;

/**
 * The {@code pattern_text} shapes in {@link NativeSelfTest}: {@link PatternTextFormat} writes
 * {@code lucene-pattern-text}'s bytes; over an NRT index with deletions, each {@link
 * PatternPhraseQuery} matches what a positional phrase on the same messages matches (in Java, with
 * and without the template shortcut, exact and with slop), and each exact one is encoded and
 * compared by {@code NativeSelfTest.compare} (native hits and scores against Java's).
 */
final class NativeSelfTestPatternText {
    private NativeSelfTestPatternText() {}

    /** {@code format.rs}'s {@code GOLDEN}: message, template's UTF-8, stored variables, in hex. */
    private static final String[][] GOLDEN = {
        {
            "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-16 src: /10.250.19.102:54106",
            "022002200220494e464f206466732e446174614e6f64652444617461586365697665723a20526563656976696e6720626c6f636b2002207372633a2002",
            "06303831313039063230333531380331343307626c6b5f2d3136142f31302e3235302e31392e3130323a3534313036" },
        { "  two  spaces ", "202074776f202073706163657320", "" },
        { "unicode ünïcødé 3ms \u0002 x", "756e69636f646520c3bc6ec3af63c3b864c3a9200220022078", "03336d730102" },
        { "", "", "" } };

    /** Queries encoded and compared. */
    static int encoded;

    private static final String[] PHRASES = {
        "connection closed by",
        "failed password for root",
        "failed password",
        "instruction cache parity error corrected",
        "parity error",
        "closed by peer",
        "receiving block",
        "spaced message",
        "password for invalid user",
        "no such phrase anywhere",
        "user admin0 from",
        "admin1 from",
        "port 40009",
        "connection 3 closed",
        "connection 3 closed by peer",
        "closed by 10.0.1.1" };

    private static String message(int i) {
        String ip = "10.0." + (i % 7) + "." + (i % 13);
        return switch (i % 8) {
            case 0 -> String.format("Dec 10 06:55:%02d sshd[%d]: Connection closed by %s [preauth]", i % 60, 24000 + i, ip);
            case 1 -> "Failed password for root from " + ip + " port " + (40000 + i) + " ssh2";
            case 2 -> "Failed password for invalid user admin" + (i % 3) + " from " + ip;
            case 3 -> "instruction cache parity error corrected";
            case 4 -> "instruction cache parity error";
            case 5 -> "connection " + (i % 5) + " closed by peer";
            case 6 -> "Receiving block blk_" + (i * 31) + " src: /" + ip + ":50010";
            default -> "  spaced   message " + i + "  ";
        };
    }

    private static Document document(int id, String m) {
        FieldType docsOnly = new FieldType();
        docsOnly.setTokenized(true);
        docsOnly.setOmitNorms(true);
        docsOnly.setIndexOptions(IndexOptions.DOCS);
        docsOnly.freeze();
        Document doc = new Document();
        doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
        doc.add(new Field("message", m, docsOnly));
        PatternTextFormat.Split s = PatternTextFormat.split(m);
        doc.add(new SortedDocValuesField("message" + PatternTextFieldMapper.TEMPLATE_SUFFIX, new BytesRef(s.template())));
        if (s.args().isEmpty() == false) {
            doc.add(new StoredField("message" + PatternTextFieldMapper.ARGS_SUFFIX, PatternTextFormat.encodeArgs(s.args())));
        }
        doc.add(new StoredField("after", m));
        doc.add(new TextField("positional", m, Field.Store.NO));
        return doc;
    }

    /** {@code text}'s phrase on {@code field}, analyzed by {@code a}: positions as the analyzer gives them. */
    private static PhraseQuery phrase(Analyzer a, String field, String text, int slop) throws Exception {
        PhraseQuery.Builder b = new PhraseQuery.Builder().setSlop(slop);
        try (var ts = a.tokenStream(field, text)) {
            var term = ts.addAttribute(org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute.class);
            var inc = ts.addAttribute(org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute.class);
            ts.reset();
            int pos = -1;
            while (ts.incrementToken()) {
                pos += inc.getPositionIncrement();
                b.add(new Term(field, BytesRef.deepCopyOf(term.getBytesRef())), pos);
            }
            ts.end();
        }
        return b.build();
    }

    private static PhraseQuery on(PhraseQuery p, String field) {
        PhraseQuery.Builder b = new PhraseQuery.Builder().setSlop(p.getSlop());
        for (int i = 0; i < p.getTerms().length; i++) {
            b.add(new Term(field, p.getTerms()[i].bytes()), p.getPositions()[i]);
        }
        return b.build();
    }

    static void run(Random r, BiConsumer<Boolean, String> check, NativeSelfTestExtended.Compare compare) throws Exception {
        HexFormat hex = HexFormat.of();
        for (String[] g : GOLDEN) {
            PatternTextFormat.Split s = PatternTextFormat.split(g[0]);
            String t = hex.formatHex(s.template().getBytes(StandardCharsets.UTF_8));
            String a = hex.formatHex(PatternTextFormat.encodeArgs(s.args()));
            check.accept(t.equals(g[1]) && a.equals(g[2]), "pattern_text writes lucene-pattern-text's bytes for " + g[0] + ": " + t + " " + a);
            byte[] bytes = PatternTextFormat.encodeArgs(s.args());
            check.accept(
                PatternTextFormat.join(s.template(), PatternTextFormat.decodeArgs(bytes, 0, bytes.length)).equals(g[0]),
                "pattern_text rebuilds " + g[0]
            );
        }
        String long300 = "9".repeat(300);
        byte[] longArgs = PatternTextFormat.encodeArgs(PatternTextFormat.split("id " + long300 + " done").args());
        check.accept(hex.formatHex(longArgs, 0, 4).equals("ac023939"), "pattern_text writes a two-byte length");
        for (byte[] bad : new byte[][] { { 5, 'a' }, { (byte) 0x80 }, { (byte) 0xff, (byte) 0xff, (byte) 0xff, (byte) 0xff, (byte) 0xff, 1 } }) {
            boolean refused;
            try {
                PatternTextFormat.decodeArgs(bad, 0, bad.length);
                refused = false;
            } catch (IllegalArgumentException e) {
                refused = true;
            }
            check.accept(refused, "pattern_text refuses corrupt variables " + hex.formatHex(bad));
        }

        Analyzer standard = new StandardAnalyzer();
        Analyzer stop = new StandardAnalyzer(new CharArraySet(List.of("by", "for", "from"), true));
        check.accept(PatternTextAnalysis.splitsOnSpaces(standard, "message"), "the standard analyzer allows the template shortcut");
        check.accept(PatternTextAnalysis.splitsOnSpaces(stop, "message"), "a stop filter allows the template shortcut");
        check.accept(PatternTextAnalysis.splitsOnSpaces(new KeywordAnalyzer(), "message") == false, "the keyword analyzer does not");
        check.accept(
            PatternTextAnalysis.splitsOnSpaces(new org.apache.lucene.analysis.shingle.ShingleAnalyzerWrapper(standard, 2), "message") == false,
            "a shingle filter does not"
        );
        Path synDir = Files.createTempDirectory("lucene-rust-selftest-pattern-syn");
        Files.writeString(synDir.resolve("syn.txt"), "user admin1 => superuser\n");
        Analyzer synonyms = org.apache.lucene.analysis.custom.CustomAnalyzer.builder(synDir)
            .withTokenizer("standard")
            .addTokenFilter("lowercase")
            .addTokenFilter("synonymGraph", "synonyms", "syn.txt")
            .build();
        check.accept(PatternTextAnalysis.splitsOnSpaces(synonyms, "message") == false, "a synonym filter does not, whatever its rules");

        for (Analyzer analyzer : new Analyzer[] { standard, stop }) {
            String which = analyzer == standard ? "standard" : "stop";
            Path dir = Files.createTempDirectory("lucene-rust-selftest-pattern-text");
            NativeReaders readers = new NativeReaders();
            try (FSDirectory d = FSDirectory.open(dir); IndexWriter w = new IndexWriter(d, new IndexWriterConfig(analyzer))) {
                DirectoryReader reader = null;
                List<String> messages = new ArrayList<>();
                for (int round = 0; round < 3; round++) {
                    for (int i = 0; i < 150 + r.nextInt(100); i++) {
                        String m = message(messages.size());
                        w.addDocument(document(messages.size(), m));
                        messages.add(m);
                    }
                    for (int i = 0; i < 10; i++) {
                        w.deleteDocuments(new Term("id", Integer.toString(r.nextInt(messages.size()))));
                    }
                    w.commit();
                    DirectoryReader next = reader == null ? DirectoryReader.open(w) : DirectoryReader.openIfChanged(reader, w);
                    if (next != null) {
                        if (reader != null) {
                            reader.close();
                        }
                        reader = next;
                    }
                    // every message rebuilds from its template and variables, as derived source reads it
                    boolean rebuilt = true;
                    for (LeafReaderContext leaf : reader.leaves()) {
                        StoredFields stored = leaf.reader().storedFields();
                        for (int doc = 0; doc < leaf.reader().maxDoc(); doc++) {
                            String m = PatternTextFieldMapper.message(leaf.reader(), "message", doc);
                            rebuilt &= m != null && m.equals(stored.document(doc).get("after"));
                        }
                    }
                    check.accept(rebuilt, "pattern_text " + which + " round " + round + ": every message rebuilds");
                    IndexSearcher searcher = new IndexSearcher(reader);
                    List<Query> queries = new ArrayList<>();
                    for (String text : PHRASES) {
                        for (int slop : new int[] { 0, 2 }) {
                            PhraseQuery p = phrase(analyzer, "message", text, slop);
                            int want = searcher.count(on(p, "positional"));
                            for (boolean parts : new boolean[] { true, false }) {
                                Query q = PatternPhraseQuery.of("message", p, analyzer, slop == 0, parts);
                                int got = searcher.count(q);
                                check.accept(got == want, "pattern_text " + which + " [" + text + "]~" + slop + " parts=" + parts + ": " + got + " hits, positional " + want);
                                if (slop == 0) {
                                    queries.add(q);
                                    queries.add(new BooleanQuery.Builder().add(q, Occur.MUST).add(new TermQuery(new Term("id", "7")), Occur.SHOULD).build());
                                } else {
                                    check.accept(QueryEncoder.encode(q, f -> true).blob() == null, "a phrase with slop runs in Java");
                                }
                            }
                        }
                    }
                    // stacked terms at one position match either
                    MultiPhraseQuery.Builder mp = new MultiPhraseQuery.Builder();
                    mp.add(new Term[] { new Term("message", "3"), new Term("message", "4") }, 0);
                    mp.add(new Term[] { new Term("message", "closed") }, 1);
                    MultiPhraseQuery multi = mp.build();
                    MultiPhraseQuery.Builder mpp = new MultiPhraseQuery.Builder();
                    mpp.add(new Term[] { new Term("positional", "3"), new Term("positional", "4") }, 0);
                    mpp.add(new Term[] { new Term("positional", "closed") }, 1);
                    Query mq = PatternPhraseQuery.of("message", multi, analyzer, true, true);
                    int want = searcher.count(mpp.build());
                    check.accept(want > 0 && searcher.count(mq) == want, "pattern_text " + which + " stacked terms: " + searcher.count(mq) + " hits, positional " + want);
                    queries.add(mq);
                    for (Query q : queries) {
                        QueryEncoder.Encoded enc = QueryEncoder.encode(q, f -> true);
                        check.accept(enc.blob() != null, "pattern_text phrase does not encode: " + q + " (" + enc.fallbackReason() + ")");
                        encoded += enc.blob() != null ? 1 : 0;
                    }
                    compare.run("pattern_text " + which + " round " + round + " (" + reader.leaves().size() + " segments, " + reader.numDeletedDocs() + " deleted)", reader, readers, queries);
                }
                reader.close();
                check.accept(readers.openCount() == 0, "pattern_text: no native readers left open");
            }
            try (Stream<Path> s = Files.walk(dir)) {
                s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }
}
