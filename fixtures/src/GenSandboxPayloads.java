/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.payloads.PayloadSpanUtil;
import org.apache.lucene.queries.spans.SpanMultiTermQueryWrapper;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanOrQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * lucene-sandbox's {@code PayloadSpanUtil} (with its {@code PayloadSpanCollector}) over a
 * two-segment index with deletions, whose {@code body} carries a payload per position (none at
 * every fifth) and whose {@code plain} has none.
 *
 * <pre>
 *   sandbox_payloads/
 *     index/          the index
 *     payloads.tsv    query spec, then the payloads in Java's order (hex, comma-separated),
 *                     or the exception's class and message
 * </pre>
 *
 * <p>A query spec is prefix notation over space-separated tokens: {@code T f w} a TermQuery;
 * {@code P slop n w..} a PhraseQuery on body; {@code MP slop n (pos k w..)..} a
 * MultiPhraseQuery on body; {@code B m s n q..} a BooleanQuery of m MUST, s SHOULD and n MUST_NOT
 * clauses, added in that order; {@code D n q..} a DisjunctionMaxQuery; {@code ST w} a
 * SpanTermQuery on body, {@code SN slop inOrder n s..} a SpanNearQuery, {@code SO n s..} a
 * SpanOrQuery, {@code SP prefix} a SpanMultiTermQueryWrapper of a PrefixQuery; {@code W pattern}
 * a WildcardQuery (a query PayloadSpanUtil ignores).
 *
 * <p>Deleted documents are read: the spans come from {@code SpanWeight.getSpans}, which does not
 * look at live docs.
 */
public class GenSandboxPayloads {

  static final String[] SPECS = {
    "T body w1",
    "T body w9",
    "T body zz",
    "T plain w1",
    "T id 3",
    "P 0 2 w1 w2",
    "P 2 2 w2 w1",
    "P 1 3 w0 w1 w3",
    "P 0 1 w1",
    "MP 0 2 0 2 w1 w2 1 1 w3",
    "MP 1 2 0 1 w0 2 2 w1 w4",
    "MP 0 3 0 1 w2 0 1 w3 3 1 w0",
    "MP 0 2 2 1 w1 0 1 w2",
    "B 1 1 1 T body w4 T body w5 T body w6",
    "B 0 2 0 P 0 2 w1 w2 T body w7",
    "D 2 T body w8 SN 1 true 2 ST w1 ST w0",
    "ST w2",
    "SN 0 false 2 ST w0 ST w1",
    "SN 3 true 3 ST w1 SO 2 ST w2 ST w3 ST w0",
    "SO 2 SP w1 ST w5",
    "SN 1 true 2 SP w ST w2",
    "W body w1*",
    "B 1 0 0 W body w*",
    "BF 1 1 1 T body w4 P 0 2 w1 w2 T body w6",
  };

  static BytesRef payload(int doc, int pos) {
    if (pos % 5 == 4) return null;
    int len = 1 + (doc + pos) % 3;
    byte[] b = new byte[len];
    for (int i = 0; i < len; i++) b[i] = (byte) (doc * 7 + pos * 13 + i * 31);
    return new BytesRef(b);
  }

  /** Whitespace tokens, each with {@code payload(doc, position)}; the doc is the text's first token. */
  static final class PayloadAnalyzer extends Analyzer {
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer t = new WhitespaceTokenizer();
      TokenStream ts =
          new TokenFilter(t) {
            final PayloadAttribute pay = addAttribute(PayloadAttribute.class);
            int pos;
            int doc;

            @Override
            public boolean incrementToken() throws IOException {
              if (!input.incrementToken()) return false;
              if (pos == 0) {
                doc = Integer.parseInt(input.getAttribute(
                    org.apache.lucene.analysis.tokenattributes.CharTermAttribute.class)
                    .toString().substring(1));
              }
              pay.setPayload(payload(doc, pos++));
              return true;
            }

            @Override
            public void reset() throws IOException {
              super.reset();
              pos = 0;
            }
          };
      return new TokenStreamComponents(t, ts);
    }
  }

  static int at;
  static String[] toks;

  static Query query() {
    String t = toks[at++];
    switch (t) {
      case "T":
        return new TermQuery(new Term(toks[at++], toks[at++]));
      case "P": {
        int slop = Integer.parseInt(toks[at++]);
        int n = Integer.parseInt(toks[at++]);
        PhraseQuery.Builder b = new PhraseQuery.Builder().setSlop(slop);
        for (int i = 0; i < n; i++) b.add(new Term("body", toks[at++]));
        return b.build();
      }
      case "MP": {
        int slop = Integer.parseInt(toks[at++]);
        int n = Integer.parseInt(toks[at++]);
        MultiPhraseQuery.Builder b = new MultiPhraseQuery.Builder().setSlop(slop);
        for (int i = 0; i < n; i++) {
          int pos = Integer.parseInt(toks[at++]);
          int k = Integer.parseInt(toks[at++]);
          Term[] ts = new Term[k];
          for (int j = 0; j < k; j++) ts[j] = new Term("body", toks[at++]);
          b.add(ts, pos);
        }
        return b.build();
      }
      case "B": {
        int m = Integer.parseInt(toks[at++]);
        int s = Integer.parseInt(toks[at++]);
        int n = Integer.parseInt(toks[at++]);
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        for (int i = 0; i < m; i++) b.add(query(), BooleanClause.Occur.MUST);
        for (int i = 0; i < s; i++) b.add(query(), BooleanClause.Occur.SHOULD);
        for (int i = 0; i < n; i++) b.add(query(), BooleanClause.Occur.MUST_NOT);
        return b.build();
      }
      case "BF": {
        // MUST, FILTER and SHOULD clauses: walked in that order by both.
        int m = Integer.parseInt(toks[at++]);
        int f = Integer.parseInt(toks[at++]);
        int s = Integer.parseInt(toks[at++]);
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        for (int i = 0; i < m; i++) b.add(query(), BooleanClause.Occur.MUST);
        for (int i = 0; i < f; i++) b.add(query(), BooleanClause.Occur.FILTER);
        for (int i = 0; i < s; i++) b.add(query(), BooleanClause.Occur.SHOULD);
        return b.build();
      }
      case "D": {
        int n = Integer.parseInt(toks[at++]);
        List<Query> qs = new ArrayList<>();
        for (int i = 0; i < n; i++) qs.add(query());
        return new DisjunctionMaxQuery(qs, 0.1f);
      }
      case "W":
        return new WildcardQuery(new Term(toks[at++], toks[at++]));
      default:
        at--;
        return span();
    }
  }

  static SpanQuery span() {
    String t = toks[at++];
    switch (t) {
      case "ST":
        return new SpanTermQuery(new Term("body", toks[at++]));
      case "SN": {
        int slop = Integer.parseInt(toks[at++]);
        boolean inOrder = Boolean.parseBoolean(toks[at++]);
        int n = Integer.parseInt(toks[at++]);
        SpanQuery[] cs = new SpanQuery[n];
        for (int i = 0; i < n; i++) cs[i] = span();
        return new SpanNearQuery(cs, slop, inOrder);
      }
      case "SO": {
        int n = Integer.parseInt(toks[at++]);
        SpanQuery[] cs = new SpanQuery[n];
        for (int i = 0; i < n; i++) cs[i] = span();
        return new SpanOrQuery(cs);
      }
      case "SP":
        return new SpanMultiTermQueryWrapper<>(new PrefixQuery(new Term("body", toks[at++])));
      default:
        throw new AssertionError(t);
    }
  }

  static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder();
    for (byte x : b) sb.append(String.format("%02x", x & 0xff));
    return sb.toString();
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("sandbox_payloads");
    if (Files.exists(root)) {
      try (Stream<Path> walk = Files.walk(root)) {
        for (Path p : walk.sorted((x, y) -> y.compareTo(x)).toList()) Files.delete(p);
      }
    }
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(20261019L);
    FieldType withPositions = new FieldType(TextField.TYPE_NOT_STORED);
    withPositions.freeze();
    FieldType docsOnly = new FieldType(TextField.TYPE_NOT_STORED);
    docsOnly.setIndexOptions(IndexOptions.DOCS);
    docsOnly.freeze();
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new PayloadAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      int id = 0;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int size : new int[] {25, 15}) {
          for (int k = 0; k < size; k++) {
            StringBuilder body = new StringBuilder("d").append(id);
            int n = 2 + r.nextInt(9);
            for (int i = 0; i < n; i++) body.append(" w").append(r.nextInt(10));
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            doc.add(new Field("body", body.toString(), withPositions));
            doc.add(new Field("plain", body.toString(), docsOnly));
            w.addDocument(doc);
            id++;
          }
          w.commit();
        }
        for (String del : new String[] {"3", "17", "30"}) w.deleteDocuments(new Term("id", del));
        w.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 2) throw new AssertionError("segments");
        PayloadSpanUtil util = new PayloadSpanUtil(reader.getContext());
        StringBuilder out = new StringBuilder();
        for (String spec : SPECS) {
          toks = spec.split(" ");
          at = 0;
          Query q = query();
          if (at != toks.length) throw new AssertionError(spec);
          out.append(spec).append('\t');
          try {
            List<String> ps = new ArrayList<>();
            for (byte[] p : util.getPayloadsForQuery(q)) ps.add(hex(p));
            out.append(ps.size()).append('\t').append(String.join(",", ps));
          } catch (RuntimeException e) {
            out.append(e.getClass().getSimpleName()).append(": ").append(e.getMessage());
          }
          out.append('\n');
        }
        Files.writeString(root.resolve("payloads.tsv"), out);
      }
    }
  }
}
