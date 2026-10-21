import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.payloads.SpanPayloadCheckQuery;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.sandbox.search.CombinedFieldQuery;
import org.apache.lucene.sandbox.search.CoveringQuery;
import org.apache.lucene.sandbox.search.PhraseWildcardQuery;
import org.apache.lucene.sandbox.search.TermAutomatonQuery;
import org.apache.lucene.sandbox.search.TokenStreamToTermAutomatonQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.FuzzyQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * lucene-sandbox's scoring queries, recorded from Lucene 10.5.0 for
 * crates/lucene-search/tests/sandbox_query_fixtures.rs. Written to {@code sandbox_queries/}:
 *
 * <pre>
 *   index/         three segments (90, 60, 30 documents) with deletions: body, title (text),
 *                  id (string), mm (numeric doc values on most documents)
 *   searches.tsv   query spec, then search(q, 20)'s total hits and doc:scoreBits per hit
 *   errors.tsv     query spec, then the exception building or searching it throws
 * </pre>
 *
 * <p>Specs, space separated and nested by prefix:
 *
 * <pre>
 *   T field term                               TermQuery
 *   B n Q...                                   BooleanQuery, n SHOULD clauses
 *   boost x Q                                  BoostQuery
 *   CFS n term... m field weight...            sandbox CombinedFieldQuery
 *   COV c|f N|field n Q...                     CoveringQuery, constant or long-field minimum
 *   TAQ field states accept,... n s>d>term...  TermAutomatonQuery ("*": any term), then finish()
 *   TS field preserve n term/inc/len...        TokenStreamToTermAutomatonQuery over those tokens
 *   PW field maxExp slop n kind:text...        PhraseWildcardQuery: t term, p prefix, w wildcard,
 *                                              f fuzzy (text~edits)
 *   P field slop n term...                     PhraseQuery
 *   MP field n (m term...)...                  MultiPhraseQuery, n positions of m terms each
 *   ST field term                              SpanTermQuery
 *   SN slop inOrder n Q...                     SpanNearQuery over span specs
 *   SPC payload Q                              SpanPayloadCheckQuery, one payload to match
 * </pre>
 */
public class GenSandboxQueries {

  static String words(Random r, int n, int vocab) {
    StringBuilder sb = new StringBuilder();
    for (int k = 0; k < n; k++) {
      double x = r.nextDouble();
      int w = (int) (vocab * x * x);
      if (k > 0) sb.append(' ');
      sb.append('w').append(w);
    }
    return sb.toString();
  }

  // --- query specs ---------------------------------------------------------------------------

  static final class Parser {
    final String[] tok;
    int pos;

    Parser(String spec) {
      tok = spec.split(" ");
    }

    String next() {
      return tok[pos++];
    }

    int i() {
      return Integer.parseInt(next());
    }

    Query query() throws IOException {
      String op = next();
      switch (op) {
        case "T":
          {
            String f = next();
            return new TermQuery(new Term(f, next()));
          }
        case "B":
          {
            int n = i();
            BooleanQuery.Builder b = new BooleanQuery.Builder();
            for (int k = 0; k < n; k++) b.add(query(), BooleanClause.Occur.SHOULD);
            return b.build();
          }
        case "boost":
          {
            float x = Float.parseFloat(next());
            return new BoostQuery(query(), x);
          }
        case "CFS":
          {
            CombinedFieldQuery.Builder b = new CombinedFieldQuery.Builder();
            int n = i();
            for (int k = 0; k < n; k++) b.addTerm(new BytesRef(next()));
            int m = i();
            for (int k = 0; k < m; k++) {
              String f = next();
              b.addField(f, Float.parseFloat(next()));
            }
            return b.build();
          }
        case "COV":
          {
            String kind = next();
            String arg = next();
            LongValuesSource min =
                kind.equals("c")
                    ? LongValuesSource.constant(Long.parseLong(arg))
                    : LongValuesSource.fromLongField(arg);
            int n = i();
            List<Query> qs = new ArrayList<>();
            for (int k = 0; k < n; k++) qs.add(query());
            return new CoveringQuery(qs, min);
          }
        case "TAQ":
          {
            TermAutomatonQuery q = new TermAutomatonQuery(next());
            int states = i();
            for (int k = 0; k < states; k++) q.createState();
            String accept = next();
            if (!accept.equals("-")) {
              for (String a : accept.split(",")) q.setAccept(Integer.parseInt(a), true);
            }
            int n = i();
            for (int k = 0; k < n; k++) {
              String[] t = next().split(">", 3);
              int s = Integer.parseInt(t[0]), d = Integer.parseInt(t[1]);
              if (t[2].equals("*")) q.addAnyTransition(s, d);
              else q.addTransition(s, d, t[2]);
            }
            q.finish();
            return q;
          }
        case "TS":
          {
            String f = next();
            boolean preserve = next().equals("1");
            int n = i();
            String[] tokens = new String[n];
            for (int k = 0; k < n; k++) tokens[k] = next();
            TokenStreamToTermAutomatonQuery ts = new TokenStreamToTermAutomatonQuery();
            ts.setPreservePositionIncrements(preserve);
            return ts.toQuery(f, new Canned(tokens));
          }
        case "PW":
          {
            String f = next();
            int maxExp = i();
            int slop = i();
            int n = i();
            PhraseWildcardQuery.Builder b = new PhraseWildcardQuery.Builder(f, maxExp);
            b.setSlop(slop);
            for (int k = 0; k < n; k++) {
              String[] p = next().split(":", 2);
              switch (p[0]) {
                case "t":
                  b.addTerm(new BytesRef(p[1]));
                  break;
                case "p":
                  b.addMultiTerm(new PrefixQuery(new Term(f, p[1])));
                  break;
                case "w":
                  b.addMultiTerm(new WildcardQuery(new Term(f, p[1])));
                  break;
                case "f":
                  {
                    String[] e = p[1].split("~");
                    b.addMultiTerm(new FuzzyQuery(new Term(f, e[0]), Integer.parseInt(e[1])));
                    break;
                  }
                default:
                  throw new IllegalArgumentException(p[0]);
              }
            }
            return b.build();
          }
        case "P":
          {
            String f = next();
            int slop = i();
            int n = i();
            String[] terms = new String[n];
            for (int k = 0; k < n; k++) terms[k] = next();
            return new PhraseQuery(slop, f, terms);
          }
        case "MP":
          {
            String f = next();
            int n = i();
            MultiPhraseQuery.Builder b = new MultiPhraseQuery.Builder();
            for (int k = 0; k < n; k++) {
              Term[] terms = new Term[i()];
              for (int j = 0; j < terms.length; j++) terms[j] = new Term(f, next());
              b.add(terms);
            }
            return b.build();
          }
        case "ST":
          {
            String f = next();
            return new SpanTermQuery(new Term(f, next()));
          }
        case "SN":
          {
            int slop = i();
            boolean inOrder = next().equals("1");
            SpanQuery[] clauses = new SpanQuery[i()];
            for (int k = 0; k < clauses.length; k++) clauses[k] = (SpanQuery) query();
            return new SpanNearQuery(clauses, slop, inOrder);
          }
        case "SPC":
          {
            BytesRef payload = new BytesRef(next());
            return new SpanPayloadCheckQuery((SpanQuery) query(), List.of(payload));
          }
        default:
          throw new IllegalArgumentException(op);
      }
    }
  }

  /** A token stream of {@code term/positionIncrement/positionLength} tokens. */
  static final class Canned extends TokenStream {
    final String[] tokens;
    int at;
    final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    final PositionIncrementAttribute inc = addAttribute(PositionIncrementAttribute.class);
    final PositionLengthAttribute len = addAttribute(PositionLengthAttribute.class);
    final OffsetAttribute off = addAttribute(OffsetAttribute.class);

    Canned(String[] tokens) {
      this.tokens = tokens;
    }

    @Override
    public boolean incrementToken() {
      if (at == tokens.length) return false;
      clearAttributes();
      String[] t = tokens[at].split("/");
      term.setEmpty().append(t[0]);
      inc.setPositionIncrement(Integer.parseInt(t[1]));
      len.setPositionLength(Integer.parseInt(t[2]));
      off.setOffset(at, at + 1);
      at++;
      return true;
    }

    @Override
    public void reset() throws IOException {
      super.reset();
      at = 0;
    }
  }

  static final String[] SEARCHES = {
    // CombinedFieldQuery, sandbox: several terms as one stream.
    "CFS 1 w1 2 body 1 title 1",
    "CFS 2 w1 w2 2 body 1 title 2",
    "CFS 3 w0 w3 w9 1 body 1.5",
    "CFS 2 w4 w5 2 body 2.5 title 1",
    "CFS 2 zz w2 2 body 1 title 1",
    "CFS 2 zz yy 1 body 1",
    "CFS 2 w6 w6 2 title 1 body 1",
    "boost 2.5 CFS 2 w1 w7 2 body 1 title 3",
    "B 2 CFS 2 w3 w8 1 body 1 T title w2",
    // CoveringQuery
    "COV c 1 3 T body w1 T body w2 T body w3",
    "COV c 2 3 T body w1 T body w2 T body w3",
    "COV c 3 3 T body w1 T body w2 T body w3",
    "COV c 4 3 T body w1 T body w2 T body w3",
    "COV c 0 2 T body w5 T title w5",
    "COV f mm 4 T body w0 T body w1 T body w2 T title w0",
    "COV c 1 2 PW body 100 0 2 t:w0 p:w1 T body w5",
    "COV f mm 3 T body w4 T body w6 T body w8",
    "COV f mm 2 T body w11 T body zz",
    "COV f nope 2 T body w0 T body w1",
    "COV f mm 3 T body w1 T body w1 T body w2",
    "COV f mm 2 B 2 T body w1 T body w2 T title w3",
    "boost 3 COV f mm 3 T body w0 T title w1 T body w2",
    "B 2 COV f mm 2 T body w1 T body w2 T title w0",
    // TermAutomatonQuery
    "TAQ body 3 2 2 0>1>w0 1>2>w1",
    "TAQ body 3 2 3 0>1>w0 0>1>w2 1>2>w1",
    "TAQ body 4 3 3 0>1>w0 1>2>* 2>3>w1",
    "TAQ body 4 2,3 4 0>1>w1 1>2>w0 2>3>w2 1>3>w3",
    "TAQ body 3 1,2 2 0>1>w0 1>2>w0",
    "TAQ body 2 1 1 0>1>w3",
    "TAQ body 2 1 2 0>1>w3 0>1>w4",
    "TAQ body 3 2 4 0>1>w0 0>1>w1 1>2>w0 1>2>w1",
    "TAQ body 4 3 4 0>1>w2 1>2>* 2>2>* 2>3>w0",
    "TAQ body 3 2 2 0>1>zz 1>2>w1",
    "TAQ title 3 2 2 0>1>w0 1>2>w1",
    "boost 2 TAQ body 3 2 3 0>1>w0 0>1>w2 1>2>w1",
    "B 2 TAQ body 4 3 3 0>1>w0 1>2>* 2>3>w1 T body w5",
    "TAQ body 3 2 1 0>1>w0",
    // TokenStreamToTermAutomatonQuery
    "TS body 1 2 w0/1/1 w1/1/1",
    "TS body 1 3 w0/1/1 w2/0/1 w1/1/1",
    "TS body 1 4 w0/1/1 w3/0/2 w1/1/1 w2/1/1",
    "TS body 1 3 w1/1/1 */1/1 w0/1/1",
    "TS body 0 2 w0/1/1 w1/3/1",
    "TS body 1 1 w2/1/1",
    // PhraseWildcardQuery
    "PW body 100 0 2 t:w0 t:w1",
    "PW body 100 0 2 t:w0 p:w1",
    "PW body 100 1 2 p:w2 t:w0",
    "PW body 100 0 3 t:w1 w:w?0 t:w2",
    "PW body 3 0 2 t:w0 p:w1",
    "PW body 5 0 2 p:w1 p:w2",
    "PW body 4 1 3 p:w1 t:w0 p:w2",
    "PW body 1 0 2 p:w1 p:w2",
    "PW body 100 2 2 f:w15~1 t:w0",
    "PW body 100 0 2 t:zz p:w1",
    "PW body 100 0 2 p:zz t:w1",
    "PW body 100 0 1 p:w3",
    "PW body 100 0 1 t:w3",
    "PW title 100 0 2 t:w0 p:w",
    "boost 1.5 PW body 100 1 2 t:w1 p:w0",
  };

  static final String[] ERRORS = {
    "CFS 1 w1 1 body 0.5",
    "TAQ body 3 2 2 0>1>* 1>2>w1",
    "TAQ body 3 2 2 0>1>w0 1>2>*",
    "TAQ body 2 0,1 1 0>1>w0",
    "TS body 1 2 w0/1/1 w1/2/1",
    "PW body 100 0 0",
    "PW body 100 -1 2 t:w0 t:w1",
    "COV c 1 0",
    // Positional queries over a field indexed without positions (id): IllegalStateException
    // from the scorer, unless rewrite turns the query into term queries first.
    "P id 0 2 1 2",
    "P id 2 2 1 zz",
    "P id 0 1 1",
    "P nope 0 2 a b",
    "P body 0 2 w0 w1",
    "B 2 T body w1 P id 1 2 1 2",
    "boost 2 P id 0 3 1 2 3",
    "MP id 2 2 1 2 1 3",
    "MP id 1 2 1 2",
    "ST id 1",
    "ST id zz",
    "SN 0 1 2 ST id 1 ST id 2",
    "SPC x ST id 1",
    "SPC x ST id zz",
  };

  static String hits(IndexSearcher searcher, Query q) throws IOException {
    TopDocs td = searcher.search(q, 20);
    StringBuilder sb = new StringBuilder();
    sb.append(td.totalHits.value()).append('\t');
    for (ScoreDoc sd : td.scoreDocs) {
      sb.append(sd.doc).append(':').append(Integer.toHexString(Float.floatToIntBits(sd.score))).append(',');
    }
    return sb.toString();
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("sandbox_queries");
    if (Files.exists(root)) {
      try (Stream<Path> walk = Files.walk(root)) {
        for (Path p : walk.sorted((x, y) -> y.compareTo(x)).toList()) Files.delete(p);
      }
    }
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(20261019L);
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      int id = 0;
      StringBuilder docs = new StringBuilder();
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int size : new int[] {90, 60, 30}) {
          for (int k = 0; k < size; k++) {
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            String body = words(r, 1 + r.nextInt(r.nextInt(4) == 0 ? 40 : 10), 20);
            doc.add(new TextField("body", body, Field.Store.NO));
            String title = r.nextInt(5) != 0 ? words(r, 1 + r.nextInt(4), 8) : null;
            if (title != null) doc.add(new TextField("title", title, Field.Store.NO));
            long mm = r.nextInt(6) != 0 ? r.nextInt(4) : -1;
            if (mm >= 0) doc.add(new NumericDocValuesField("mm", mm));
            docs.append(id).append('\t').append(body).append('\t')
                .append(title == null ? "-" : title).append('\t').append(mm).append('\n');
            w.addDocument(doc);
            id++;
          }
          w.commit();
        }
        for (String del : new String[] {"3", "17", "40", "95", "101", "130", "171"}) {
          w.deleteDocuments(new Term("id", del));
        }
        w.commit();
      }
      Files.writeString(root.resolve("docs.tsv"), docs);

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 3) throw new AssertionError("segments");
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        StringBuilder out = new StringBuilder();
        for (String spec : SEARCHES) {
          Query q = new Parser(spec).query();
          out.append(spec).append('\t').append(hits(searcher, q)).append('\n');
        }
        Files.writeString(root.resolve("searches.tsv"), out);
        StringBuilder err = new StringBuilder();
        for (String spec : ERRORS) {
          String msg;
          try {
            searcher.search(new Parser(spec).query(), 20);
            msg = "OK";
          } catch (RuntimeException ex) {
            msg = ex.getClass().getSimpleName() + ": " + ex.getMessage();
          }
          err.append(spec).append('\t').append(msg).append('\n');
        }
        Files.writeString(root.resolve("errors.tsv"), err);
      }
    }
  }
}
