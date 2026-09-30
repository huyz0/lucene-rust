import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.TermFrequencyAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.FieldInvertState;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.AttributeSource;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * What {@code IndexingChain} reads from a token stream besides the term: a {@code TokenFilter}'s
 * {@code PayloadAttribute} and {@code TermFrequencyAttribute}, and the {@code FieldInvertState} a
 * {@code Similarity.computeNorm} sees ({@code maxTermFrequency}, {@code position}, {@code offset},
 * the attribute source as {@code end()} left it).
 *
 * <p>The analyzer ({@link Attrs}, per-field reuse, position-increment gap 3, offset gap 5) is
 * {@code StandardTokenizer} + {@code LowerCaseFilter} + {@link Tag}, which on field {@code pay}
 * gives a term starting before {@code m} the payload {@code [length, first char]}, one starting at
 * {@code t} or later an empty payload (none, to the postings), and on field {@code tf} sets the term
 * frequency to the term's length. Fields per document: {@code id} (a stored {@code StringField}),
 * {@code pay} (two values, positions and offsets), {@code tf} ({@code DOCS_AND_FREQS}, custom
 * frequencies) and {@code docs} ({@code DOCS}, a repeated term).
 *
 * <p>The writer's similarity is {@link StateSim}: its norm packs every {@code FieldInvertState}
 * value, and it records each state it is handed in {@code states.txt} (sorted), so the Rust test
 * compares both the norms and the states themselves.
 *
 * <p>{@code token_attributes/flushed}: one segment of {@link #PER_SEGMENT} documents. {@code
 * token_attributes/merged}: a first segment whose {@code pay} values carry no payload (every word
 * from {@code m} to {@code s}), a second like {@code flushed}'s, force-merged into one. The Rust
 * test writes the same documents through the document API and compares every segment file but the
 * {@code .si} byte for byte (segment id normalised).
 */
public class GenTokenAttributes {
  static final int PER_SEGMENT = 120;
  static final String[] WORDS = {
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "xray", "yankee", "zulu"
  };
  /** {@code mike} .. {@code sierra}: no payload from {@link Tag}. */
  static final int PLAIN_FROM = 12;
  static final int PLAIN_TO = 19;

  static long value(int i, int k) {
    long x = (i + 1) * 2654435761L + k * 40503L;
    return (x ^ (x >>> 13)) % 100000;
  }

  static String word(int i, int k, boolean plain) {
    if (plain) {
      return WORDS[PLAIN_FROM + (int) (value(i, k) % (PLAIN_TO - PLAIN_FROM))];
    }
    return WORDS[(int) (value(i, k) % WORDS.length)];
  }

  static String words(int i, int from, int n, boolean plain) {
    StringBuilder b = new StringBuilder();
    for (int k = from; k < from + n; k++) {
      if (b.length() > 0) {
        b.append(k % 3 == 0 ? ", " : " ");
      }
      b.append(word(i, k, plain));
    }
    return b.toString();
  }

  static final class Tag extends TokenFilter {
    final String field;
    final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    final PayloadAttribute payload = addAttribute(PayloadAttribute.class);
    final TermFrequencyAttribute freq = addAttribute(TermFrequencyAttribute.class);

    Tag(TokenStream in, String field) {
      super(in);
      this.field = field;
    }

    @Override
    public boolean incrementToken() throws IOException {
      if (!input.incrementToken()) {
        return false;
      }
      char c = term.charAt(0);
      if (field.equals("pay")) {
        if (c < 'm') {
          payload.setPayload(new BytesRef(new byte[] {(byte) term.length(), (byte) c}));
        } else if (c >= 't') {
          payload.setPayload(new BytesRef(new byte[0]));
        }
      } else if (field.equals("tf")) {
        freq.setTermFrequency(term.length());
      }
      return true;
    }
  }

  static final class Attrs extends Analyzer {
    Attrs() {
      super(PER_FIELD_REUSE_STRATEGY);
    }

    @Override
    protected TokenStreamComponents createComponents(String field) {
      StandardTokenizer src = new StandardTokenizer();
      return new TokenStreamComponents(src, new Tag(new LowerCaseFilter(src), field));
    }

    @Override
    public int getPositionIncrementGap(String field) {
      return 3;
    }

    @Override
    public int getOffsetGap(String field) {
      return 5;
    }
  }

  /** Records every state; the norm packs them all. */
  static final class StateSim extends Similarity {
    final List<String> states = new ArrayList<>();

    @Override
    public long computeNorm(FieldInvertState s) {
      AttributeSource a = s.getAttributeSource();
      long endOffset = a == null ? -1 : a.getAttribute(OffsetAttribute.class).endOffset();
      long endInc =
          a == null ? -1 : a.getAttribute(PositionIncrementAttribute.class).getPositionIncrement();
      states.add(
          s.getName()
              + " length="
              + s.getLength()
              + " maxTermFrequency="
              + s.getMaxTermFrequency()
              + " position="
              + s.getPosition()
              + " offset="
              + s.getOffset()
              + " uniqueTermCount="
              + s.getUniqueTermCount()
              + " numOverlap="
              + s.getNumOverlap()
              + " endOffset="
              + endOffset
              + " endIncrement="
              + endInc);
      return norm(
          s.getLength(),
          s.getMaxTermFrequency(),
          s.getPosition(),
          s.getOffset(),
          s.getUniqueTermCount(),
          s.getNumOverlap(),
          endOffset,
          endInc);
    }

    /** The Rust test's `norm`. */
    static long norm(
        long length,
        long maxFreq,
        long position,
        long offset,
        long unique,
        long overlap,
        long endOffset,
        long endInc) {
      return 1
          + length
          + 7 * maxFreq
          + 131 * position
          + 1031 * offset
          + 10007 * unique
          + 100003 * overlap
          + 1000003 * (endOffset + 2)
          + 10000019 * (endInc + 2);
    }

    @Override
    public SimScorer scorer(
        float boost, CollectionStatistics collectionStats, TermStatistics... termStats) {
      return new SimScorer() {
        @Override
        public float score(float freq, long norm) {
          return freq;
        }
      };
    }
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("token_attributes");
    if (Files.exists(root)) {
      try (var walk = Files.walk(root)) {
        walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    write(root.resolve("flushed"), false);
    write(root.resolve("merged"), true);
    System.out.println("wrote token_attributes/");
  }

  static Document doc(int i, boolean plain) {
    FieldType pay = new FieldType();
    pay.setTokenized(true);
    pay.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    pay.freeze();
    FieldType tf = new FieldType();
    tf.setTokenized(true);
    tf.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    tf.freeze();
    FieldType docs = new FieldType();
    docs.setTokenized(true);
    docs.setIndexOptions(IndexOptions.DOCS);
    docs.freeze();
    Document d = new Document();
    d.add(new StringField("id", "d" + i, Field.Store.YES));
    d.add(new Field("pay", words(i, 0, 4, plain), pay));
    d.add(new Field("pay", words(i, 4, 3, plain) + " ", pay));
    d.add(new Field("tf", words(i, 7, 3, false) + " " + word(i, 7, false), tf));
    d.add(new Field("docs", word(i, 10, false) + " " + word(i, 10, false) + " " + word(i, 11, false), docs));
    return d;
  }

  static void write(Path out, boolean merge) throws IOException {
    Files.createDirectories(out);
    StateSim sim = new StateSim();
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new Attrs());
      cfg.setSimilarity(sim);
      cfg.setUseCompoundFile(false);
      cfg.setRAMBufferSizeMB(256);
      if (merge) {
        TieredMergePolicy tmp = new TieredMergePolicy();
        tmp.setNoCFSRatio(0.0);
        cfg.setMergePolicy(tmp);
      } else {
        cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      }
      cfg.setMaxFullFlushMergeWaitMillis(0);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        int segments = merge ? 2 : 1;
        for (int seg = 0; seg < segments; seg++) {
          for (int i = seg * PER_SEGMENT; i < (seg + 1) * PER_SEGMENT; i++) {
            w.addDocument(doc(i, merge && seg == 0));
          }
          w.commit();
        }
        if (merge) {
          w.forceMerge(1);
          w.commit();
        }
      }
      List<String> states = new ArrayList<>(sim.states);
      Collections.sort(states);
      Files.write(out.resolve("states.txt"), states, StandardCharsets.UTF_8);
      CheckIndex.Status status;
      try (CheckIndex check = new CheckIndex(dir)) {
        status = check.checkIndex();
      }
      if (!status.clean) {
        throw new AssertionError("CheckIndex failed on " + out);
      }
    }
  }
}
