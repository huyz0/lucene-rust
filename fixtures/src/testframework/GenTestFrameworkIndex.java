import com.carrotsearch.randomizedtesting.RandomizedRunner;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.ConcurrentMergeScheduler;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LogByteSizeMergePolicy;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.LogMergePolicy;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeScheduler;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.index.AlcoholicMergePolicy;
import org.apache.lucene.tests.index.MockIndexWriterEventListener;
import org.apache.lucene.tests.index.MockRandomMergePolicy;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.RunWith;
import org.junit.runner.notification.Failure;

/**
 * Differential fixtures for lucene-test-framework's writer configuration (M14.h h4): what Lucene
 * 10.5.0's {@code newIndexWriterConfig(r, a)} and {@code maybeChangeLiveIndexWriterConfig(r, c)}
 * draw from a seeded {@code java.util.Random r}.
 *
 * <p>Both helpers also draw from the test's own {@code random()} (the CMS throttle and its
 * force-merge rate, the default CMS class, {@code LogMergePolicy}'s target search concurrency,
 * {@code TieredMergePolicy}'s deletes percentage, the {@code rarely()} choosing the full-flush
 * wait). Those values are not recorded; everything drawn from {@code r} is, and a {@code tail}
 * line ({@code r.nextLong()} afterwards) proves the number of draws matches.
 *
 * <p>Writes {@code index.txt}: for each of 200 seeds a {@code seed} line, then {@code iwc ...}
 * lines for {@code newIndexWriterConfig(new Random(seed), a)}; then, for the first 100 seeds, a
 * {@code mode} line and 40 {@code live <i> ...} lines, the state after each of 40 calls to {@code
 * maybeChangeLiveIndexWriterConfig(new Random(seed * 31 + 1), c)} on a config made from {@code
 * new Random(seed * 31 + 2)}, with {@code liveIWCFlushMode} set to mode {@code seed % 3}. Doubles
 * travel as their {@code doubleToLongBits}, sizes as bytes.
 *
 * <p>Writes {@code riw.txt}: for each of 100 seeds, what {@code new RandomIndexWriter(new
 * Random(seed), dir)} drew (with {@code liveIWCFlushMode} set to mode {@code seed % 3}, which it prints: its first flush point, soft-deletes ratio, whether it force-merges at
 * random, and the outer stream's next long), then, after 300 adds, updates and deletes, its
 * document count, flush point and its own stream's next long.
 *
 * <p>Writes {@code dvonly.txt}: doc-values updates naming fields no document has yet ({@code
 * verifyOrCreateDvOnlyField} creates them) and fields a segment lacks, then every document's
 * values and the merged field infos.
 *
 * <p>{@code newIndexWriterConfig} reads {@code classEnvRule}, so the dump runs as a {@link
 * LuceneTestCase} under {@link JUnitCore} (with assertions enabled, which its rules require). Run
 * with no {@code tests.*} system properties: {@code rarely} sees multiplier 1, not nightly. Usage:
 * java GenTestFrameworkIndex <outdir> [ignored]
 */
public class GenTestFrameworkIndex {
  static final StringBuilder OUT = new StringBuilder();
  static final StringBuilder RIW = new StringBuilder();
  static final StringBuilder DV = new StringBuilder();
  static final int SEEDS = 200;
  static final int LIVE_SEEDS = 100;
  static final int LIVE_CHANGES = 40;
  static final int RIW_SEEDS = 100;
  static final int RIW_OPS = 300;

  static void line(Object... parts) {
    for (int i = 0; i < parts.length; i++) {
      if (i > 0) OUT.append(' ');
      OUT.append(parts[i]);
    }
    OUT.append('\n');
  }

  static String bits(double d) {
    return Long.toString(Double.doubleToLongBits(d));
  }

  static long bytes(double mb) {
    return (long) (mb * 1024 * 1024);
  }

  static long field(LogMergePolicy mp, String name) throws Exception {
    Field f = LogMergePolicy.class.getDeclaredField(name);
    f.setAccessible(true);
    return f.getLong(mp);
  }

  static String policy(MergePolicy mp) throws Exception {
    String cfs = bits(mp.getNoCFSRatio()) + " " + bytes(mp.getMaxCFSSegmentSizeMB());
    if (mp instanceof MockRandomMergePolicy) {
      return "mock";
    } else if (mp instanceof TieredMergePolicy tmp) {
      return "tiered "
          + tmp.getMaxMergeAtOnce()
          + " "
          + bytes(tmp.getMaxMergedSegmentMB())
          + " "
          + bytes(tmp.getFloorSegmentMB())
          + " "
          + bits(tmp.getForceMergeDeletesPctAllowed())
          + " "
          + (long) tmp.getSegmentsPerTier()
          + " "
          + tmp.getTargetSearchConcurrency()
          + " "
          + cfs;
    } else if (mp instanceof AlcoholicMergePolicy amp) {
      return "alcoholic " + field(amp, "maxMergeSize") + " " + cfs;
    } else if (mp instanceof LogMergePolicy lmp) {
      String kind =
          lmp instanceof LogDocMergePolicy
              ? "doc"
              : lmp instanceof LogByteSizeMergePolicy ? "byte" : "?";
      return "log "
          + kind
          + " "
          + lmp.getCalibrateSizeByDeletes()
          + " "
          + lmp.getMergeFactor()
          + " "
          + cfs;
    }
    throw new AssertionError("unexpected policy " + mp);
  }

  static String scheduler(MergeScheduler ms) {
    if (ms instanceof SerialMergeScheduler) return "serial";
    ConcurrentMergeScheduler cms = (ConcurrentMergeScheduler) ms;
    return "cms " + cms.getMaxThreadCount() + " " + cms.getMaxMergeCount();
  }

  static void iwc(long seed) throws Exception {
    Random r = new Random(seed);
    IndexWriterConfig c = LuceneTestCase.newIndexWriterConfig(r, new MockAnalyzer(new Random(0)));
    MergeScheduler ms = c.getMergeScheduler();
    if (ms instanceof SerialMergeScheduler) {
      line("iwc", "ms", "serial");
    } else {
      ConcurrentMergeScheduler cms = (ConcurrentMergeScheduler) ms;
      if (Double.isInfinite(cms.getForceMergeMBPerSec())) {
        line("iwc", "ms", "default");
      } else {
        line(
            "iwc",
            "ms",
            "concurrent",
            cms.getClass().isAnonymousClass(),
            cms.getMaxThreadCount(),
            cms.getMaxMergeCount());
      }
    }
    line("iwc", "mbd", c.getMaxBufferedDocs());
    line("iwc", "mp", policy(c.getMergePolicy()));
    line(
        "iwc",
        "flags",
        c.getMergedSegmentWarmer() != null,
        c.getUseCompoundFile(),
        c.getReaderPooling(),
        c.isCheckPendingFlushOnUpdate(),
        c.getIndexWriterEventListener() instanceof MockIndexWriterEventListener);
    line("iwc", "tail", r.nextLong());
  }

  static void live(long seed) throws Exception {
    Field mode = LuceneTestCase.class.getDeclaredField("liveIWCFlushMode");
    mode.setAccessible(true);
    Object saved = mode.get(null);
    @SuppressWarnings({"unchecked", "rawtypes"})
    Object m =
        ((Class<? extends Enum>) mode.getType()).getEnumConstants()[(int) (seed % 3)];
    mode.set(null, m);
    try {
      line("mode", m);
      IndexWriterConfig c =
          LuceneTestCase.newIndexWriterConfig(
              new Random(seed * 31 + 2), new MockAnalyzer(new Random(0)));
      Random r = new Random(seed * 31 + 1);
      for (int i = 0; i < LIVE_CHANGES; i++) {
        LuceneTestCase.maybeChangeLiveIndexWriterConfig(r, c);
        line(
            "live",
            i,
            c.getMaxBufferedDocs(),
            bits(c.getRAMBufferSizeMB()),
            c.getMergedSegmentWarmer() != null,
            c.getUseCompoundFile(),
            scheduler(c.getMergeScheduler()),
            policy(c.getMergePolicy()));
      }
      line("live", "tail", r.nextLong());
    } finally {
      mode.set(null, saved);
    }
  }

  static Object get(Object o, String name) throws Exception {
    Field f = RandomIndexWriter.class.getDeclaredField(name);
    f.setAccessible(true);
    return f.get(o);
  }

  static void riwLine(Object... parts) {
    for (int i = 0; i < parts.length; i++) {
      if (i > 0) RIW.append(' ');
      RIW.append(parts[i]);
    }
    RIW.append('\n');
  }

  static Document doc(int i) {
    Document d = new Document();
    d.add(new StringField("id", Integer.toString(i), org.apache.lucene.document.Field.Store.YES));
    d.add(new TextField("body", "text " + i + " of " + (i % 13), org.apache.lucene.document.Field.Store.NO));
    return d;
  }

  /**
   * {@code new RandomIndexWriter(new Random(seed), dir)}: its constructor's draws, then those of
   * {@value #RIW_OPS} adds, updates and deletes (no reader, no close: their draws depend on the
   * segment count).
   */
  static void riw(long seed) throws Exception {
    Field modeField = LuceneTestCase.class.getDeclaredField("liveIWCFlushMode");
    modeField.setAccessible(true);
    Object saved = modeField.get(null);
    @SuppressWarnings({"unchecked", "rawtypes"})
    Object mode =
        ((Class<? extends Enum>) modeField.getType()).getEnumConstants()[(int) (seed % 3)];
    modeField.set(null, mode);
    try (ByteBuffersDirectory dir = new ByteBuffersDirectory()) {
      Random r = new Random(seed);
      RandomIndexWriter w = new RandomIndexWriter(r, dir);
      riwLine(
          "riw",
          seed,
          mode,
          get(w, "flushAt"),
          bits((Double) get(w, "softDeletesRatio")),
          get(w, "doRandomForceMerge"),
          r.nextLong());
      for (int i = 0; i < RIW_OPS; i++) {
        if (i % 7 == 3) {
          w.updateDocument(new Term("id", Integer.toString(i / 2)), doc(i));
        } else if (i % 11 == 5) {
          w.deleteDocuments(new Term("id", Integer.toString(i / 3)));
        } else if (i % 17 == 9) {
          w.updateDocuments(new Term("id", Integer.toString(i / 2)), java.util.List.of(doc(i)));
        } else {
          w.addDocument(doc(i));
        }
      }
      riwLine(
          "ops", get(w, "docCount"), get(w, "flushAt"), ((Random) get(w, "r")).nextLong());
      w.w.rollback();
    } finally {
      modeField.set(null, saved);
    }
  }

  static Document idDoc(int id) {
    Document d = new Document();
    d.add(new StringField("id", Integer.toString(id), org.apache.lucene.document.Field.Store.YES));
    return d;
  }

  /** Doc-values updates of fields nothing has registered, and of a segment without the field. */
  static void dvOnly() throws Exception {
    try (ByteBuffersDirectory dir = new ByteBuffersDirectory()) {
      org.apache.lucene.index.IndexWriter w =
          new org.apache.lucene.index.IndexWriter(dir, new IndexWriterConfig());
      // updateNumericDocValue refuses a field that does not exist; updateDocValues creates it
      try {
        w.updateNumericDocValue(new Term("id", "0"), "n", 5);
        DV.append("updateNumericDocValue created n\n");
      } catch (IllegalArgumentException e) {
        DV.append("refused ").append(e.getMessage()).append('\n');
      }
      w.updateDocValues(
          new Term("id", "0"), new org.apache.lucene.document.NumericDocValuesField("n", 5));
      w.addDocument(idDoc(0));
      w.commit();
      Document one = idDoc(1);
      one.add(new org.apache.lucene.document.NumericDocValuesField("n", 1));
      w.addDocument(one);
      w.commit();
      w.updateNumericDocValue(new Term("id", "0"), "n", 7);
      w.updateDocValues(
          new Term("id", "1"), new org.apache.lucene.document.NumericDocValuesField("m", 3));
      w.updateDocValues(
          new Term("id", "0"),
          new org.apache.lucene.document.BinaryDocValuesField(
              "b", new org.apache.lucene.util.BytesRef("x")));
      w.commit();
      Document two = idDoc(2);
      two.add(new org.apache.lucene.document.NumericDocValuesField("m", 4));
      w.addDocument(two);
      w.close();
      try (org.apache.lucene.index.DirectoryReader r =
          org.apache.lucene.index.DirectoryReader.open(dir)) {
        java.util.TreeMap<String, String> docs = new java.util.TreeMap<>();
        for (org.apache.lucene.index.LeafReaderContext ctx : r.leaves()) {
          org.apache.lucene.index.LeafReader leaf = ctx.reader();
          for (int doc = 0; doc < leaf.maxDoc(); doc++) {
            String id = leaf.storedFields().document(doc).get("id");
            StringBuilder sb = new StringBuilder("doc " + id);
            for (String f : new String[] {"n", "m"}) {
              org.apache.lucene.index.NumericDocValues v = leaf.getNumericDocValues(f);
              sb.append(' ').append(f).append('=');
              sb.append(v != null && v.advanceExact(doc) ? Long.toString(v.longValue()) : "-");
            }
            org.apache.lucene.index.BinaryDocValues b = leaf.getBinaryDocValues("b");
            sb.append(" b=");
            sb.append(b != null && b.advanceExact(doc) ? b.binaryValue().utf8ToString() : "-");
            docs.put(id, sb.toString());
          }
        }
        for (String l : docs.values()) DV.append(l).append('\n');
        java.util.TreeMap<String, String> fields = new java.util.TreeMap<>();
        for (org.apache.lucene.index.FieldInfo fi :
            org.apache.lucene.index.FieldInfos.getMergedFieldInfos(r)) {
          fields.put(fi.name, "field " + fi.name + " " + fi.getDocValuesType() + " " + fi.getIndexOptions());
        }
        for (String l : fields.values()) DV.append(l).append('\n');
      }
    }
  }

  @RunWith(RandomizedRunner.class)
  public static class Dump extends LuceneTestCase {
    @Test
    public void testDump() throws Exception {
      for (long seed = 0; seed < SEEDS; seed++) {
        line("seed", seed);
        iwc(seed);
        if (seed < LIVE_SEEDS) live(seed);
      }
      for (long seed = 0; seed < RIW_SEEDS; seed++) {
        riw(seed);
      }
      dvOnly();
    }
  }

  public static void main(String[] args) throws Exception {
    // LuceneTestCase's rules refuse to run without assertions.
    ClassLoader.getSystemClassLoader().setDefaultAssertionStatus(true);
    Result result = JUnitCore.runClasses(Dump.class);
    for (Failure f : result.getFailures()) {
      System.err.println(f.getTrace());
    }
    if (!result.wasSuccessful() || result.getRunCount() != 1) {
      throw new AssertionError("GenTestFrameworkIndex: the dump did not run cleanly");
    }
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Files.writeString(out.resolve("index.txt"), OUT.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("riw.txt"), RIW.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("dvonly.txt"), DV.toString(), StandardCharsets.UTF_8);
  }
}
