import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SortFieldProvider;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.StringHelper;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.TreeMap;
import java.util.TreeSet;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.simpletext.SimpleTextCodec;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

/**
 * Reverse-direction check for lucene-codecs' {@code simpletext} postings: {@code
 * write_simpletext_fixture} (Rust) writes {@code _0_SimpleText_0.pst} (docs-only fields with the
 * empty term and with binary terms holding {@code \n}, {@code \} and {@code 0xff}; a freqs field
 * and a positions/offsets/payloads field, both with norms), and real Lucene opens it through its
 * own {@code SimpleTextCodec().postingsFormat()} with a hand-built {@link SegmentInfo}/{@link
 * FieldInfos}, then:
 *
 * <ul>
 *   <li>(at open) checks the {@code checksum} line, and runs {@code checkIntegrity};
 *   <li>walks every term with {@code next()}, requiring the exact term list, {@code
 *       docFreq}/{@code totalTermFreq}, and every document's freq, positions, offsets and payloads
 *       through {@code postings(ALL)} to equal what Rust wrote, and {@code seekExact}s each term
 *       on a fresh enum;
 *   <li>re-writes the file with its own {@code SimpleTextFieldsWriter} from its own reader and the
 *       norms Rust used, as {@code _0_SimpleText_1.pst}, and requires the two files to be equal
 *       byte for byte -- which pins every skip list, its levels and its competitive impacts.
 * </ul>
 *
 * <p>Then {@code codec/}: a Rust-written SimpleText index with no segments file -- {@code _0}
 * (index sort, diagnostics, attributes, every stored type, a {@code .liv}) and the compound
 * {@code _1} -- plus a {@code _2.inf} holding every kind of field. Each {@code .si}, {@code .inf},
 * {@code .liv}, {@code .scf} and {@code .fld} is read by SimpleTextCodec's own format and must
 * match Rust's manifest, then re-written by Java's writer: byte for byte equal, except the {@code
 * .si}, whose hash-ordered maps only compare as line sets. Java then commits a {@code segments_N}
 * over the Rust segments and opens them through {@link DirectoryReader}: max/num docs, live docs
 * and every stored value.
 *
 * <p>What it cannot catch: anything that binds the {@code .pst} into a segment ({@code
 * PerFieldPostingsFormat} attributes), since this port's IndexWriter does not route a field to
 * SimpleText; nor an {@code .si} whose maps are right but in an order Java would not write. The
 * Rust-written segments are not CheckIndex-consistent either: {@code _0}'s index sort names fields
 * absent from its {@code .inf}, and {@code _1} has blocks with no parent field -- the formats store
 * both as given, and only CheckIndex or an IndexWriter would object.
 */
public class VerifyCodecsSimpleText {
  static final String[] NAMES = {"id", "bin", "tag", "body"};

  public static void main(String[] args) throws Exception {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("manifest.properties"))) {
      int eq = line.indexOf('=');
      if (eq > 0) {
        manifest.put(line.substring(0, eq), line.substring(eq + 1));
      }
    }
    byte[] id = HexFormat.of().parseHex(manifest.get("id_hex"));
    int maxDoc = Integer.parseInt(manifest.get("max_doc"));

    IndexOptions[] options = {
      IndexOptions.DOCS,
      IndexOptions.DOCS,
      IndexOptions.DOCS_AND_FREQS,
      IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS
    };
    FieldInfo[] infos = new FieldInfo[NAMES.length];
    for (int i = 0; i < NAMES.length; i++) {
      infos[i] =
          new FieldInfo(
              NAMES[i],
              i,
              false,
              i < 2,
              i == 3,
              options[i],
              DocValuesType.NONE,
              DocValuesSkipIndexType.NONE,
              -1,
              new HashMap<>(),
              0,
              0,
              0,
              0,
              VectorEncoding.FLOAT32,
              VectorSimilarityFunction.EUCLIDEAN,
              false,
              false);
    }
    FieldInfos fis = new FieldInfos(infos);
    Map<String, long[]> norms = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("norms.tsv"))) {
      String[] e = line.split("\t");
      long[] values = new long[e.length - 1];
      for (int i = 1; i < e.length; i++) values[i - 1] = Long.parseLong(e[i]);
      norms.put(e[0], values);
    }
    Map<String, List<String>> byField = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("terms.tsv"))) {
      byField.computeIfAbsent(line.substring(0, line.indexOf('\t')), k -> new ArrayList<>())
          .add(line);
    }

    int failures = 0;
    PostingsFormat format = new SimpleTextCodec().postingsFormat();
    try (Directory directory = FSDirectory.open(dir)) {
      SegmentInfo si =
          new SegmentInfo(
              directory,
              Version.LATEST,
              Version.LATEST,
              "_0",
              maxDoc,
              false,
              false,
              new Lucene104Codec(),
              Collections.emptyMap(),
              id,
              new HashMap<>(),
              null);
      SegmentReadState read =
          new SegmentReadState(directory, si, fis, IOContext.DEFAULT, "SimpleText_0");
      try (FieldsProducer fields = format.fieldsProducer(read)) {
        fields.checkIntegrity();
        long termCount = 0;
        for (String field : List.of("bin", "body", "id", "tag")) {
          List<String> lines = byField.get(field);
          failures += verifyField(field, fields.terms(field), lines);
          termCount += lines.size();
        }
        System.out.println("SimpleText_0: " + termCount + " terms, ok");

        // Idempotent, so a failing run can be re-run to show its output.
        Files.deleteIfExists(dir.resolve("_0_SimpleText_1.pst"));
        SegmentWriteState write =
            new SegmentWriteState(
                InfoStream.NO_OUTPUT, directory, si, fis, null, IOContext.DEFAULT, "SimpleText_1");
        try (FieldsConsumer consumer = format.fieldsConsumer(write)) {
          consumer.write(fields, new ArrayNorms(norms));
        }
      }
    }
    byte[] rust = Files.readAllBytes(dir.resolve("_0_SimpleText_0.pst"));
    byte[] java = Files.readAllBytes(dir.resolve("_0_SimpleText_1.pst"));
    if (!Arrays.equals(rust, java)) {
      System.out.println("re-written .pst differs: Rust " + rust.length + " bytes, Java "
          + java.length + ", first difference at " + Arrays.mismatch(rust, java));
      failures++;
    }
    failures += verifyCodec(dir.resolve("codec"));
    if (failures > 0) {
      System.out.println(failures + " failure(s)");
      System.exit(1);
    }
    System.out.println("SimpleText postings and segment formats verified against real Lucene. PASS");
  }

  /** Norms out of {@code norms.tsv}, every document holding one. */
  static final class ArrayNorms extends NormsProducer {
    final Map<String, long[]> norms;

    ArrayNorms(Map<String, long[]> norms) {
      this.norms = norms;
    }

    @Override
    public NumericDocValues getNorms(FieldInfo field) {
      long[] values = norms.get(field.name);
      if (values == null) throw new IllegalStateException("no norms for " + field.name);
      return new NumericDocValues() {
        int doc = -1;

        @Override
        public long longValue() {
          return values[doc];
        }

        @Override
        public boolean advanceExact(int target) {
          doc = target;
          return target < values.length;
        }

        @Override
        public int docID() {
          return doc;
        }

        @Override
        public int nextDoc() {
          return advance(doc + 1);
        }

        @Override
        public int advance(int target) {
          doc = target < values.length ? target : DocIdSetIterator.NO_MORE_DOCS;
          return doc;
        }

        @Override
        public long cost() {
          return values.length;
        }
      };
    }

    @Override
    public void checkIntegrity() {}

    @Override
    public void close() {}
  }

  static String hex(BytesRef b) {
    return HexFormat.of().formatHex(b.bytes, b.offset, b.offset + b.length);
  }

  /** {@code write_simpletext_fixture}'s line for the enum's current term. */
  static String termLine(String field, Terms terms, TermsEnum te) throws Exception {
    StringBuilder sb = new StringBuilder(field);
    sb.append('\t').append(hex(te.term())).append('\t').append(te.docFreq()).append('\t')
        .append(te.totalTermFreq());
    PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
    for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
      int freq = pe.freq();
      sb.append('\t').append(doc).append(':').append(freq);
      if (terms.hasPositions()) {
        StringBuilder pos = new StringBuilder();
        StringBuilder offs = new StringBuilder();
        StringBuilder pays = new StringBuilder();
        for (int k = 0; k < freq; k++) {
          pos.append(k == 0 ? "" : ".").append(pe.nextPosition());
          offs.append(k == 0 ? "" : ".").append(pe.startOffset()).append('-')
              .append(pe.endOffset());
          BytesRef pl = pe.getPayload();
          pays.append(k == 0 ? "" : ".").append(pl == null || pl.length == 0 ? "-" : hex(pl));
        }
        sb.append(':').append(pos).append(':').append(offs).append(':').append(pays);
      }
    }
    return sb.toString();
  }

  static int verifyField(String field, Terms terms, List<String> lines) throws Exception {
    int failures = 0;
    TermsEnum walk = terms.iterator();
    for (String line : lines) {
      String[] e = line.split("\t", 5);
      BytesRef want = new BytesRef(HexFormat.of().parseHex(e[1]));
      BytesRef got = walk.next();
      if (got == null || !got.equals(want)) {
        System.out.println(field + " next() " + got + " expected " + want);
        return failures + 1;
      }
      String gotLine = termLine(field, terms, walk);
      if (!gotLine.equals(line)) {
        System.out.println(field + " " + e[1] + "\n  got  " + gotLine + "\n  want " + line);
        failures++;
      }
      if (!terms.iterator().seekExact(want)) {
        System.out.println(field + " seekExact missed " + e[1]);
        failures++;
      }
    }
    if (walk.next() != null) {
      System.out.println("extra term in " + field);
      failures++;
    }
    return failures;
  }

  // ---- the per-segment formats: .si, .inf, .liv, .scf, .fld ------------------------------

  static final HexFormat HEX = HexFormat.of();
  static final IOContext CTX = IOContext.DEFAULT;

  static String hexs(String s) {
    return HEX.formatHex(s.getBytes(StandardCharsets.UTF_8));
  }

  /** Lines that must equal a Rust-written manifest. */
  static int expect(Path manifest, List<String> got) throws Exception {
    List<String> want = Files.readAllLines(manifest);
    if (want.equals(got)) return 0;
    for (int i = 0; i < Math.max(want.size(), got.size()); i++) {
      String w = i < want.size() ? want.get(i) : "<none>";
      String g = i < got.size() ? got.get(i) : "<none>";
      if (!w.equals(g)) {
        System.out.println(manifest.getFileName() + " line " + i + "\n  got  " + g + "\n  want " + w);
        break;
      }
    }
    return 1;
  }

  /** Two files that must be equal byte for byte. */
  static int sameBytes(Directory a, String an, Directory b, String bn) throws Exception {
    byte[] x = readAll(a, an);
    byte[] y = readAll(b, bn);
    if (Arrays.equals(x, y)) return 0;
    System.out.println(an + " (Rust " + x.length + " bytes) vs Java's re-write " + bn + " ("
        + y.length + " bytes): first difference at " + Arrays.mismatch(x, y));
    return 1;
  }

  static byte[] readAll(Directory d, String name) throws Exception {
    try (IndexInput in = d.openInput(name, IOContext.READONCE)) {
      byte[] b = new byte[Math.toIntExact(in.length())];
      in.readBytes(b, 0, b.length);
      return b;
    }
  }

  static List<String> sortedLines(byte[] b) {
    List<String> lines = new ArrayList<>(List.of(new String(b, StandardCharsets.UTF_8).split("\n")));
    lines.removeIf(l -> l.startsWith("checksum "));
    Collections.sort(lines);
    return lines;
  }

  static void sortedMap(List<String> lines, String tag, Map<String, String> map) {
    for (Map.Entry<String, String> e : new TreeMap<>(map).entrySet()) {
      lines.add(tag + "\t" + hexs(e.getKey()) + "\t" + hexs(e.getValue()));
    }
  }

  /** The .si as {@code GenCodecsSimpleTextCodec}'s {@code si.SEG.tsv} records it. */
  static List<String> segmentInfoLines(SegmentInfo si) throws Exception {
    List<String> lines = new ArrayList<>();
    lines.add("version\t" + si.getVersion());
    lines.add("minVersion\t" + si.getMinVersion());
    lines.add("docCount\t" + si.maxDoc());
    lines.add("compound\t" + si.getUseCompoundFile());
    lines.add("hasBlocks\t" + si.getHasBlocks());
    sortedMap(lines, "diag", si.getDiagnostics());
    sortedMap(lines, "att", si.getAttributes());
    for (String f : new TreeSet<>(si.files())) lines.add("file\t" + f);
    lines.add("id\t" + HEX.formatHex(si.getId()));
    Sort sort = si.getIndexSort();
    if (sort != null) {
      for (SortField sf : sort.getSort()) {
        ByteBuffersDataOutput b = new ByteBuffersDataOutput();
        SortFieldProvider.write(sf, b);
        lines.add("sort\t" + sf.getIndexSorter().getProviderName() + "\t" + hexs(sf.toString())
            + "\t" + HEX.formatHex(b.toArrayCopy()));
      }
    }
    return lines;
  }

  /** Every property of every FieldInfo, as {@code fields.FILE.tsv} records it. */
  static List<String> fieldInfoLines(FieldInfos fis) {
    List<String> lines = new ArrayList<>();
    for (FieldInfo fi : fis) {
      StringBuilder atts = new StringBuilder();
      for (Map.Entry<String, String> e : new TreeMap<>(fi.attributes()).entrySet()) {
        if (atts.length() > 0) atts.append(',');
        atts.append(hexs(e.getKey())).append('=').append(hexs(e.getValue()));
      }
      lines.add(String.join("\t", hexs(fi.name), Integer.toString(fi.number),
          fi.getIndexOptions().toString(), Boolean.toString(fi.hasTermVectors()),
          Boolean.toString(fi.hasPayloads()), Boolean.toString(fi.omitsNorms()),
          fi.getDocValuesType().toString(), fi.docValuesSkipIndexType().toString(),
          Long.toString(fi.getDocValuesGen()), atts.toString(),
          Integer.toString(fi.getPointDimensionCount()),
          Integer.toString(fi.getPointIndexDimensionCount()),
          Integer.toString(fi.getPointNumBytes()), Integer.toString(fi.getVectorDimension()),
          fi.getVectorEncoding().name(), fi.getVectorSimilarityFunction().name(),
          Boolean.toString(fi.isSoftDeletesField()), Boolean.toString(fi.isParentField())));
    }
    return lines;
  }

  /** Every stored value as a {@code stored.SEG.tsv} line; optionally re-written by Java. */
  static final class Dump extends StoredFieldVisitor {
    final List<String> lines;
    final StoredFieldsWriter copy;

    Dump(List<String> lines, StoredFieldsWriter copy) {
      this.lines = lines;
      this.copy = copy;
    }

    @Override
    public Status needsField(FieldInfo fieldInfo) {
      return Status.YES;
    }

    void add(FieldInfo fi, String type, String value) {
      lines.add(fi.number + "\t" + type + "\t" + value);
    }

    @Override
    public void stringField(FieldInfo fi, String value) throws IOException {
      add(fi, "string", hexs(value));
      if (copy != null) copy.writeField(fi, value);
    }

    @Override
    public void binaryField(FieldInfo fi, byte[] value) throws IOException {
      add(fi, "binary", HEX.formatHex(value));
      if (copy != null) copy.writeField(fi, new BytesRef(value));
    }

    @Override
    public void intField(FieldInfo fi, int value) throws IOException {
      add(fi, "int", Integer.toString(value));
      if (copy != null) copy.writeField(fi, value);
    }

    @Override
    public void longField(FieldInfo fi, long value) throws IOException {
      add(fi, "long", Long.toString(value));
      if (copy != null) copy.writeField(fi, value);
    }

    @Override
    public void floatField(FieldInfo fi, float value) throws IOException {
      add(fi, "float",
          String.format("%08x", Float.floatToRawIntBits(value)) + "\t" + Float.toString(value));
      if (copy != null) copy.writeField(fi, value);
    }

    @Override
    public void doubleField(FieldInfo fi, double value) throws IOException {
      add(fi, "double", String.format("%016x", Double.doubleToRawLongBits(value)) + "\t"
          + Double.toString(value));
      if (copy != null) copy.writeField(fi, value);
    }
  }

  static String bitString(Bits live) {
    StringBuilder sb = new StringBuilder();
    for (int d = 0; d < live.length(); d++) sb.append(live.get(d) ? '1' : '0');
    return sb.toString();
  }

  static void clean(Path p) throws IOException {
    if (Files.exists(p)) {
      try (var walk = Files.walk(p)) {
        walk.sorted((a, b) -> b.compareTo(a)).forEach(f -> f.toFile().delete());
      }
    }
    Files.createDirectories(p);
  }

  /**
   * {@code codec/}: Rust's {@code .si}/{@code .inf}/{@code .liv}/{@code .scf}/{@code .fld}, read
   * by SimpleTextCodec's own formats and re-written by its own writers, then bound into a commit
   * Java writes and opened through {@link DirectoryReader}.
   */
  static int verifyCodec(Path root) throws Exception {
    Codec codec = new SimpleTextCodec();
    int failures = 0;
    Path javaDir = root.resolve("java");
    clean(javaDir);
    List<String> segRows = Files.readAllLines(root.resolve("segments.tsv"));
    SegmentInfos sis = new SegmentInfos(Version.LATEST.major);
    Map<String, int[]> expected = new TreeMap<>();
    try (Directory dir = FSDirectory.open(root.resolve("index"));
        Directory out = FSDirectory.open(javaDir)) {
      for (String f : dir.listAll()) {
        if (f.startsWith("segments") || f.startsWith("pending_segments")) dir.deleteFile(f);
      }
      for (String row : segRows) {
        String[] r = row.split("\t");
        String name = r[0];
        int maxDoc = Integer.parseInt(r[1]);
        int delCount = Integer.parseInt(r[2]);
        long delGen = Long.parseLong(r[3]);
        byte[] id = HEX.parseHex(r[4]);
        expected.put(name, new int[] {maxDoc, delCount});

        SegmentInfo si = codec.segmentInfoFormat().read(dir, name, id, CTX);
        failures += expect(root.resolve("si." + name + ".tsv"), segmentInfoLines(si));
        si.setCodec(codec);
        codec.segmentInfoFormat().write(out, si, CTX);
        // Diagnostics, attributes and files are hash maps/sets in Java: same lines, any order.
        if (!sortedLines(readAll(dir, name + ".si")).equals(sortedLines(readAll(out, name + ".si")))) {
          System.out.println(name + ".si: Java's re-write has different lines");
          failures++;
        }

        CompoundDirectory cfs =
            si.getUseCompoundFile() ? codec.compoundFormat().getCompoundReader(dir, si) : null;
        Directory core = cfs == null ? dir : cfs;
        if (cfs != null) {
          List<String> lines = new ArrayList<>();
          Path subPath = root.resolve("java-cfs-" + name);
          clean(subPath);
          try (Directory sub = FSDirectory.open(subPath)) {
            for (String f : cfs.listAll()) {
              lines.add(f + "\t" + cfs.fileLength(f));
              sub.copyFrom(cfs, f, f, CTX);
            }
            failures += expect(root.resolve("cfs." + name + ".tsv"), lines);
            SegmentInfo w = new SegmentInfo(sub, si.getVersion(), si.getMinVersion(), name, maxDoc,
                false, false, codec, Map.of(), id, Map.of(), null);
            w.setFiles(Arrays.asList(cfs.listAll()));
            codec.compoundFormat().write(sub, w, CTX);
            failures += sameBytes(dir, name + ".scf", sub, name + ".scf");
          }
        }

        FieldInfos fis = codec.fieldInfosFormat().read(core, si, "", CTX);
        failures += expect(root.resolve("fields." + name + ".inf.tsv"), fieldInfoLines(fis));
        codec.fieldInfosFormat().write(out, si, "", fis, CTX);
        failures += sameBytes(core, name + ".inf", out, name + ".inf");

        List<String> stored = new ArrayList<>();
        try (StoredFieldsReader reader = codec.storedFieldsFormat().fieldsReader(core, si, fis, CTX);
            StoredFieldsWriter writer = codec.storedFieldsFormat().fieldsWriter(out, si, CTX)) {
          reader.checkIntegrity();
          for (int d = 0; d < maxDoc; d++) {
            stored.add("doc\t" + d);
            writer.startDocument();
            reader.document(d, new Dump(stored, writer));
            writer.finishDocument();
          }
          writer.finish(maxDoc);
        }
        failures += expect(root.resolve("stored." + name + ".tsv"), stored);
        failures += sameBytes(core, name + ".fld", out, name + ".fld");

        SegmentCommitInfo sci =
            new SegmentCommitInfo(si, delCount, 0, delGen, -1, -1, StringHelper.randomId());
        if (delGen != -1) {
          Bits live = codec.liveDocsFormat().readLiveDocs(dir, sci, CTX);
          failures += expect(root.resolve("live." + name + ".tsv"),
              List.of(Integer.toString(live.length()), bitString(live)));
          codec.liveDocsFormat().writeLiveDocs(live, out, sci, 0, CTX);
          failures += sameBytes(dir, IndexFileNames.fileNameFromGeneration(name, "liv", delGen),
              out, IndexFileNames.fileNameFromGeneration(name, "liv", sci.getNextDelGen()));
        }
        if (cfs != null) cfs.close();
        sis.add(sci);
      }
      sis.counter = segRows.size();
      sis.commit(dir);

      // The whole index, opened the way a searcher would.
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != segRows.size()) {
          System.out.println("DirectoryReader: " + reader.leaves().size() + " leaves");
          failures++;
        }
        for (LeafReaderContext ctx : reader.leaves()) {
          SegmentReader sr = (SegmentReader) ctx.reader();
          String name = sr.getSegmentName();
          int[] e = expected.get(name);
          if (sr.maxDoc() != e[0] || sr.numDocs() != e[0] - e[1]) {
            System.out.println(name + ": maxDoc " + sr.maxDoc() + " numDocs " + sr.numDocs());
            failures++;
          }
          Bits live = sr.getLiveDocs();
          if (e[1] > 0) {
            List<String> want = Files.readAllLines(root.resolve("live." + name + ".tsv"));
            if (live == null || !bitString(live).equals(want.get(1))) {
              System.out.println(name + ": DirectoryReader's live docs differ");
              failures++;
            }
          } else if (live != null) {
            System.out.println(name + ": unexpected live docs");
            failures++;
          }
          List<String> stored = new ArrayList<>();
          StoredFields sf = sr.storedFields();
          for (int d = 0; d < sr.maxDoc(); d++) {
            stored.add("doc\t" + d);
            sf.document(d, new Dump(stored, null));
          }
          failures += expect(root.resolve("stored." + name + ".tsv"), stored);
        }
        System.out.println("codec: " + reader.leaves().size() + " segments, " + reader.maxDoc()
            + " docs (" + reader.numDocs() + " live) opened through DirectoryReader");
      }
    }

    // A field-infos file holding every kind of field, outside the index.
    try (Directory d2 = FSDirectory.open(root.resolve("infos"));
        Directory o2 = FSDirectory.open(javaDir)) {
      SegmentInfo si2 = new SegmentInfo(d2, Version.LATEST, Version.LATEST, "_2", 1, false, false,
          codec, Map.of(), new byte[16], Map.of(), null);
      FieldInfos fis = codec.fieldInfosFormat().read(d2, si2, "", CTX);
      failures += expect(root.resolve("fields._2.inf.tsv"), fieldInfoLines(fis));
      codec.fieldInfosFormat().write(o2, si2, "", fis, CTX);
      failures += sameBytes(d2, "_2.inf", o2, "_2.inf");
      System.out.println("codec: _2.inf, " + fis.size() + " fields of every kind");
    }
    return failures;
  }
}
