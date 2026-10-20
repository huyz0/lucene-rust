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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.SegmentInfoFormat;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.simpletext.SimpleTextCodec;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortFieldProvider;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.search.SortedSetSortField;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;

/**
 * lucene-codecs' {@code org.apache.lucene.codecs.simpletext}: the per-segment formats of a whole
 * {@code SimpleTextCodec} index -- segment info ({@code .si}), field infos ({@code .inf}), live
 * docs ({@code .liv}), compound file ({@code .scf}) and stored fields ({@code .fld}).
 *
 * <p>{@code flush/}: segments flushed by an {@code IndexWriter} on {@code SimpleTextCodec}, with
 * an index sort over three providers, a soft-deletes field, a parent field (document blocks), a
 * doc-values update (a field-infos generation), hard deletes, and fields of every kind: index
 * options, payloads, term vectors, norms omitted, every doc-values type and a skip index, points
 * (index dimensions below data dimensions), byte and float vectors, field-type attributes. The
 * codec is SimpleText under a {@code FilterCodec} of the same name whose segment-info writer adds
 * an attribute first. Stored values of every type: strings with newlines, backslashes and
 * supplementary characters, binary with 0x0a/0x5c/0x00/0xff, int/long extremes, floats and
 * doubles from random bits and the special values.
 *
 * <p>{@code merged/}: three flushed segments force-merged into one compound segment ({@code
 * .scf}), then a doc-values update and a delete on it.
 *
 * <pre>
 *   codecs_simpletext/codec/{flush,merged}/
 *     index/                 the index
 *     segments.tsv           per segment: name, maxDoc, delCount, softDelCount, delGen,
 *                            fieldInfosGen
 *     si.SEG.tsv             the .si as read: version, minVersion, docCount, compound, hasBlocks,
 *                            diag/att (hex key, hex value, by key), file (by name), id, sort
 *                            (provider, hex toString, hex bytes)
 *     fields.FILE.tsv        per FieldInfo of one .inf: every property, attributes by key (hex)
 *     live.SEG.tsv           size, then a 0/1 string of the live docs
 *     stored.SEG.tsv         per document "doc N", then per value: number, type, value (strings
 *                            and bytes hex; floats and doubles raw bits hex and toString)
 *     cfs.SEG.tsv            per compound entry: name, length
 * </pre>
 */
public class GenCodecsSimpleTextCodec {
  static final HexFormat HEX = HexFormat.of();
  static final IOContext CTX = IOContext.DEFAULT;

  /** SimpleText with an attribute added to every segment info it writes. */
  static Codec codec() {
    Codec simpleText = new SimpleTextCodec();
    SegmentInfoFormat si = simpleText.segmentInfoFormat();
    return new FilterCodec("SimpleText", simpleText) {
      @Override
      public SegmentInfoFormat segmentInfoFormat() {
        return new SegmentInfoFormat() {
          @Override
          public SegmentInfo read(Directory d, String name, byte[] id, IOContext c)
              throws IOException {
            return si.read(d, name, id, c);
          }

          @Override
          public void write(Directory d, SegmentInfo info, IOContext c) throws IOException {
            info.putAttribute("gen.note", "line\nbreak \\ é");
            si.write(d, info, c);
          }
        };
      }
    };
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("codecs_simpletext").resolve("codec");
    Files.createDirectories(root);
    flush(root.resolve("flush"));
    merged(root.resolve("merged"));
  }

  static void clean(Path out) throws IOException {
    if (Files.exists(out)) {
      try (var walk = Files.walk(out)) {
        walk.sorted((a, b) -> b.compareTo(a)).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
  }

  /** A token stream whose every token carries a payload, some needing escapes. */
  static final class PayStream extends TokenStream {
    final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    final PayloadAttribute payload = addAttribute(PayloadAttribute.class);
    final String[] words;
    int upto;

    PayStream(String... words) {
      this.words = words;
    }

    @Override
    public boolean incrementToken() {
      if (upto == words.length) return false;
      clearAttributes();
      String w = words[upto++];
      term.setEmpty().append(w);
      payload.setPayload(new BytesRef(new byte[] {0x0a, (byte) upto, 0x5c}));
      return true;
    }

    @Override
    public void reset() throws IOException {
      super.reset();
      upto = 0;
    }
  }

  static final String[] STRINGS = {
    "", "plain", "line\nbreak", "back\\slash\\", "\\\n\\n", "tab\tcr\r", "café 中文",
    "😀 emoji", "end\n",
  };

  static float randomFloat(Random r) {
    float f;
    do {
      f = Float.intBitsToFloat(r.nextInt());
    } while (Float.isNaN(f));
    return f;
  }

  static double randomDouble(Random r) {
    double d;
    do {
      d = Double.longBitsToDouble(r.nextLong());
    } while (Double.isNaN(d));
    return d;
  }

  static final float[] FLOATS = {
    Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 0f, -0f, Float.MIN_VALUE,
    Float.MAX_VALUE, Float.MIN_NORMAL, 1e7f, 9999999f, 0.001f, 0.00099999f, 1e-4f, 1f, -1.5f,
    3.4e38f, 1.0E10f, 100f, 2e-45f, 8.5e-44f,
  };
  static final double[] DOUBLES = {
    Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 0d, -0d, Double.MIN_VALUE,
    Double.MAX_VALUE, Double.MIN_NORMAL, 1e7, 9999999.999, 0.001, 9.99e-4, 1e-4, 1d, -2.5,
    1e23, 2e-323, 5e-324, 1.7976931348623157E308, 4.35, 0.1, 1.0E21,
  };

  /** One document's stored values, by its index {@code i}. */
  static void stored(Document doc, int i, Random r) {
    doc.add(new StoredField("s", STRINGS[i % STRINGS.length]));
    if (i % 3 == 0) {
      byte[] b = new byte[r.nextInt(7)];
      for (int k = 0; k < b.length; k++) {
        int c = r.nextInt(5);
        b[k] = (byte) (c == 0 ? 0x0a : c == 1 ? 0x5c : c == 2 ? 0x00 : c == 3 ? 0xff : r.nextInt());
      }
      doc.add(new StoredField("b", new BytesRef(b)));
    }
    int[] ints = {Integer.MIN_VALUE, Integer.MAX_VALUE, 0, -1, r.nextInt()};
    doc.add(new StoredField("i", ints[i % ints.length]));
    long[] longs = {Long.MIN_VALUE, Long.MAX_VALUE, 0, -1, r.nextLong()};
    doc.add(new StoredField("l", longs[i % longs.length]));
    doc.add(new StoredField("f", i < FLOATS.length ? FLOATS[i] : randomFloat(r)));
    doc.add(new StoredField("f", randomFloat(r)));
    doc.add(new StoredField("d", i < DOUBLES.length ? DOUBLES[i] : randomDouble(r)));
    doc.add(new StoredField("d", randomDouble(r)));
    if (i % 4 == 1) {
      doc.add(new StoredField("s", "second " + i));
    }
  }

  static final FieldType FREQS = new FieldType(TextField.TYPE_NOT_STORED);
  static final FieldType OFFSETS = new FieldType(TextField.TYPE_NOT_STORED);
  static final FieldType VECTORS = new FieldType(TextField.TYPE_NOT_STORED);
  static final FieldType PAYLOADS = new FieldType(TextField.TYPE_NOT_STORED);
  static final FieldType DIMS = new FieldType();
  static final FieldType ATTRS = new FieldType(StringField.TYPE_NOT_STORED);

  static {
    FREQS.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    FREQS.freeze();
    OFFSETS.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    OFFSETS.freeze();
    VECTORS.setStoreTermVectors(true);
    VECTORS.setStoreTermVectorPositions(true);
    VECTORS.setStoreTermVectorOffsets(true);
    VECTORS.freeze();
    PAYLOADS.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    PAYLOADS.freeze();
    DIMS.setDimensions(3, 2, 4);
    DIMS.freeze();
    ATTRS.putAttribute("k1", "v\n1");
    ATTRS.putAttribute("k2\\", "");
    ATTRS.putAttribute("a0", "x");
    ATTRS.freeze();
  }

  /** Every kind of field, then the stored values. */
  static Document doc(int i, Random r, boolean everything) {
    Document doc = new Document();
    doc.add(new StringField("id", "id" + i, Field.Store.YES));
    stored(doc, i, r);
    if (!everything) return doc;
    doc.add(new Field("freqs", "w" + (i % 7) + " w" + (i % 3), FREQS));
    doc.add(new Field("offs", "o" + (i % 5), OFFSETS));
    doc.add(new Field("tv", "t" + (i % 4) + " t1", VECTORS));
    doc.add(new Field("pay", new PayStream("p" + (i % 3), "q"), PAYLOADS));
    doc.add(new Field("attr", "a" + (i % 2), ATTRS));
    doc.add(new NumericDocValuesField("num", r.nextInt(50) - 25));
    doc.add(new NumericDocValuesField("upd", i));
    doc.add(new SortedNumericDocValuesField("snum", r.nextInt(10)));
    doc.add(new SortedNumericDocValuesField("snum", r.nextInt(10)));
    doc.add(new SortedSetDocValuesField("sset", new BytesRef("v" + r.nextInt(6))));
    doc.add(new SortedDocValuesField("sdv", new BytesRef("s" + (i % 9))));
    doc.add(new BinaryDocValuesField("bdv", new BytesRef(new byte[] {(byte) i, 0x0a})));
    doc.add(NumericDocValuesField.indexedField("skip", i));
    doc.add(new IntPoint("ipt", i, -i));
    byte[] dims = new byte[12];
    r.nextBytes(dims);
    doc.add(new Field("dims", new BytesRef(dims), DIMS));
    doc.add(
        new KnnFloatVectorField(
            "fvec", new float[] {1 + i, 2, -3, 0.5f}, VectorSimilarityFunction.COSINE));
    doc.add(
        new KnnByteVectorField(
            "bvec", new byte[] {(byte) i, 1, -1}, VectorSimilarityFunction.DOT_PRODUCT));
    doc.add(
        new KnnFloatVectorField(
            "mvec", new float[] {i, 1}, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT));
    return doc;
  }

  static void flush(Path out) throws IOException {
    clean(out);
    Path idx = out.resolve("index");
    Random r = new Random(0x51_2026_1020L);
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setCodec(codec());
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setMaxBufferedDocs(100000);
    cfg.setRAMBufferSizeMB(256);
    cfg.setSoftDeletesField("soft");
    cfg.setParentField("parent");
    SortField num = new SortField("num", SortField.Type.LONG, true);
    num.setMissingValue(Long.MIN_VALUE);
    cfg.setIndexSort(
        new Sort(
            num,
            new SortedNumericSortField(
                "snum", SortField.Type.INT, false, SortedNumericSelector.Type.MAX),
            new SortedSetSortField("sset", true, SortedSetSelector.Type.MIN)));
    try (Directory dir = FSDirectory.open(idx);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      int i = 0;
      while (i < 150) {
        if (i % 10 == 0) {
          // a block: two children and their parent
          List<Document> block = new ArrayList<>();
          for (int k = 0; k < 3; k++) {
            block.add(doc(i++, r, true));
          }
          w.addDocuments(block);
        } else if (i % 17 == 0) {
          Document empty = new Document();
          empty.add(new StringField("id", "id" + i++, Field.Store.NO));
          w.addDocument(empty);
        } else {
          w.addDocument(doc(i++, r, true));
        }
      }
      w.commit();
      w.deleteDocuments(new Term("id", "id5"), new Term("id", "id77"), new Term("id", "id149"));
      w.updateNumericDocValue(new Term("id", "id8"), "upd", 1000L);
      w.softUpdateDocument(
          new Term("id", "id9"), doc(9, r, true), new NumericDocValuesField("soft", 1));
      w.commit();
    }
    manifests(out, idx);
  }

  static void merged(Path out) throws IOException {
    clean(out);
    Path idx = out.resolve("index");
    Random r = new Random(0x52_2026_1020L);
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setCodec(codec());
    cfg.setUseCompoundFile(true);
    TieredMergePolicy tmp = new TieredMergePolicy();
    tmp.setSegmentsPerTier(50);
    tmp.setNoCFSRatio(1.0);
    cfg.setMergePolicy(tmp);
    cfg.setMaxBufferedDocs(100000);
    cfg.setRAMBufferSizeMB(256);
    try (Directory dir = FSDirectory.open(idx);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      int i = 0;
      for (int seg = 0; seg < 3; seg++) {
        for (int k = 0; k < 60; k++) {
          w.addDocument(doc(i++, r, k % 2 == 0));
        }
        w.commit();
      }
      w.forceMerge(1);
      w.commit();
      w.deleteDocuments(new Term("id", "id3"), new Term("id", "id100"));
      w.updateNumericDocValue(new Term("id", "id4"), "upd", -7L);
      w.commit();
    }
    manifests(out, idx);
  }

  static String hex(String s) {
    return HEX.formatHex(s.getBytes(StandardCharsets.UTF_8));
  }

  static void manifests(Path out, Path idx) throws IOException {
    Codec codec = new SimpleTextCodec();
    List<String> segments = new ArrayList<>();
    try (Directory dir = FSDirectory.open(idx)) {
      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      for (SegmentCommitInfo sci : sis) {
        SegmentInfo si = sci.info;
        segments.add(
            String.join(
                "\t",
                si.name,
                Integer.toString(si.maxDoc()),
                Integer.toString(sci.getDelCount()),
                Integer.toString(sci.getSoftDelCount()),
                Long.toString(sci.getDelGen()),
                Long.toString(sci.getFieldInfosGen())));
        segmentInfo(out.resolve("si." + si.name + ".tsv"), si);
        CompoundDirectory cfs =
            si.getUseCompoundFile() ? codec.compoundFormat().getCompoundReader(dir, si) : null;
        Directory core = cfs == null ? dir : cfs;
        if (cfs != null) {
          List<String> lines = new ArrayList<>();
          for (String name : cfs.listAll()) {
            lines.add(name + "\t" + cfs.fileLength(name));
          }
          Files.write(out.resolve("cfs." + si.name + ".tsv"), lines);
        }
        FieldInfos fis = codec.fieldInfosFormat().read(core, si, "", CTX);
        fieldInfos(out.resolve("fields." + si.name + ".inf.tsv"), fis);
        if (sci.getFieldInfosGen() != -1) {
          String suffix = Long.toString(sci.getFieldInfosGen(), Character.MAX_RADIX);
          fis = codec.fieldInfosFormat().read(dir, si, suffix, CTX);
          fieldInfos(out.resolve("fields." + si.name + "_" + suffix + ".inf.tsv"), fis);
        }
        if (sci.hasDeletions()) {
          Bits live = codec.liveDocsFormat().readLiveDocs(dir, sci, CTX);
          StringBuilder sb = new StringBuilder();
          for (int d = 0; d < live.length(); d++) {
            sb.append(live.get(d) ? '1' : '0');
          }
          Files.write(
              out.resolve("live." + si.name + ".tsv"),
              List.of(Integer.toString(live.length()), sb.toString()));
        }
        try (StoredFieldsReader reader =
            codec.storedFieldsFormat().fieldsReader(core, si, fis, CTX)) {
          List<String> lines = new ArrayList<>();
          for (int d = 0; d < si.maxDoc(); d++) {
            lines.add("doc\t" + d);
            reader.document(d, new Dump(lines));
          }
          Files.write(out.resolve("stored." + si.name + ".tsv"), lines);
        }
        if (cfs != null) cfs.close();
      }
    }
    Files.write(out.resolve("segments.tsv"), segments);
  }

  /** Every stored value, in file order. */
  static final class Dump extends StoredFieldVisitor {
    final List<String> lines;

    Dump(List<String> lines) {
      this.lines = lines;
    }

    @Override
    public Status needsField(FieldInfo fieldInfo) {
      return Status.YES;
    }

    void add(FieldInfo fi, String type, String value) {
      lines.add(fi.number + "\t" + type + "\t" + value);
    }

    @Override
    public void stringField(FieldInfo fi, String value) {
      add(fi, "string", hex(value));
    }

    @Override
    public void binaryField(FieldInfo fi, byte[] value) {
      add(fi, "binary", HEX.formatHex(value));
    }

    @Override
    public void intField(FieldInfo fi, int value) {
      add(fi, "int", Integer.toString(value));
    }

    @Override
    public void longField(FieldInfo fi, long value) {
      add(fi, "long", Long.toString(value));
    }

    @Override
    public void floatField(FieldInfo fi, float value) {
      add(
          fi,
          "float",
          String.format("%08x", Float.floatToRawIntBits(value)) + "\t" + Float.toString(value));
    }

    @Override
    public void doubleField(FieldInfo fi, double value) {
      add(
          fi,
          "double",
          String.format("%016x", Double.doubleToRawLongBits(value))
              + "\t"
              + Double.toString(value));
    }
  }

  static void sortedMap(List<String> lines, String tag, Map<String, String> map) {
    for (Map.Entry<String, String> e : new TreeMap<>(map).entrySet()) {
      lines.add(tag + "\t" + hex(e.getKey()) + "\t" + hex(e.getValue()));
    }
  }

  static void segmentInfo(Path path, SegmentInfo si) throws IOException {
    List<String> lines = new ArrayList<>();
    lines.add("version\t" + si.getVersion());
    lines.add("minVersion\t" + si.getMinVersion());
    lines.add("docCount\t" + si.maxDoc());
    lines.add("compound\t" + si.getUseCompoundFile());
    lines.add("hasBlocks\t" + si.getHasBlocks());
    sortedMap(lines, "diag", si.getDiagnostics());
    sortedMap(lines, "att", si.getAttributes());
    for (String f : new java.util.TreeSet<>(si.files())) {
      lines.add("file\t" + f);
    }
    lines.add("id\t" + HEX.formatHex(si.getId()));
    Sort sort = si.getIndexSort();
    if (sort != null) {
      for (SortField sf : sort.getSort()) {
        ByteBuffersDataOutput b = new ByteBuffersDataOutput();
        SortFieldProvider.write(sf, b);
        lines.add(
            "sort\t"
                + sf.getIndexSorter().getProviderName()
                + "\t"
                + hex(sf.toString())
                + "\t"
                + HEX.formatHex(b.toArrayCopy()));
      }
    }
    Files.write(path, lines);
  }

  static void fieldInfos(Path path, FieldInfos fis) throws IOException {
    List<String> lines = new ArrayList<>();
    for (FieldInfo fi : fis) {
      StringBuilder atts = new StringBuilder();
      for (Map.Entry<String, String> e : new TreeMap<>(fi.attributes()).entrySet()) {
        if (atts.length() > 0) atts.append(',');
        atts.append(hex(e.getKey())).append('=').append(hex(e.getValue()));
      }
      lines.add(
          String.join(
              "\t",
              hex(fi.name),
              Integer.toString(fi.number),
              fi.getIndexOptions().toString(),
              Boolean.toString(fi.hasTermVectors()),
              Boolean.toString(fi.hasPayloads()),
              Boolean.toString(fi.omitsNorms()),
              fi.getDocValuesType().toString(),
              fi.docValuesSkipIndexType().toString(),
              Long.toString(fi.getDocValuesGen()),
              atts.toString(),
              Integer.toString(fi.getPointDimensionCount()),
              Integer.toString(fi.getPointIndexDimensionCount()),
              Integer.toString(fi.getPointNumBytes()),
              Integer.toString(fi.getVectorDimension()),
              fi.getVectorEncoding().name(),
              fi.getVectorSimilarityFunction().name(),
              Boolean.toString(fi.isSoftDeletesField()),
              Boolean.toString(fi.isParentField())));
    }
    Files.write(path, lines);
  }
}
