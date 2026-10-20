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
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.PostingsReaderBase;
import org.apache.lucene.codecs.PostingsWriterBase;
import org.apache.lucene.codecs.blockterms.BlockTermsReader;
import org.apache.lucene.codecs.blockterms.BlockTermsWriter;
import org.apache.lucene.codecs.blockterms.FixedGapTermsIndexReader;
import org.apache.lucene.codecs.blockterms.FixedGapTermsIndexWriter;
import org.apache.lucene.codecs.blockterms.TermsIndexReaderBase;
import org.apache.lucene.codecs.blockterms.TermsIndexWriterBase;
import org.apache.lucene.codecs.blockterms.VariableGapTermsIndexReader;
import org.apache.lucene.codecs.blockterms.VariableGapTermsIndexWriter;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsReader;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsWriter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermState;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IOUtils;

/**
 * lucene-codecs' {@code org.apache.lucene.codecs.blockterms}. Lucene 10.5.0 ships no postings
 * format over it (the old {@code LuceneFixedGap}/{@code LuceneVarGap*} live in the test
 * framework), so this generator defines three, each {@code BlockTermsWriter}/{@code
 * BlockTermsReader} over {@code Lucene104} postings, registered through SPI so a {@code
 * DirectoryReader} (and the force-merge) can open them by name:
 *
 * <ul>
 *   <li>{@code BlockTermsFixedGap}: {@code FixedGapTermsIndexWriter(state, 16)} -- {@code id}
 *       (DOCS, with the empty term) and {@code pay} (offsets and payloads);
 *   <li>{@code BlockTermsVarGapN}: {@code VariableGapTermsIndexWriter} with {@code
 *       EveryNTermSelector(20)} -- {@code bin} (DOCS, binary and long terms, the empty term) and
 *       {@code tag} (DOCS_AND_FREQS);
 *   <li>{@code BlockTermsVarGapDF}: {@code EveryNOrDocFreqTermSelector(40, 24)} -- {@code body}
 *       (positions, a few words with high docFreq).
 * </ul>
 *
 * <pre>
 *   codecs_blockterms/
 *     flush/index/             one flushed segment, deletes inside it (kept in the postings)
 *     flush/meta.tsv           maxDoc
 *     flush/fields.tsv         per field: number, IndexOptions, hasPayloads, size, sums,
 *                              docCount, codec suffix, min and max term
 *     flush/postings.FIELD.tsv per term: hex, ord ("-" without ord support), docFreq,
 *                              totalTermFreq, then every document as
 *                              doc:freq:positions:offsets:payloads
 *     flush/norms.tsv          per field with norms: every document's norm
 *     flush/probes.tsv         per field, one reused enum: seekCeil (status, term, docFreq, ord),
 *                              seekExact on a fresh enum, and the term after the ceiling (ord)
 *     flush/ords.tsv           per fixed-gap field, one reused enum: seekExact(ord) -> term,
 *                              docFreq, ord(), then next() and its ord
 *     flush/states.tsv         per field, sampled terms: seekExact(term, state) on a fresh enum
 *                              -> docFreq, totalTermFreq, then two next()s
 *     merged/...               the same over three flushed segments, force-merged
 * </pre>
 */
public class GenCodecsBlockTerms {
  static final HexFormat HEX = HexFormat.of();
  static final String[] FIELDS = {"bin", "body", "id", "pay", "tag"};
  static final int FIXED_INTERVAL = 16;
  static final int EVERY_N = 20;
  static final int DF_THRESH = 40;
  static final int DF_INTERVAL = 24;

  /** {@code BlockTermsWriter} over a {@code FixedGapTermsIndexWriter(state, 16)}. */
  public static final class FixedGap extends PostingsFormat {
    public FixedGap() {
      super("BlockTermsFixedGap");
    }

    @Override
    public FieldsConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
      return consumer(state, new FixedGapTermsIndexWriter(state, FIXED_INTERVAL));
    }

    @Override
    public FieldsProducer fieldsProducer(SegmentReadState state) throws IOException {
      return producer(state, new FixedGapTermsIndexReader(state));
    }
  }

  /** {@code BlockTermsWriter} over a {@code VariableGapTermsIndexWriter}, every 20th term. */
  public static final class VarGapEveryN extends PostingsFormat {
    public VarGapEveryN() {
      super("BlockTermsVarGapN");
    }

    @Override
    public FieldsConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
      return consumer(
          state,
          new VariableGapTermsIndexWriter(
              state, new VariableGapTermsIndexWriter.EveryNTermSelector(EVERY_N)));
    }

    @Override
    public FieldsProducer fieldsProducer(SegmentReadState state) throws IOException {
      return producer(state, new VariableGapTermsIndexReader(state));
    }
  }

  /** {@code BlockTermsWriter} over a {@code VariableGapTermsIndexWriter}, docFreq 40 or 24th. */
  public static final class VarGapDocFreq extends PostingsFormat {
    public VarGapDocFreq() {
      super("BlockTermsVarGapDF");
    }

    @Override
    public FieldsConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
      return consumer(
          state,
          new VariableGapTermsIndexWriter(
              state,
              new VariableGapTermsIndexWriter.EveryNOrDocFreqTermSelector(DF_THRESH, DF_INTERVAL)));
    }

    @Override
    public FieldsProducer fieldsProducer(SegmentReadState state) throws IOException {
      return producer(state, new VariableGapTermsIndexReader(state));
    }
  }

  static FieldsConsumer consumer(SegmentWriteState state, TermsIndexWriterBase index)
      throws IOException {
    PostingsWriterBase docs = null;
    boolean success = false;
    try {
      docs = new Lucene104PostingsWriter(state);
      FieldsConsumer ret = new BlockTermsWriter(index, state, docs);
      success = true;
      return ret;
    } finally {
      if (!success) {
        IOUtils.closeWhileHandlingException(docs, index);
      }
    }
  }

  static FieldsProducer producer(SegmentReadState state, TermsIndexReaderBase index)
      throws IOException {
    PostingsReaderBase postings = null;
    boolean success = false;
    try {
      postings = new Lucene104PostingsReader(state);
      FieldsProducer ret = new BlockTermsReader(index, postings, state);
      success = true;
      return ret;
    } finally {
      if (!success) {
        IOUtils.closeWhileHandlingException(postings, index);
      }
    }
  }

  /** Registers the three formats by name, as a META-INF/services file would. */
  static void registerFormats() throws IOException {
    Path spi = Files.createTempDirectory("blockterms-spi");
    Path services = spi.resolve("META-INF/services/org.apache.lucene.codecs.PostingsFormat");
    Files.createDirectories(services.getParent());
    Files.writeString(
        services,
        "GenCodecsBlockTerms$FixedGap\nGenCodecsBlockTerms$VarGapEveryN\n"
            + "GenCodecsBlockTerms$VarGapDocFreq\n");
    PostingsFormat.reloadPostingsFormats(
        new URLClassLoader(
            new URL[] {spi.toUri().toURL()}, GenCodecsBlockTerms.class.getClassLoader()));
  }

  public static void main(String[] args) throws IOException {
    registerFormats();
    Path root = Path.of(args[0]).resolve("codecs_blockterms");
    Files.createDirectories(root);
    run(root.resolve("flush"), false);
    run(root.resolve("merged"), true);
  }

  static String word(int i) {
    StringBuilder sb = new StringBuilder();
    int x = i;
    do {
      sb.append((char) ('a' + x % 26));
      x /= 26;
    } while (x > 0);
    return sb.toString();
  }

  /** A binary term: a shared prefix from a small set, then random bytes, sometimes long. */
  static BytesRef binary(Random r) {
    int prefix = r.nextInt(6);
    int len = r.nextInt(10) == 0 ? 100 + r.nextInt(250) : 1 + r.nextInt(12);
    byte[] b = new byte[2 + len];
    b[0] = (byte) (prefix == 0 ? 0x00 : prefix == 5 ? 0xff : 0x80 + prefix);
    b[1] = (byte) r.nextInt(3);
    for (int i = 2; i < b.length; i++) {
      b[i] = (byte) (i < 40 && r.nextBoolean() ? 0x41 : r.nextInt(256));
    }
    return new BytesRef(b);
  }

  /** A token stream with offsets and (some) payloads. */
  static final class PayStream extends TokenStream {
    final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    final OffsetAttribute offset = addAttribute(OffsetAttribute.class);
    final PayloadAttribute payload = addAttribute(PayloadAttribute.class);
    final String[] words;
    int upto;
    int at;

    PayStream(String[] words) {
      this.words = words;
    }

    @Override
    public boolean incrementToken() {
      if (upto == words.length) return false;
      clearAttributes();
      String w = words[upto++];
      term.setEmpty().append(w);
      offset.setOffset(at, at + w.length());
      at += w.length() + 1 + (upto % 3);
      if (upto % 4 != 0) {
        byte[] p = new byte[upto % 5];
        for (int i = 0; i < p.length; i++) p[i] = (byte) (w.charAt(0) * 7 + i);
        payload.setPayload(new BytesRef(p));
      }
      return true;
    }

    @Override
    public void reset() throws IOException {
      super.reset();
      upto = 0;
      at = 0;
    }
  }

  static void run(Path out, boolean merged) throws IOException {
    if (Files.exists(out)) {
      try (var walk = Files.walk(out)) {
        walk.sorted((a, b) -> b.compareTo(a)).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
    Path idx = out.resolve("index");
    Random r = new Random(merged ? 53 : 51);
    Map<String, PostingsFormat> formats =
        Map.of(
            "id", PostingsFormat.forName("BlockTermsFixedGap"),
            "pay", PostingsFormat.forName("BlockTermsFixedGap"),
            "bin", PostingsFormat.forName("BlockTermsVarGapN"),
            "tag", PostingsFormat.forName("BlockTermsVarGapN"),
            "body", PostingsFormat.forName("BlockTermsVarGapDF"));
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setUseCompoundFile(false);
    TieredMergePolicy tmp = new TieredMergePolicy();
    tmp.setSegmentsPerTier(50);
    tmp.setNoCFSRatio(0.0);
    cfg.setMergePolicy(merged ? tmp : NoMergePolicy.INSTANCE);
    cfg.setMaxBufferedDocs(100000);
    cfg.setRAMBufferSizeMB(256);
    cfg.setCodec(
        new Lucene104Codec() {
          @Override
          public PostingsFormat getPostingsFormatForField(String field) {
            PostingsFormat f = formats.get(field);
            return f != null ? f : super.getPostingsFormatForField(field);
          }
        });
    FieldType freqs = new FieldType(TextField.TYPE_NOT_STORED);
    freqs.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    freqs.freeze();
    FieldType offsets = new FieldType(TextField.TYPE_NOT_STORED);
    offsets.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    offsets.freeze();
    int numDocs = merged ? 1500 : 1700;
    List<String> ids = new ArrayList<>();
    try (Directory dir = FSDirectory.open(idx);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = 0; i < numDocs; i++) {
        if (merged && (i == 500 || i == 1000)) w.flush();
        String id = "id" + Integer.toString(r.nextInt(1 << 30), 16) + "_" + i;
        ids.add(id);
        Document d = new Document();
        d.add(new StringField("id", id, Field.Store.NO));
        if (i % 401 == 7) {
          // The empty term, first in its field.
          d.add(new StringField("id", "", Field.Store.NO));
          d.add(new StringField("bin", new BytesRef(), Field.Store.NO));
        }
        for (int j = 0, n = 1 + r.nextInt(2); j < n; j++) {
          d.add(new StringField("bin", binary(r), Field.Store.NO));
        }
        StringBuilder tags = new StringBuilder();
        for (int j = 0, n = 1 + r.nextInt(4); j < n; j++) {
          tags.append('t').append(r.nextInt(300)).append(' ');
        }
        d.add(new Field("tag", tags.toString(), freqs));
        StringBuilder body = new StringBuilder();
        for (int j = 0; j < 10; j++) {
          // Skewed: a few words hold hundreds of positions (and docFreq >= 40).
          body.append(word(r.nextInt(1 + r.nextInt(3000)))).append(' ');
        }
        d.add(new TextField("body", body.toString(), Field.Store.NO));
        String[] words = new String[4];
        for (int j = 0; j < words.length; j++) words[j] = word(r.nextInt(1 + r.nextInt(600)));
        d.add(new Field("pay", new PayStream(words), offsets));
        w.addDocument(d);
        if (i > 0 && r.nextInt(25) == 0) {
          w.deleteDocuments(new Term("id", ids.get(r.nextInt(ids.size()))));
        }
      }
      if (merged) {
        w.forceMerge(1);
      }
      w.commit();
    }

    try (Directory dir = FSDirectory.open(idx)) {
      SegmentInfos infos = SegmentInfos.readLatestCommit(dir);
      if (infos.size() != 1) throw new IllegalStateException("expected one segment");
      SegmentCommitInfo sci = infos.info(0);
      int maxDoc = sci.info.maxDoc();
      Files.writeString(out.resolve("meta.tsv"), "max_doc\t" + maxDoc + "\n");
      List<String> fields = new ArrayList<>();
      List<String> norms = new ArrayList<>();
      List<String> probes = new ArrayList<>();
      List<String> ords = new ArrayList<>();
      List<String> states = new ArrayList<>();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        for (String field : FIELDS) {
          FieldInfo fi = leaf.getFieldInfos().fieldInfo(field);
          Terms terms = leaf.terms(field);
          fields.add(
              field + "\t" + fi.number + "\t" + fi.getIndexOptions() + "\t" + fi.hasPayloads()
                  + "\t" + terms.size() + "\t" + terms.getSumTotalTermFreq() + "\t"
                  + terms.getSumDocFreq() + "\t" + terms.getDocCount() + "\t"
                  + fi.getAttribute("PerFieldPostingsFormat.format") + "_"
                  + fi.getAttribute("PerFieldPostingsFormat.suffix") + "\t"
                  + hex(terms.getMin()) + "\t" + hex(terms.getMax()));
          NumericDocValues nv = leaf.getNormValues(field);
          if (nv != null) {
            long[] values = new long[maxDoc];
            for (int doc = nv.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = nv.nextDoc()) {
              values[doc] = nv.longValue();
            }
            StringBuilder sb = new StringBuilder(field);
            for (long v : values) sb.append('\t').append(v);
            norms.add(sb.toString());
          }
          boolean doOrd = supportsOrd(terms);
          dumpPostings(terms, doOrd, out.resolve("postings." + field + ".tsv"));
          probes(terms, field, doOrd, r, probes);
          if (doOrd) {
            ords(terms, field, r, ords);
          }
          states(terms, field, states);
        }
      }
      Files.write(out.resolve("fields.tsv"), fields, StandardCharsets.UTF_8);
      Files.write(out.resolve("norms.tsv"), norms, StandardCharsets.UTF_8);
      Files.write(out.resolve("probes.tsv"), probes, StandardCharsets.UTF_8);
      Files.write(out.resolve("ords.tsv"), ords, StandardCharsets.UTF_8);
      Files.write(out.resolve("states.tsv"), states, StandardCharsets.UTF_8);
    }
  }

  static String hex(BytesRef t) {
    return HEX.formatHex(t.bytes, t.offset, t.offset + t.length);
  }

  /** {@code TermsIndexReaderBase.supportsOrd()}, as the enum exposes it. */
  static boolean supportsOrd(Terms terms) throws IOException {
    TermsEnum te = terms.iterator();
    te.next();
    try {
      te.ord();
      return true;
    } catch (UnsupportedOperationException e) {
      return false;
    }
  }

  static void dumpPostings(Terms terms, boolean doOrd, Path out) throws IOException {
    List<String> lines = new ArrayList<>();
    boolean positions = terms.hasPositions();
    boolean offsets = terms.hasOffsets();
    boolean payloads = terms.hasPayloads();
    TermsEnum te = terms.iterator();
    PostingsEnum pe = null;
    for (BytesRef t = te.next(); t != null; t = te.next()) {
      StringBuilder sb = new StringBuilder();
      sb.append(hex(t)).append('\t').append(doOrd ? Long.toString(te.ord()) : "-").append('\t')
          .append(te.docFreq()).append('\t').append(te.totalTermFreq());
      pe = te.postings(pe, PostingsEnum.ALL);
      for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
        int freq = pe.freq();
        sb.append('\t').append(doc).append(':').append(freq);
        if (positions) {
          StringBuilder pos = new StringBuilder();
          StringBuilder offs = new StringBuilder();
          StringBuilder pays = new StringBuilder();
          for (int k = 0; k < freq; k++) {
            int p = pe.nextPosition();
            pos.append(k == 0 ? "" : ".").append(p);
            if (offsets) {
              offs.append(k == 0 ? "" : ".").append(pe.startOffset()).append('-')
                  .append(pe.endOffset());
            }
            if (payloads) {
              BytesRef pl = pe.getPayload();
              pays.append(k == 0 ? "" : ".")
                  .append(pl == null || pl.length == 0 ? "-" : hex(pl));
            }
          }
          sb.append(':').append(pos).append(':').append(offs).append(':').append(pays);
        }
      }
      lines.add(sb.toString());
    }
    Files.write(out, lines, StandardCharsets.UTF_8);
  }

  static void probes(Terms terms, String field, boolean doOrd, Random r, List<String> probes)
      throws IOException {
    List<BytesRef> targets = new ArrayList<>();
    TermsEnum all = terms.iterator();
    int k = 0;
    for (BytesRef t = all.next(); t != null; t = all.next()) {
      if (k++ % 7 == 0) {
        targets.add(BytesRef.deepCopyOf(t));
        // A target just past the term, and one sharing all but its last byte.
        byte[] longer = new byte[t.length + 1];
        System.arraycopy(t.bytes, t.offset, longer, 0, t.length);
        targets.add(new BytesRef(longer));
        if (t.length > 1) targets.add(new BytesRef(t.bytes, t.offset, t.length - 1));
      }
    }
    for (int i = 0; i < 150; i++) {
      BytesRef cand;
      switch (field) {
        case "id":
          cand = new BytesRef("id" + Integer.toString(r.nextInt(1 << 30), 16));
          break;
        case "tag":
          cand = new BytesRef("t" + r.nextInt(400));
          break;
        case "bin":
          cand = binary(r);
          break;
        default:
          cand = new BytesRef(word(r.nextInt(6000)) + (r.nextInt(3) == 0 ? "z" : ""));
      }
      targets.add(cand);
    }
    targets.add(new BytesRef(""));
    targets.add(new BytesRef("zzzzzzzz"));
    targets.add(new BytesRef(new byte[] {(byte) 0xff, (byte) 0xff, (byte) 0xff}));
    // In shuffled order, so the reused enum seeks backwards and forwards.
    java.util.Collections.shuffle(targets, r);
    TermsEnum te = terms.iterator();
    for (BytesRef t : targets) {
      TermsEnum.SeekStatus status = te.seekCeil(t);
      String ceil =
          status == TermsEnum.SeekStatus.END
              ? "-"
              : hex(te.term()) + ":" + te.docFreq() + ":" + (doOrd ? te.ord() : "-");
      boolean exact = terms.iterator().seekExact(t);
      // The next term after the ceiling, through the same (reused) enum.
      String next = "-";
      if (status != TermsEnum.SeekStatus.END) {
        BytesRef n = te.next();
        if (n != null) next = hex(n) + ":" + (doOrd ? te.ord() : "-");
      } else {
        // Past END the enum's input is past the field's last block; start a fresh one.
        te = terms.iterator();
      }
      probes.add(field + "\t" + hex(t) + "\t" + status + "\t" + ceil + "\t" + exact + "\t" + next);
    }
  }

  static void ords(Terms terms, String field, Random r, List<String> ords) throws IOException {
    long size = terms.size();
    List<Long> targets = new ArrayList<>();
    targets.add(0L);
    targets.add(size - 1);
    for (int i = 0; i < 200; i++) targets.add((long) r.nextInt((int) size));
    for (long o = 0; o < size; o += 37) targets.add(o);
    java.util.Collections.shuffle(targets, r);
    TermsEnum te = terms.iterator();
    for (long ord : targets) {
      te.seekExact(ord);
      String line =
          field + "\t" + ord + "\t" + hex(te.term()) + "\t" + te.docFreq() + "\t" + te.ord();
      BytesRef n = te.next();
      line += "\t" + (n == null ? "-" : hex(n) + ":" + te.ord());
      if (n == null) {
        te = terms.iterator();
      }
      ords.add(line);
    }
  }

  /** seekExact(term, state) on a fresh enum (the deferred seek), then next() twice. */
  static void states(Terms terms, String field, List<String> states) throws IOException {
    TermsEnum all = terms.iterator();
    int k = 0;
    for (BytesRef t = all.next(); t != null; t = all.next()) {
      if (k++ % 11 != 3) continue;
      BytesRef term = BytesRef.deepCopyOf(t);
      TermState state = all.termState();
      TermsEnum te = terms.iterator();
      te.seekExact(term, state);
      StringBuilder sb =
          new StringBuilder(field + "\t" + hex(term) + "\t" + te.docFreq() + "\t"
              + te.totalTermFreq());
      for (int i = 0; i < 2; i++) {
        BytesRef n = te.next();
        sb.append('\t').append(n == null ? "-" : hex(n));
        if (n == null) break;
      }
      states.add(sb.toString());
    }
  }
}
