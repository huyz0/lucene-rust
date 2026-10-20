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
import java.util.Random;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.simpletext.SimpleTextCodec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.Impacts;
import org.apache.lucene.index.ImpactsEnum;
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
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * lucene-codecs' {@code org.apache.lucene.codecs.simpletext}, one group per part of the codec.
 *
 * <p>{@code postings/}: {@code SimpleTextPostingsFormat} ({@code .pst}). The format is
 * package-private and not registered as a {@code PostingsFormat} (only {@code SimpleTextCodec}
 * is), so this generator registers a public {@code "SimpleText"} format that delegates to {@code
 * new SimpleTextCodec().postingsFormat()} and routes every field to it from a {@code Lucene104}
 * per-field codec: the {@code .pst} is SimpleText's own bytes, while the {@code .si}/{@code .fnm}
 * /norms stay in the formats this port already reads. Fields: {@code id} (DOCS, unique, with the
 * empty term), {@code cat} (DOCS, a few terms holding most documents: every skip level), {@code
 * bin} (DOCS, binary terms holding newlines, backslashes, 0x00 and 0xff), {@code tag}
 * (DOCS_AND_FREQS), {@code body} (positions, skewed) and {@code pay} (offsets and payloads that
 * need escaping).
 *
 * <pre>
 *   codecs_simpletext/postings/
 *     flush/index/             one flushed segment, deletes inside it (kept in the postings)
 *     flush/meta.tsv           maxDoc
 *     flush/fields.tsv         per field: number, IndexOptions, hasPayloads, size, sums,
 *                              docCount, codec suffix
 *     flush/postings.FIELD.tsv per term: hex, docFreq, totalTermFreq, then every document as
 *                              doc:freq[:positions:offsets:payloads]
 *     flush/norms.tsv          per field with norms: every document's norm
 *     flush/probes.tsv         per field, one reused enum: seekCeil (status, term, docFreq,
 *                              totalTermFreq), seekExact on a fresh enum, the term after the ceiling
 *     flush/advance.tsv        per sampled term: a fresh postings enum driven by nextDoc/advance,
 *                              each step's target and landing document (with its positions)
 *     flush/impacts.tsv        per sampled term: impacts(FREQS) driven by advanceShallow/advance,
 *                              each step's getImpacts() (levels, docIdUpTo, freq-norm pairs)
 *     merged/...               the same over three flushed segments, force-merged
 * </pre>
 */
public class GenCodecsSimpleText {
  static final HexFormat HEX = HexFormat.of();
  static final String[] FIELDS = {"bin", "body", "cat", "id", "pay", "tag"};

  /** SimpleText's postings format under a name {@code PostingsFormat.forName} resolves. */
  public static final class SimpleText extends PostingsFormat {
    private final PostingsFormat in = new SimpleTextCodec().postingsFormat();

    public SimpleText() {
      super("SimpleText");
    }

    @Override
    public FieldsConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
      return in.fieldsConsumer(state);
    }

    @Override
    public FieldsProducer fieldsProducer(SegmentReadState state) throws IOException {
      return in.fieldsProducer(state);
    }
  }

  /** Registers {@link SimpleText} by name, as a META-INF/services file would. */
  static void registerFormats() throws IOException {
    Path spi = Files.createTempDirectory("simpletext-spi");
    Path services = spi.resolve("META-INF/services/org.apache.lucene.codecs.PostingsFormat");
    Files.createDirectories(services.getParent());
    Files.writeString(services, "GenCodecsSimpleText$SimpleText\n");
    PostingsFormat.reloadPostingsFormats(
        new URLClassLoader(
            new URL[] {spi.toUri().toURL()}, GenCodecsSimpleText.class.getClassLoader()));
  }

  public static void main(String[] args) throws IOException {
    registerFormats();
    Path root = Path.of(args[0]).resolve("codecs_simpletext");
    Files.createDirectories(root);
    postings(root.resolve("postings").resolve("flush"), false);
    postings(root.resolve("postings").resolve("merged"), true);
  }

  static void clean(Path out) throws IOException {
    if (Files.exists(out)) {
      try (var walk = Files.walk(out)) {
        walk.sorted((a, b) -> b.compareTo(a)).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
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

  /** A binary term from a pool of 40, holding the bytes SimpleText escapes. */
  static BytesRef binary(int k) {
    Random r = new Random(1000 + k);
    byte[] b = new byte[1 + r.nextInt(6)];
    for (int i = 0; i < b.length; i++) {
      int c = r.nextInt(6);
      b[i] = (byte) (c == 0 ? 0x0a : c == 1 ? 0x5c : c == 2 ? 0x00 : c == 3 ? 0xff : 0x41 + k);
    }
    return new BytesRef(b);
  }

  /** A token stream with offsets and payloads, some holding newlines and backslashes. */
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
        for (int i = 0; i < p.length; i++) {
          int c = (w.charAt(0) + i) % 4;
          p[i] = (byte) (c == 0 ? 0x0a : c == 1 ? 0x5c : w.charAt(0) * 7 + i);
        }
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

  static void postings(Path out, boolean merged) throws IOException {
    clean(out);
    Path idx = out.resolve("index");
    Random r = new Random(merged ? 73 : 71);
    PostingsFormat simpleText = PostingsFormat.forName("SimpleText");
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
            return simpleText;
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
        }
        d.add(new StringField("cat", "c" + r.nextInt(1 + r.nextInt(5)), Field.Store.NO));
        if (r.nextInt(3) != 0) {
          d.add(new StringField("bin", binary(r.nextInt(1 + r.nextInt(40))), Field.Store.NO));
        }
        StringBuilder tags = new StringBuilder();
        for (int j = 0, n = 1 + r.nextInt(5); j < n; j++) {
          tags.append('t').append(r.nextInt(1 + r.nextInt(120))).append(' ');
        }
        d.add(new Field("tag", tags.toString(), freqs));
        StringBuilder body = new StringBuilder();
        for (int j = 0, n = 2 + r.nextInt(12); j < n; j++) {
          body.append(word(r.nextInt(1 + r.nextInt(2000)))).append(' ');
        }
        d.add(new TextField("body", body.toString(), Field.Store.NO));
        String[] words = new String[1 + r.nextInt(5)];
        for (int j = 0; j < words.length; j++) words[j] = word(r.nextInt(1 + r.nextInt(300)));
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
      List<String> advances = new ArrayList<>();
      List<String> impacts = new ArrayList<>();
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
                  + fi.getAttribute("PerFieldPostingsFormat.suffix"));
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
          dumpPostings(terms, out.resolve("postings." + field + ".tsv"));
          probes(terms, field, r, probes);
          advances(terms, field, r, advances);
          impacts(terms, field, r, impacts);
        }
      }
      Files.write(out.resolve("fields.tsv"), fields, StandardCharsets.UTF_8);
      Files.write(out.resolve("norms.tsv"), norms, StandardCharsets.UTF_8);
      Files.write(out.resolve("probes.tsv"), probes, StandardCharsets.UTF_8);
      Files.write(out.resolve("advance.tsv"), advances, StandardCharsets.UTF_8);
      Files.write(out.resolve("impacts.tsv"), impacts, StandardCharsets.UTF_8);
    }
  }

  static String hex(BytesRef t) {
    return HEX.formatHex(t.bytes, t.offset, t.offset + t.length);
  }

  /** The enum's current document as doc:freq[:positions:offsets:payloads]. */
  static String doc(Terms terms, PostingsEnum pe) throws IOException {
    int freq = pe.freq();
    StringBuilder sb = new StringBuilder().append(pe.docID()).append(':').append(freq);
    if (terms.hasPositions()) {
      StringBuilder pos = new StringBuilder();
      StringBuilder offs = new StringBuilder();
      StringBuilder pays = new StringBuilder();
      for (int k = 0; k < freq; k++) {
        int p = pe.nextPosition();
        pos.append(k == 0 ? "" : ".").append(p);
        if (terms.hasOffsets()) {
          offs.append(k == 0 ? "" : ".").append(pe.startOffset()).append('-')
              .append(pe.endOffset());
        }
        if (terms.hasPayloads()) {
          BytesRef pl = pe.getPayload();
          pays.append(k == 0 ? "" : ".").append(pl == null || pl.length == 0 ? "-" : hex(pl));
        }
      }
      sb.append(':').append(pos).append(':').append(offs).append(':').append(pays);
    }
    return sb.toString();
  }

  static void dumpPostings(Terms terms, Path out) throws IOException {
    List<String> lines = new ArrayList<>();
    TermsEnum te = terms.iterator();
    PostingsEnum pe = null;
    for (BytesRef t = te.next(); t != null; t = te.next()) {
      StringBuilder sb = new StringBuilder();
      sb.append(hex(t)).append('\t').append(te.docFreq()).append('\t').append(te.totalTermFreq());
      pe = te.postings(pe, PostingsEnum.ALL);
      for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
        sb.append('\t').append(doc(terms, pe));
      }
      lines.add(sb.toString());
    }
    Files.write(out, lines, StandardCharsets.UTF_8);
  }

  static void probes(Terms terms, String field, Random r, List<String> probes)
      throws IOException {
    List<BytesRef> targets = new ArrayList<>();
    TermsEnum all = terms.iterator();
    int k = 0;
    for (BytesRef t = all.next(); t != null; t = all.next()) {
      if (k++ % 5 == 0) {
        targets.add(BytesRef.deepCopyOf(t));
        byte[] longer = new byte[t.length + 1];
        System.arraycopy(t.bytes, t.offset, longer, 0, t.length);
        targets.add(new BytesRef(longer));
        if (t.length > 1) targets.add(new BytesRef(t.bytes, t.offset, t.length - 1));
      }
    }
    for (int i = 0; i < 60; i++) {
      BytesRef cand;
      switch (field) {
        case "id":
          cand = new BytesRef("id" + Integer.toString(r.nextInt(1 << 30), 16));
          break;
        case "tag":
          cand = new BytesRef("t" + r.nextInt(200));
          break;
        case "cat":
          cand = new BytesRef("c" + r.nextInt(8));
          break;
        case "bin":
          cand = binary(r.nextInt(60));
          break;
        default:
          cand = new BytesRef(word(r.nextInt(4000)) + (r.nextInt(3) == 0 ? "z" : ""));
      }
      targets.add(cand);
    }
    targets.add(new BytesRef(""));
    targets.add(new BytesRef("zzzzzzzz"));
    targets.add(new BytesRef(new byte[] {(byte) 0xff, (byte) 0xff, (byte) 0xff}));
    java.util.Collections.shuffle(targets, r);
    TermsEnum te = terms.iterator();
    for (BytesRef t : targets) {
      TermsEnum.SeekStatus status = te.seekCeil(t);
      String ceil =
          status == TermsEnum.SeekStatus.END
              ? "-"
              : hex(te.term()) + ":" + te.docFreq() + ":" + te.totalTermFreq();
      boolean exact = terms.iterator().seekExact(t);
      String next = "-";
      if (status != TermsEnum.SeekStatus.END) {
        BytesRef n = te.next();
        if (n != null) next = hex(n) + ":" + te.docFreq();
      }
      probes.add(field + "\t" + hex(t) + "\t" + status + "\t" + ceil + "\t" + exact + "\t" + next);
    }
  }

  /** The terms worth driving through the skip list: every one past a block, and a few below. */
  static List<BytesRef> sampled(Terms terms) throws IOException {
    List<BytesRef> out = new ArrayList<>();
    TermsEnum te = terms.iterator();
    int small = 0;
    for (BytesRef t = te.next(); t != null; t = te.next()) {
      if (te.docFreq() > 8 || (te.docFreq() > 1 && small++ % 9 == 0)) {
        out.add(BytesRef.deepCopyOf(t));
      }
    }
    return out;
  }

  /** nextDoc/advance on a fresh enum: {@code n=doc} or {@code aTARGET=doc}, NO_MORE_DOCS as -. */
  static void advances(Terms terms, String field, Random r, List<String> lines)
      throws IOException {
    int flags = terms.hasPositions() ? PostingsEnum.ALL : PostingsEnum.FREQS;
    for (BytesRef t : sampled(terms)) {
      for (int pass = 0; pass < 2; pass++) {
        TermsEnum te = terms.iterator();
        te.seekExact(t);
        PostingsEnum pe = te.postings(null, flags);
        StringBuilder sb = new StringBuilder(field + "\t" + hex(t) + "\t" + flags);
        int doc = -1;
        while (doc != DocIdSetIterator.NO_MORE_DOCS) {
          int choice = r.nextInt(4);
          if (choice == 0) {
            doc = pe.nextDoc();
            sb.append("\tn=");
          } else {
            int gap = pass == 0 ? 1 + r.nextInt(40) : 1 + r.nextInt(400);
            int target = (int) Math.min((long) doc + gap, Integer.MAX_VALUE - 1);
            doc = pe.advance(target);
            sb.append("\ta").append(target).append('=');
          }
          sb.append(doc == DocIdSetIterator.NO_MORE_DOCS ? "-" : doc(terms, pe));
        }
        lines.add(sb.toString());
      }
    }
  }

  static String impactsText(Impacts impacts) {
    StringBuilder sb = new StringBuilder().append(impacts.numLevels());
    for (int level = 0; level < impacts.numLevels(); level++) {
      sb.append('/').append(impacts.getDocIdUpTo(level)).append(':');
      org.apache.lucene.index.FreqAndNormBuffer buf = impacts.getImpacts(level);
      for (int i = 0; i < buf.size; i++) {
        sb.append(i == 0 ? "" : ".").append(buf.freqs[i]).append('-').append(buf.norms[i]);
      }
    }
    return sb.toString();
  }

  /**
   * impacts(FREQS) on a fresh enum: {@code sTARGET=impacts} after advanceShallow, {@code
   * aTARGET=doc:freq=impacts} after advance, NO_MORE_DOCS as -.
   */
  static void impacts(Terms terms, String field, Random r, List<String> lines)
      throws IOException {
    for (BytesRef t : sampled(terms)) {
      TermsEnum te = terms.iterator();
      te.seekExact(t);
      ImpactsEnum ie = te.impacts(PostingsEnum.FREQS);
      StringBuilder sb = new StringBuilder(field + "\t" + hex(t));
      sb.append("\ti=").append(impactsText(ie.getImpacts()));
      int doc = -1;
      while (doc != DocIdSetIterator.NO_MORE_DOCS) {
        int target = (int) Math.min((long) doc + 1 + r.nextInt(120), Integer.MAX_VALUE - 1);
        if (r.nextInt(3) == 0) {
          ie.advanceShallow(target);
          sb.append("\ts").append(target).append('=').append(impactsText(ie.getImpacts()));
        } else {
          doc = ie.advance(target);
          sb.append("\ta").append(target).append('=');
          if (doc == DocIdSetIterator.NO_MORE_DOCS) {
            sb.append('-');
          } else {
            sb.append(doc).append(':').append(ie.freq()).append('=')
                .append(impactsText(ie.getImpacts()));
          }
        }
      }
      lines.add(sb.toString());
    }
  }
}
