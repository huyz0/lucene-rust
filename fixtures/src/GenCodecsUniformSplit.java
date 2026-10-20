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
import java.util.Random;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.uniformsplit.BlockDecoder;
import org.apache.lucene.codecs.uniformsplit.BlockEncoder;
import org.apache.lucene.codecs.uniformsplit.UniformSplitPostingsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CodecReader;
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
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.DataOutput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.automaton.Automaton;
import org.apache.lucene.util.automaton.CompiledAutomaton;
import org.apache.lucene.util.automaton.Operations;
import org.apache.lucene.util.automaton.RegExp;

/**
 * lucene-codecs' {@code org.apache.lucene.codecs.uniformsplit}: indexes whose {@code id} (DOCS)
 * and {@code body} (positions) fields are routed to a default {@code UniformSplitPostingsFormat}
 * (32 +- 3 lines per block) and whose {@code bin} (DOCS, binary and long terms), {@code tag}
 * (DOCS_AND_FREQS) and {@code pay} (offsets and payloads) fields to a {@code
 * UniformSplitPostingsFormat(8, 2, null, null, true)}: two groups, {@code UniformSplit_N}, each a
 * {@code .ustd}/{@code .ustb} over {@code Lucene104} postings.
 *
 * <p>The flushed segment also gets a third group written outside {@code IndexWriter}: its
 * postings, re-encoded through {@code UniformSplitPostingsFormat(16, 3, XOR, XOR, false)}'s own
 * {@code fieldsConsumer} into {@code flush/encoded/} (suffix {@code UniformSplit_Xor}), with a test
 * {@link BlockEncoder} that XORs every byte with 0x5a and appends their sum, so the encoded size
 * differs from the decoded one. {@code PerFieldPostingsFormat} resolves a format by its SPI name,
 * which would lose the encoder, hence the side door. Its contents equal the main index's (checked
 * here), so the main manifests describe it.
 *
 * <pre>
 *   codecs_uniformsplit/
 *     flush/index/             one flushed segment, deletes inside it (kept in the postings)
 *     flush/encoded/           the XOR-encoded group over the same postings
 *     flush/meta.tsv           maxDoc
 *     flush/fields.tsv         per field: number, IndexOptions, hasPayloads, size, sums,
 *                              docCount, codec suffix, min and max term
 *     flush/postings.FIELD.tsv per term: hex, docFreq, totalTermFreq, then every document
 *                              as doc:freq:positions:offsets:payloads
 *     flush/norms.tsv          per field with norms: every document's norm
 *     flush/probes.tsv         per field, one reused enum: seekCeil (status, term, docFreq),
 *                              seekExact on a fresh enum, and the term after the ceiling
 *     flush/states.tsv         per field: termState from one enum, seekExact(term, state) on a
 *                              fresh one -> docFreq, totalTermFreq, then two next()
 *     flush/intersect.tsv      per field, pattern and start term: Terms.intersect's terms
 *     merged/...               the same over three flushed segments, force-merged (no encoded/)
 * </pre>
 */
public class GenCodecsUniformSplit {
  static final HexFormat HEX = HexFormat.of();
  static final String[] FIELDS = {"bin", "body", "id", "pay", "tag"};
  static final String[] SMALL_BLOCK_FIELDS = {"bin", "pay", "tag"};
  static final String ENCODED_SUFFIX = "UniformSplit_Xor";
  static final String[] PATTERNS = {
    "[a-c].*", ".*ab.*", "b[a-z]?", "(ab|cd)e.*", "[^a].{0,2}", "id1[0-9a-f]*_1.*", "t[0-3].?",
    "q.*", "a", ".*ef", "id.*_7", "t1[0-9]*", "id3.*", "[a-z]{2}"
  };
  static final String[] STARTS = {null, "b", "ca", "id5", "t2", "t15", "zz", "a", "id3f"};

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("codecs_uniformsplit");
    Files.createDirectories(root);
    run(root.resolve("flush"), false);
    run(root.resolve("merged"), true);
  }

  /** XORs every byte with 0x5a and appends the (byte) sum of the plain bytes. */
  static final class XorEncoder implements BlockEncoder {
    @Override
    public WritableBytes encode(DataInput in, long length) throws IOException {
      byte[] b = new byte[Math.toIntExact(length) + 1];
      byte sum = 0;
      for (int i = 0; i < length; i++) {
        byte x = in.readByte();
        sum += x;
        b[i] = (byte) (x ^ 0x5a);
      }
      b[b.length - 1] = sum;
      return new WritableBytes() {
        @Override
        public long size() {
          return b.length;
        }

        @Override
        public void writeTo(DataOutput out) throws IOException {
          out.writeBytes(b, 0, b.length);
        }
      };
    }
  }

  /** {@link XorEncoder}'s inverse; refuses a block whose trailing sum does not match. */
  static final class XorDecoder implements BlockDecoder {
    @Override
    public BytesRef decode(DataInput in, long length) throws IOException {
      if (length < 1) throw new IOException("empty xor block");
      byte[] b = new byte[Math.toIntExact(length - 1)];
      byte sum = 0;
      for (int i = 0; i < b.length; i++) {
        b[i] = (byte) (in.readByte() ^ 0x5a);
        sum += b[i];
      }
      if (in.readByte() != sum) throw new IOException("xor block sum mismatch");
      return new BytesRef(b);
    }
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
    PostingsFormat big = new UniformSplitPostingsFormat();
    PostingsFormat small = new UniformSplitPostingsFormat(8, 2, null, null, true);
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
            if (List.of(SMALL_BLOCK_FIELDS).contains(field)) return small;
            if (List.of(FIELDS).contains(field)) return big;
            return super.getPostingsFormatForField(field);
          }
        });
    FieldType freqs = new FieldType(TextField.TYPE_NOT_STORED);
    freqs.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    freqs.freeze();
    FieldType offsets = new FieldType(TextField.TYPE_NOT_STORED);
    offsets.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    offsets.freeze();
    int numDocs = merged ? 1500 : 1600;
    List<String> ids = new ArrayList<>();
    try (Directory dir = FSDirectory.open(idx);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = 0; i < numDocs; i++) {
        if (merged && (i == 500 || i == 1000)) w.flush();
        String id = "id" + Integer.toString(r.nextInt(1 << 30), 16) + "_" + i;
        ids.add(id);
        Document d = new Document();
        d.add(new StringField("id", id, Field.Store.NO));
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
          // Skewed: a few words hold hundreds of positions (lastPosBlockOffset).
          body.append(word(r.nextInt(1 + r.nextInt(3000)))).append(' ');
        }
        // The empty term, in a few documents (a seed line of length zero).
        if (i % 97 == 0) d.add(new StringField("bin", new BytesRef(), Field.Store.NO));
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
      List<String> states = new ArrayList<>();
      List<String> intersect = new ArrayList<>();
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
          Files.write(
              out.resolve("postings." + field + ".tsv"), dumpPostings(terms),
              StandardCharsets.UTF_8);
          probes(terms, field, r, probes);
          states(terms, field, states);
          intersect(terms, field, intersect);
        }
        if (!merged) {
          writeEncoded(out, sci, (CodecReader) leaf);
        }
      }
      Files.write(out.resolve("fields.tsv"), fields, StandardCharsets.UTF_8);
      Files.write(out.resolve("norms.tsv"), norms, StandardCharsets.UTF_8);
      Files.write(out.resolve("probes.tsv"), probes, StandardCharsets.UTF_8);
      Files.write(out.resolve("states.tsv"), states, StandardCharsets.UTF_8);
      Files.write(out.resolve("intersect.tsv"), intersect, StandardCharsets.UTF_8);
    }
  }

  /**
   * Re-writes the segment's postings through an XOR-encoding {@code UniformSplitPostingsFormat}
   * into {@code out/encoded}, then reads them back through its {@code fieldsProducer} and requires
   * every field's terms, postings and intersections to equal the main index's.
   */
  static void writeEncoded(Path out, SegmentCommitInfo sci, CodecReader leaf) throws IOException {
    Path encoded = out.resolve("encoded");
    Files.createDirectories(encoded);
    UniformSplitPostingsFormat format =
        new UniformSplitPostingsFormat(16, 3, new XorEncoder(), new XorDecoder(), false);
    try (Directory dir = FSDirectory.open(encoded)) {
      SegmentWriteState write =
          new SegmentWriteState(
              InfoStream.NO_OUTPUT,
              dir,
              sci.info,
              leaf.getFieldInfos(),
              null,
              IOContext.DEFAULT,
              ENCODED_SUFFIX);
      try (FieldsConsumer consumer = format.fieldsConsumer(write)) {
        consumer.write(leaf.getPostingsReader(), leaf.getNormsReader());
      }
      SegmentReadState read =
          new SegmentReadState(dir, sci.info, leaf.getFieldInfos(), IOContext.DEFAULT, ENCODED_SUFFIX);
      try (FieldsProducer producer = format.fieldsProducer(read)) {
        producer.checkIntegrity();
        for (String field : FIELDS) {
          Terms want = leaf.terms(field);
          Terms got = producer.terms(field);
          if (!dumpPostings(want).equals(dumpPostings(got))
              || want.getSumDocFreq() != got.getSumDocFreq()
              || want.getSumTotalTermFreq() != got.getSumTotalTermFreq()
              || want.getDocCount() != got.getDocCount()) {
            throw new IllegalStateException("encoded group differs in " + field);
          }
          List<String> a = new ArrayList<>();
          List<String> b = new ArrayList<>();
          intersect(want, field, a);
          intersect(got, field, b);
          if (!a.equals(b)) throw new IllegalStateException("encoded intersect differs: " + field);
        }
      }
    }
  }

  static String hex(BytesRef t) {
    return HEX.formatHex(t.bytes, t.offset, t.offset + t.length);
  }

  static void intersect(Terms terms, String field, List<String> intersect) throws IOException {
    for (String pattern : PATTERNS) {
      Automaton a =
          Operations.determinize(
              new RegExp(pattern, RegExp.ALL, 0).toAutomaton(),
              Operations.DEFAULT_DETERMINIZE_WORK_LIMIT);
      CompiledAutomaton compiled = new CompiledAutomaton(a, false, true, false);
      if (compiled.type != CompiledAutomaton.AUTOMATON_TYPE.NORMAL) {
        intersect.add(field + "\t" + pattern + "\t-\t" + compiled.type);
        continue;
      }
      for (String start : STARTS) {
        TermsEnum te = terms.intersect(compiled, start == null ? null : new BytesRef(start));
        StringBuilder sb =
            new StringBuilder(field + "\t" + pattern + "\t" + (start == null ? "-" : start));
        for (BytesRef t = te.next(); t != null; t = te.next()) {
          sb.append('\t').append(hex(t)).append(':').append(te.docFreq()).append(':')
              .append(te.totalTermFreq());
        }
        intersect.add(sb.toString());
      }
    }
  }

  static List<String> dumpPostings(Terms terms) throws IOException {
    List<String> lines = new ArrayList<>();
    boolean positions = terms.hasPositions();
    boolean offsets = terms.hasOffsets();
    boolean payloads = terms.hasPayloads();
    TermsEnum te = terms.iterator();
    PostingsEnum pe = null;
    for (BytesRef t = te.next(); t != null; t = te.next()) {
      StringBuilder sb = new StringBuilder();
      sb.append(hex(t)).append('\t').append(te.docFreq()).append('\t').append(te.totalTermFreq());
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
    return lines;
  }

  static void probes(Terms terms, String field, Random r, List<String> probes) throws IOException {
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
          status == TermsEnum.SeekStatus.END ? "-" : hex(te.term()) + ":" + te.docFreq();
      boolean exact = terms.iterator().seekExact(t);
      // The next term after the ceiling, through the same (reused) enum.
      String next = "-";
      if (status != TermsEnum.SeekStatus.END) {
        BytesRef n = te.next();
        if (n != null) next = hex(n);
      } else {
        // Past END, UniformSplit's next() would restart from the first block; use a fresh enum.
        te = terms.iterator();
      }
      probes.add(field + "\t" + hex(t) + "\t" + status + "\t" + ceil + "\t" + exact + "\t" + next);
    }
  }

  /** seekExact(term, state) on a fresh enum, from a state another enum produced, then next(). */
  static void states(Terms terms, String field, List<String> states) throws IOException {
    TermsEnum all = terms.iterator();
    int k = 0;
    for (BytesRef t = all.next(); t != null; t = all.next()) {
      if (k++ % 11 != 0) continue;
      BytesRef term = BytesRef.deepCopyOf(t);
      TermsEnum a = terms.iterator();
      if (!a.seekExact(term)) throw new IllegalStateException("missed " + term);
      TermState state = a.termState();
      TermsEnum b = terms.iterator();
      b.seekExact(term, state);
      StringBuilder sb = new StringBuilder(field);
      sb.append('\t').append(hex(term)).append('\t').append(b.docFreq()).append('\t')
          .append(b.totalTermFreq());
      for (int i = 0; i < 2; i++) {
        BytesRef n = b.next();
        sb.append('\t').append(n == null ? "-" : hex(n));
        if (n == null) break;
      }
      states.add(sb.toString());
    }
  }
}
