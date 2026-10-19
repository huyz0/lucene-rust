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
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.memory.DirectPostingsFormat;
import org.apache.lucene.codecs.memory.FSTPostingsFormat;
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
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.automaton.Automaton;
import org.apache.lucene.util.automaton.CompiledAutomaton;
import org.apache.lucene.util.automaton.Operations;
import org.apache.lucene.util.automaton.RegExp;

/**
 * lucene-codecs' {@code org.apache.lucene.codecs.memory} formats: indexes whose {@code id}
 * (DOCS), {@code tag} (DOCS_AND_FREQS), {@code body} (positions) and {@code pay} (offsets and
 * payloads) fields are routed to {@code FSTPostingsFormat} (group {@code FST50_0}), and whose
 * {@code did}, {@code dbody} and {@code dpay} fields are routed to {@code DirectPostingsFormat}
 * (group {@code Direct_0}, the {@code Lucene104} files {@code DirectFields} loads into memory).
 *
 * <pre>
 *   codecs_memory/
 *     flush/index/             one flushed segment, deletes inside it (kept in the postings)
 *     flush/meta.tsv           maxDoc
 *     flush/fields.tsv         per field: number, IndexOptions, hasPayloads, size, sums,
 *                              docCount, Terms.toString
 *     flush/postings.FIELD.tsv per term: hex, docFreq, totalTermFreq, then every document as
 *                              doc:freq:positions:offsets:payloads (all documents, deleted
 *                              ones included)
 *     flush/norms.tsv          per field with norms: every document's norm
 *     flush/probes.tsv         per field: seekCeil (status, term) and seekExact on present
 *                              and absent targets
 *     flush/intersect.tsv      per field and pattern (and start term): the terms
 *                              Terms.intersect enumerates, with docFreq
 *     merged/...               the same over three flushed segments, force-merged
 * </pre>
 */
public class GenCodecsMemory {
  static final HexFormat HEX = HexFormat.of();
  static final String[] FST_FIELDS = {"id", "tag", "body", "pay"};
  static final String[] DIRECT_FIELDS = {"did", "dbody", "dpay"};
  static final String[] PATTERNS = {
    "[a-c].*", ".*ab.*", "b[a-z]?", "(ab|cd)e.*", "[^a].{0,2}", "id1[0-9a-f]*_1.*", "t[0-3].?",
    "q.*", "a"
  };

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("codecs_memory");
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
    Random r = new Random(merged ? 23 : 19);
    PostingsFormat fst = new FSTPostingsFormat();
    PostingsFormat direct = new DirectPostingsFormat();
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
            if (List.of(FST_FIELDS).contains(field)) return fst;
            if (List.of(DIRECT_FIELDS).contains(field)) return direct;
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
        d.add(new StringField("did", id, Field.Store.NO));
        StringBuilder tags = new StringBuilder();
        for (int j = 0, n = 1 + r.nextInt(4); j < n; j++) {
          tags.append('t').append(r.nextInt(6)).append(' ');
        }
        d.add(new Field("tag", tags.toString(), freqs));
        StringBuilder body = new StringBuilder();
        for (int j = 0; j < 10; j++) {
          // Skewed: a few words hold hundreds of positions.
          body.append(word(r.nextInt(1 + r.nextInt(3000)))).append(' ');
        }
        d.add(new TextField("body", body.toString(), Field.Store.NO));
        d.add(new TextField("dbody", body.toString(), Field.Store.NO));
        String[] words = new String[4];
        for (int j = 0; j < words.length; j++) words[j] = word(r.nextInt(1 + r.nextInt(400)));
        d.add(new Field("pay", new PayStream(words), offsets));
        d.add(new Field("dpay", new PayStream(words), offsets));
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
      List<String> intersect = new ArrayList<>();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        List<String> all = new ArrayList<>(List.of(FST_FIELDS));
        all.addAll(List.of(DIRECT_FIELDS));
        for (String field : all) {
          FieldInfo fi = leaf.getFieldInfos().fieldInfo(field);
          Terms terms = leaf.terms(field);
          fields.add(
              field + "\t" + fi.number + "\t" + fi.getIndexOptions() + "\t" + fi.hasPayloads()
                  + "\t" + terms.size() + "\t" + terms.getSumTotalTermFreq() + "\t"
                  + terms.getSumDocFreq() + "\t" + terms.getDocCount() + "\t" + terms);
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
            for (String start : new String[] {null, "b", "ca", "id5", "t2", "zz"}) {
              // DirectIntersectTermsEnum requires the automaton to accept
              // startTerm's prefix (it asserts so, and indexes a dead state
              // otherwise); the FST enum takes any start term.
              if (start != null
                  && List.of(DIRECT_FIELDS).contains(field)
                  && !livePrefix(compiled, start)) {
                continue;
              }
              TermsEnum te =
                  terms.intersect(compiled, start == null ? null : new BytesRef(start));
              StringBuilder sb =
                  new StringBuilder(field + "\t" + pattern + "\t" + (start == null ? "-" : start));
              for (BytesRef t = te.next(); t != null; t = te.next()) {
                sb.append('\t').append(HEX.formatHex(t.bytes, t.offset, t.offset + t.length))
                    .append(':').append(te.docFreq()).append(':').append(te.totalTermFreq());
              }
              intersect.add(sb.toString());
            }
          }
        }
      }
      Files.write(out.resolve("fields.tsv"), fields, StandardCharsets.UTF_8);
      Files.write(out.resolve("norms.tsv"), norms, StandardCharsets.UTF_8);
      Files.write(out.resolve("probes.tsv"), probes, StandardCharsets.UTF_8);
      Files.write(out.resolve("intersect.tsv"), intersect, StandardCharsets.UTF_8);
    }
  }

  /** Whether {@code compiled} has a live state after every byte of {@code start}. */
  static boolean livePrefix(CompiledAutomaton compiled, String start) {
    int state = 0;
    for (byte b : start.getBytes(StandardCharsets.UTF_8)) {
      state = compiled.runAutomaton.step(state, b & 0xFF);
      if (state == -1) return false;
    }
    return true;
  }

  static void dumpPostings(Terms terms, Path out) throws IOException {
    List<String> lines = new ArrayList<>();
    boolean positions = terms.hasPositions();
    boolean offsets = terms.hasOffsets();
    boolean payloads = terms.hasPayloads();
    TermsEnum te = terms.iterator();
    PostingsEnum pe = null;
    for (BytesRef t = te.next(); t != null; t = te.next()) {
      StringBuilder sb = new StringBuilder();
      sb.append(HEX.formatHex(t.bytes, t.offset, t.offset + t.length))
          .append('\t').append(te.docFreq()).append('\t').append(te.totalTermFreq());
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
                  .append(pl == null || pl.length == 0
                      ? "-"
                      : HEX.formatHex(pl.bytes, pl.offset, pl.offset + pl.length));
            }
          }
          sb.append(':').append(pos).append(':').append(offs).append(':').append(pays);
        }
      }
      lines.add(sb.toString());
    }
    Files.write(out, lines, StandardCharsets.UTF_8);
  }

  static void probes(Terms terms, String field, Random r, List<String> probes) throws IOException {
    List<BytesRef> targets = new ArrayList<>();
    TermsEnum all = terms.iterator();
    int k = 0;
    for (BytesRef t = all.next(); t != null; t = all.next()) {
      if (k++ % 7 == 0) targets.add(BytesRef.deepCopyOf(t));
    }
    for (int i = 0; i < 150; i++) {
      String cand;
      switch (field) {
        case "id":
        case "did":
          cand = "id" + Integer.toString(r.nextInt(1 << 30), 16);
          break;
        case "tag":
          cand = "t" + r.nextInt(12);
          break;
        default:
          cand = word(r.nextInt(6000)) + (r.nextInt(3) == 0 ? "z" : "");
      }
      targets.add(new BytesRef(cand));
    }
    targets.add(new BytesRef(""));
    targets.add(new BytesRef("zzzzzzzz"));
    TermsEnum te = terms.iterator();
    for (BytesRef t : targets) {
      TermsEnum.SeekStatus status = te.seekCeil(t);
      String ceil =
          status == TermsEnum.SeekStatus.END
              ? "-"
              : HEX.formatHex(te.term().bytes, te.term().offset, te.term().offset + te.term().length)
                  + ":" + te.docFreq();
      boolean exact = terms.iterator().seekExact(t);
      // The next term after the ceiling, through the same (reused) enum.
      String next = "-";
      if (status != TermsEnum.SeekStatus.END) {
        BytesRef n = te.next();
        if (n != null) next = HEX.formatHex(n.bytes, n.offset, n.offset + n.length);
      }
      probes.add(
          field + "\t" + HEX.formatHex(t.bytes, t.offset, t.offset + t.length) + "\t" + status
              + "\t" + ceil + "\t" + exact + "\t" + next);
    }
  }
}
