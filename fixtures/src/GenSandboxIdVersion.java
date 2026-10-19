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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.sandbox.codecs.idversion.IDVersionPostingsFormat;
import org.apache.lucene.sandbox.codecs.idversion.IDVersionSegmentTermsEnum;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;

/**
 * lucene-sandbox's {@code IDVersionPostingsFormat}: indexes whose {@code id} field (default block
 * sizes, 25/48) and {@code id2} field (4/9, so blocks floor and nest deeply) are routed to it,
 * beside a {@code body} text field under the default format. Every document's id carries its
 * version as an 8-byte payload.
 *
 * <pre>
 *   sandbox_idversion/
 *     flush/index/       one flushed segment: updates and deletes inside it, so the writer
 *                        skips deleted documents (SegmentWriteState.liveDocs)
 *     flush/docs.tsv     every document added: docID, field, id, version
 *     flush/ops.tsv      per field: the field's stats, every term in order (term, doc, version),
 *                        then a random walk of seekExact(id, minVersion) / seekCeil / next on
 *                        one enum and what each returned
 *     merged/index/      three flushed segments with updates across them, force-merged
 *     merged/ops.tsv     as above
 * </pre>
 */
public class GenSandboxIdVersion {
  static final String ALPHABET = "abcde0123";

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("sandbox_idversion");
    run(root.resolve("flush"), false);
    run(root.resolve("merged"), true);
  }

  /** A single token with the version as its payload, as the sandbox tests index ids. */
  static final class IdTokenStream extends TokenStream {
    final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
    final PayloadAttribute payloadAtt = addAttribute(PayloadAttribute.class);
    final String id;
    final long version;
    boolean done;

    IdTokenStream(String id, long version) {
      this.id = id;
      this.version = version;
    }

    @Override
    public boolean incrementToken() {
      if (done) return false;
      clearAttributes();
      termAtt.append(id);
      BytesRef payload = new BytesRef(8);
      IDVersionPostingsFormat.longToBytes(version, payload);
      payloadAtt.setPayload(payload);
      done = true;
      return true;
    }

    @Override
    public void reset() throws IOException {
      super.reset();
      done = false;
    }
  }

  static final FieldType ID_TYPE = new FieldType();

  static {
    ID_TYPE.setOmitNorms(true);
    ID_TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    ID_TYPE.setTokenized(true);
    ID_TYPE.freeze();
  }

  static String randomId(Random r) {
    int len = 1 + r.nextInt(r.nextInt(10) == 0 ? 12 : 5);
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < len; i++) sb.append(ALPHABET.charAt(r.nextInt(ALPHABET.length())));
    return sb.toString();
  }

  static long randomVersion(Random r) {
    switch (r.nextInt(10)) {
      case 0:
        return 0;
      case 1:
        return IDVersionPostingsFormat.MAX_VERSION - r.nextInt(3);
      case 2:
        return r.nextInt(100);
      default:
        return r.nextLong() & ((1L << 40) - 1);
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
    Random r = new Random(merged ? 7 : 42);
    PostingsFormat idFormat = new IDVersionPostingsFormat();
    PostingsFormat id2Format = new IDVersionPostingsFormat(4, 9);
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setUseCompoundFile(false);
    // No natural merges; forceMerge(1) still merges under a TieredMergePolicy.
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
            switch (field) {
              case "id":
                return idFormat;
              case "id2":
                return id2Format;
              default:
                return super.getPostingsFormatForField(field);
            }
          }
        });
    List<String> docs = new ArrayList<>();
    try (Directory dir = FSDirectory.open(idx); IndexWriter w = new IndexWriter(dir, cfg)) {
      List<String> used = new ArrayList<>();
      int numDocs = merged ? 1200 : 3000;
      int doc = 0;
      for (int i = 0; i < numDocs; i++) {
        if (merged && (i == 400 || i == 800)) w.flush();
        String id;
        boolean update = !used.isEmpty() && r.nextInt(8) == 0;
        if (update) {
          id = used.get(r.nextInt(used.size()));
        } else {
          id = randomId(r);
          if (used.contains(id)) update = true;
          else used.add(id);
        }
        long version = randomVersion(r);
        Document d = new Document();
        d.add(new Field("id", new IdTokenStream(id, version), ID_TYPE));
        docs.add(doc + "\tid\t" + id + "\t" + version);
        if (r.nextInt(3) == 0) {
          // id2 values are unique per document: the id with its number.
          String id2 = id + "." + i;
          long v2 = randomVersion(r);
          d.add(new Field("id2", new IdTokenStream(id2, v2), ID_TYPE));
          docs.add(doc + "\tid2\t" + id2 + "\t" + v2);
        }
        d.add(new TextField("body", "w" + (i % 7) + " w" + (i % 3), Field.Store.NO));
        d.add(new StoredField("n", i));
        if (update) {
          w.updateDocument(new Term("id", id), d);
        } else {
          w.addDocument(d);
        }
        doc++;
        if (r.nextInt(50) == 0) {
          String gone = used.get(r.nextInt(used.size()));
          w.deleteDocuments(new Term("id", gone));
        }
      }
      if (merged) w.forceMerge(1);
      w.commit();
    }
    if (!merged) {
      Files.write(out.resolve("docs.tsv"), docs, StandardCharsets.UTF_8);
    }
    List<String> lines = new ArrayList<>();
    try (Directory dir = FSDirectory.open(idx); DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 1) throw new IllegalStateException("one segment expected");
      LeafReader leaf = reader.leaves().get(0).reader();
      Bits live = leaf.getLiveDocs();
      StringBuilder del = new StringBuilder("deleted");
      for (int d = 0; d < leaf.maxDoc(); d++) {
        if (live != null && !live.get(d)) del.append(' ').append(d);
      }
      lines.add(del.toString());
      for (String field : new String[] {"id", "id2"}) {
        ops(leaf.terms(field), field, new Random(field.hashCode() + (merged ? 1 : 0)), lines);
      }
    }
    Files.write(out.resolve("ops.tsv"), lines, StandardCharsets.UTF_8);
  }

  static String hit(TermsEnum te) throws IOException {
    PostingsEnum pe = te.postings(null, PostingsEnum.PAYLOADS);
    int doc = pe.nextDoc();
    pe.nextPosition();
    long version = IDVersionPostingsFormat.bytesToLong(pe.getPayload());
    return te.term().utf8ToString() + " " + doc + " " + version;
  }

  static String around(Random r, List<String> terms) {
    String t = terms.get(r.nextInt(terms.size()));
    switch (r.nextInt(5)) {
      case 0:
        return t + ALPHABET.charAt(r.nextInt(ALPHABET.length()));
      case 1:
        return t.substring(0, r.nextInt(t.length()));
      case 2:
        return randomId(r);
      case 3:
        return "z" + t;
      default:
        return t.substring(0, t.length() - 1) + "f";
    }
  }

  static void ops(Terms terms, String field, Random r, List<String> lines) throws IOException {
    lines.add(
        "field\t"
            + field
            + "\t"
            + terms.size()
            + " "
            + terms.getDocCount()
            + " "
            + terms.getSumDocFreq()
            + " "
            + terms.getSumTotalTermFreq()
            + " "
            + terms.getMin().utf8ToString()
            + " "
            + terms.getMax().utf8ToString());
    TermsEnum all = terms.iterator();
    List<String> termList = new ArrayList<>();
    Map<String, Long> versions = new HashMap<>();
    while (all.next() != null) {
      String h = hit(all);
      lines.add("term\t" + h);
      String[] p = h.split(" ");
      termList.add(p[0]);
      versions.put(p[0], Long.parseLong(p[2]));
    }
    IDVersionSegmentTermsEnum te = (IDVersionSegmentTermsEnum) terms.iterator();
    boolean positioned = false;
    for (int i = 0; i < 3000; i++) {
      int op = r.nextInt(positioned ? 10 : 7);
      if (op < 4) {
        String t = r.nextInt(3) == 0 ? around(r, termList) : termList.get(r.nextInt(termList.size()));
        Long v = versions.get(t);
        long min;
        switch (r.nextInt(5)) {
          case 0:
            min = 0;
            break;
          case 1:
            min = v == null ? 5 : v;
            break;
          case 2:
            min = v == null ? 1L << 39 : v + 1;
            break;
          case 3:
            min = v == null || v == 0 ? 0 : v - 1;
            break;
          default:
            min = r.nextLong() & ((1L << 40) - 1);
        }
        boolean found = te.seekExact(new BytesRef(t), min);
        positioned = found;
        lines.add(
            "exact\t"
                + t
                + " "
                + min
                + "\t"
                + (found ? "true " + te.getVersion() + " " + hit(te) : "false"));
      } else if (op < 7) {
        String t = r.nextInt(2) == 0 ? around(r, termList) : termList.get(r.nextInt(termList.size()));
        TermsEnum.SeekStatus s = te.seekCeil(new BytesRef(t));
        positioned = s != TermsEnum.SeekStatus.END;
        lines.add("ceil\t" + t + "\t" + s + (positioned ? " " + hit(te) : ""));
      } else {
        BytesRef next = te.next();
        positioned = next != null;
        lines.add("next\t\t" + (positioned ? hit(te) : "null"));
      }
    }
  }
}
