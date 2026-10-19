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
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.bloom.BloomFilteringPostingsFormat;
import org.apache.lucene.codecs.bloom.FuzzySet;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.store.ByteBuffersDataOutput;

/**
 * lucene-codecs' {@code BloomFilteringPostingsFormat}: indexes whose {@code id} field is routed
 * to one bloom instance over the default {@code Lucene104PostingsFormat} (group {@code
 * BloomFilter_0}), and whose {@code tag} and {@code body} fields share a second instance over
 * {@code Lucene104PostingsFormat(4, 9)} ({@code BloomFilter_1}). {@code body} carries many more
 * distinct terms than documents, so its filter saturates past 90% and is left out of the
 * {@code .blm}; {@code tag} has few, so its filter is downsized.
 *
 * <pre>
 *   codecs_bloom/
 *     fuzzyset.tsv       FuzzySet's static sizing math and a serialized set, per input
 *     flush/index/       one flushed segment, deletes inside it (kept in the postings)
 *     flush/blm.tsv      per group: the delegate name, then per bloomed field its number,
 *                        hashCount, bloomSize and cardinality
 *     flush/probes.tsv   per field: present and absent terms, FuzzySet.contains and
 *                        TermsEnum.seekExact through the reader, docFreq when found
 *     flush/meta.tsv     maxDoc
 *     merged/...         the same over three flushed segments with deletes, force-merged
 * </pre>
 *
 * <p>Run with {@code Math.log}/{@code Math.pow} intrinsics disabled (gen-fixtures.sh), so {@code
 * createOptimalSet} computes with StrictMath's fdlibm, as the port does.
 */
public class GenCodecsBloom {
  static final HexFormat HEX = HexFormat.of();

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("codecs_bloom");
    Files.createDirectories(root);
    fuzzySetTable(root.resolve("fuzzyset.tsv"));
    run(root.resolve("flush"), false);
    run(root.resolve("merged"), true);
  }

  /** The static sizing functions and one serialized set per row. */
  static void fuzzySetTable(Path out) throws IOException {
    List<String> lines = new ArrayList<>();
    int[] bits = {0, 1, 63, 64, 100, 127, 128, 1000, 65535, 65536, 1 << 20, 1 << 30, Integer.MAX_VALUE};
    for (int b : bits) {
      lines.add("nearest\t" + b + "\t" + FuzzySet.getNearestSetSize(b));
    }
    int[] values = {1, 10, 100, 1000, 3000, 100000, 10000000};
    float[] sats = {0.1f, 0.5f, 0.9f};
    for (int v : values) {
      for (float s : sats) {
        lines.add(
            "nearest_sat\t" + v + "\t" + Float.floatToIntBits(s) + "\t"
                + FuzzySet.getNearestSetSize(v, s));
      }
    }
    int[][] est = {{63, 0}, {63, 10}, {1023, 500}, {1023, 1022}, {1 << 20, 12345}};
    for (int[] e : est) {
      for (int k = 1; k <= 3; k++) {
        lines.add(
            "estimate\t" + e[0] + "\t" + e[1] + "\t" + k + "\t"
                + FuzzySet.getEstimatedNumberUniqueValuesAllowingForCollisions(e[0], e[1], k));
      }
    }
    // createOptimalSet + addValue + downsize + serialize, over random terms.
    Random r = new Random(17);
    int[] sizes = {1, 7, 100, 1500, 3000, 20000};
    float[] fpps = {0.1023f, 0.01f, 0.5f};
    for (int n : sizes) {
      for (float fpp : fpps) {
        FuzzySet set = FuzzySet.createOptimalSet(n, fpp);
        int adds = n == 1 ? 1 : n / (1 + r.nextInt(4));
        StringBuilder added = new StringBuilder();
        for (int i = 0; i < adds; i++) {
          BytesRef term = new BytesRef(randomTerm(r));
          set.addValue(term);
          if (i < 3) added.append(i == 0 ? "" : ",").append(HEX.formatHex(term.bytes, term.offset, term.offset + term.length));
        }
        FuzzySet down = set.downsize(set.getTargetMaxSaturation());
        lines.add(
            "optimal\t" + n + "\t" + Float.floatToIntBits(fpp) + "\t" + adds + "\t" + set
                + "\t" + Float.floatToIntBits(set.getSaturation()) + "\t"
                + set.getEstimatedUniqueValues() + "\t" + (down == null ? "null" : down.toString())
                + "\t" + serialize(down == null ? set : down) + "\t" + added);
      }
    }
    // The terms `optimal` rows added are regenerated from the same seed by the
    // test; a set built over 0..n's decimal strings is easier to replay.
    for (int n : new int[] {5, 64, 1000}) {
      FuzzySet set = FuzzySet.createSetBasedOnMaxMemory(n);
      for (int i = 0; i < n; i++) {
        set.addValue(new BytesRef(Integer.toString(i)));
      }
      StringBuilder contains = new StringBuilder();
      for (int i = n; i < n + 200; i++) {
        contains.append(set.contains(new BytesRef(Integer.toString(i))) == FuzzySet.ContainsResult.NO ? '0' : '1');
      }
      lines.add("max_memory\t" + n + "\t" + set + "\t" + serialize(set) + "\t" + contains);
    }
    Files.write(out, lines, StandardCharsets.UTF_8);
  }

  static String serialize(FuzzySet set) throws IOException {
    ByteBuffersDataOutput out = new ByteBuffersDataOutput();
    set.serialize(out);
    return HEX.formatHex(out.toArrayCopy());
  }

  static String randomTerm(Random r) {
    int len = 1 + r.nextInt(12);
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < len; i++) sb.append((char) ('a' + r.nextInt(26)));
    return sb.toString();
  }

  static String word(int i) {
    // Deterministic, varied-length words from a number.
    StringBuilder sb = new StringBuilder();
    int x = i;
    do {
      sb.append((char) ('a' + x % 26));
      x /= 26;
    } while (x > 0);
    return sb.toString();
  }

  static void run(Path out, boolean merged) throws IOException {
    if (Files.exists(out)) {
      try (var walk = Files.walk(out)) {
        walk.sorted((a, b) -> b.compareTo(a)).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
    Path idx = out.resolve("index");
    Random r = new Random(merged ? 11 : 5);
    PostingsFormat idFormat = new BloomFilteringPostingsFormat(new Lucene104PostingsFormat());
    PostingsFormat textFormat =
        new BloomFilteringPostingsFormat(new Lucene104PostingsFormat(4, 9));
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
            switch (field) {
              case "id":
                return idFormat;
              case "tag":
              case "body":
                return textFormat;
              default:
                return super.getPostingsFormatForField(field);
            }
          }
        });
    int numDocs = merged ? 1500 : 3000;
    List<String> ids = new ArrayList<>();
    try (Directory dir = FSDirectory.open(idx); IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = 0; i < numDocs; i++) {
        if (merged && (i == 500 || i == 1000)) w.flush();
        String id = "id" + Integer.toString(r.nextInt(1 << 30), 36) + "_" + i;
        ids.add(id);
        Document d = new Document();
        d.add(new StringField("id", id, Field.Store.NO));
        d.add(new StringField("tag", "t" + r.nextInt(200), Field.Store.NO));
        StringBuilder body = new StringBuilder();
        for (int j = 0; j < 20; j++) {
          body.append(word(r.nextInt(400000))).append(' ');
        }
        d.add(new TextField("body", body.toString(), Field.Store.NO));
        d.add(new StringField("other", "o" + (i % 17), Field.Store.NO));
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
      Files.writeString(out.resolve("meta.tsv"), "max_doc\t" + sci.info.maxDoc() + "\n");

      List<String> blm = new ArrayList<>();
      List<List<FuzzySet>> setsByGroup = new ArrayList<>();
      List<List<Integer>> numbersByGroup = new ArrayList<>();
      for (int g = 0; g < 2; g++) {
        String suffix = "BloomFilter_" + g;
        List<FuzzySet> sets = new ArrayList<>();
        List<Integer> numbers = new ArrayList<>();
        try (ChecksumIndexInput in =
            dir.openChecksumInput(sci.info.name + "_" + suffix + ".blm")) {
          CodecUtil.checkIndexHeader(in, "BloomFilter", 3, 3, sci.info.getId(), suffix);
          String delegate = in.readString();
          int n = in.readInt();
          blm.add("group\t" + suffix + "\t" + delegate + "\t" + n);
          for (int i = 0; i < n; i++) {
            int number = in.readInt();
            FuzzySet set = FuzzySet.deserialize(in);
            sets.add(set);
            numbers.add(number);
            blm.add("field\t" + suffix + "\t" + number + "\t" + set);
          }
          CodecUtil.checkFooter(in);
        }
        setsByGroup.add(sets);
        numbersByGroup.add(numbers);
      }
      Files.write(out.resolve("blm.tsv"), blm, StandardCharsets.UTF_8);

      List<String> probes = new ArrayList<>();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        for (String field : new String[] {"id", "tag", "body"}) {
          int number = leaf.getFieldInfos().fieldInfo(field).number;
          FuzzySet set = null;
          for (int g = 0; g < 2; g++) {
            int at = numbersByGroup.get(g).indexOf(number);
            if (at >= 0) set = setsByGroup.get(g).get(at);
          }
          Terms terms = leaf.terms(field);
          Set<String> present = new HashSet<>();
          List<BytesRef> sample = new ArrayList<>();
          TermsEnum all = terms.iterator();
          int k = 0;
          for (BytesRef t = all.next(); t != null; t = all.next()) {
            present.add(t.utf8ToString());
            if (k++ % 13 == 0) sample.add(BytesRef.deepCopyOf(t));
          }
          probes.add("terms\t" + field + "\t" + terms.size() + "\t" + (set == null ? "-" : set));
          int absent = 0;
          while (absent < 400) {
            String cand;
            switch (field) {
              case "id":
                cand = "id" + Integer.toString(r.nextInt(1 << 30), 36) + "_" + r.nextInt(numDocs);
                break;
              case "tag":
                cand = "t" + r.nextInt(400);
                break;
              default:
                cand = word(r.nextInt(800000));
            }
            if (!present.contains(cand)) {
              sample.add(new BytesRef(cand));
              absent++;
            }
          }
          for (BytesRef t : sample) {
            String contains =
                set == null ? "-" : (set.contains(t) == FuzzySet.ContainsResult.NO ? "NO" : "MAYBE");
            TermsEnum te = terms.iterator();
            boolean found = te.seekExact(t);
            probes.add(
                "probe\t" + field + "\t" + HEX.formatHex(t.bytes, t.offset, t.offset + t.length)
                    + "\t" + contains + "\t" + found + "\t" + (found ? te.docFreq() : -1));
          }
        }
      }
      Files.write(out.resolve("probes.tsv"), probes, StandardCharsets.UTF_8);
    }
  }
}
