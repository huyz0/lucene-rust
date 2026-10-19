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
import java.util.List;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.sandbox.index.MergeOnFlushMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * lucene-sandbox's {@code MergeOnFlushMergePolicy} over a {@code TieredMergePolicy} (no compound
 * files, 50 segments a tier, so it never merges on its own): every commit and near-real-time reader of a writer
 * that waits for full-flush merges, and of one that does not.
 *
 * <pre>
 *   sandbox_merge_on_flush/
 *     wait/manifest.txt     maxFullFlushMergeWaitMillis 500 (Lucene's default)
 *     nowait/manifest.txt   maxFullFlushMergeWaitMillis 0: no full-flush merges
 * </pre>
 *
 * <p>Each step flushes one segment per letter -- {@code s} one small document, {@code b} one
 * document with a 16 KiB stored value, above the 0.008 MB (8388-byte) threshold -- then commits
 * ({@code c}) or opens a reader ({@code r}). The manifest lists each commit's and reader's
 * segments ({@code name:maxDoc}), then the final commit's documents in order.
 */
public class GenSandboxMergeOnFlush {
  static final String[] STEPS = {"ssc", "bsc", "sr", "bbc", "sssc", "bsbr", "c", "sc"};

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("sandbox_merge_on_flush");
    run(root.resolve("wait"), 500);
    run(root.resolve("nowait"), 0);
  }

  static String big(int n) {
    StringBuilder sb = new StringBuilder();
    long x = 0x9E3779B97F4A7C15L * (n + 1);
    while (sb.length() < 16384) {
      x ^= x << 13;
      x ^= x >>> 7;
      x ^= x << 17;
      sb.append((char) ('a' + (int) ((x >>> 33) % 26)));
    }
    return sb.toString();
  }

  static void run(Path out, long waitMillis) throws IOException {
    if (Files.exists(out)) {
      try (var walk = Files.walk(out)) {
        walk.sorted((a, b) -> b.compareTo(a)).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
    List<String> lines = new ArrayList<>();
    try (Directory dir = FSDirectory.open(out)) {
      TieredMergePolicy tmp = new TieredMergePolicy();
      // Never a natural merge at these counts: every merge is a full-flush one.
      // A natural merge would also diverge for a reason outside this policy:
      // Java's IndexWriter.flush() runs maybeMerge(FULL_FLUSH), the Rust
      // writer's explicit flush() does not (it merges after a commit).
      tmp.setSegmentsPerTier(50);
      tmp.setMaxMergeAtOnce(50);
      tmp.setNoCFSRatio(0.0);
      MergeOnFlushMergePolicy mp = new MergeOnFlushMergePolicy(tmp);
      mp.setSmallSegmentThresholdMB(0.008);
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(mp);
      cfg.setMergeScheduler(new SerialMergeScheduler());
      cfg.setMaxFullFlushMergeWaitMillis(waitMillis);
      int next = 0;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (String step : STEPS) {
          for (char k : step.substring(0, step.length() - 1).toCharArray()) {
            Document doc = new Document();
            doc.add(new StoredField("id", String.format("d%03d", next)));
            if (k == 'b') doc.add(new StoredField("big", big(next)));
            next++;
            w.addDocument(doc);
            w.flush();
          }
          if (step.endsWith("c")) {
            w.commit();
            SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
            StringBuilder sb = new StringBuilder("commit ").append(sis.getGeneration());
            for (SegmentCommitInfo sci : sis) {
              sb.append(' ').append(sci.info.name).append(':').append(sci.info.maxDoc());
            }
            lines.add(sb.toString());
          } else {
            try (DirectoryReader r = DirectoryReader.open(w)) {
              StringBuilder sb = new StringBuilder("reader");
              for (LeafReaderContext leaf : r.leaves()) {
                SegmentCommitInfo sci = ((SegmentReader) leaf.reader()).getSegmentInfo();
                sb.append(' ').append(sci.info.name).append(':').append(sci.info.maxDoc());
              }
              lines.add(sb.toString());
            }
          }
        }
      }
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        StringBuilder sb = new StringBuilder("docs");
        for (LeafReaderContext leaf : r.leaves()) {
          var stored = leaf.reader().storedFields();
          for (int d = 0; d < leaf.reader().maxDoc(); d++) {
            sb.append(' ').append(stored.document(d).get("id"));
          }
        }
        lines.add(sb.toString());
      }
    }
    Files.write(out.resolve("manifest.txt"), lines, StandardCharsets.UTF_8);
  }
}
