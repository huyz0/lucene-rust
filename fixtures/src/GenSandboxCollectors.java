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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Random;
import java.util.stream.Stream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.sandbox.search.LargeNumHitsTopDocsCollector;
import org.apache.lucene.sandbox.search.LargeNumHitsTopDocsCollectorManager;
import org.apache.lucene.sandbox.search.ProfilerCollector;
import org.apache.lucene.sandbox.search.ProfilerCollectorManager;
import org.apache.lucene.sandbox.search.ProfilerCollectorResult;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * lucene-sandbox's collectors: {@code LargeNumHitsTopDocsCollector} (and its manager) and the
 * collector profilers ({@code ProfilerCollector}, {@code ProfilerCollectorManager}), over a
 * three-segment index with deletions.
 *
 * <pre>
 *   sandbox_collectors/
 *     index/         the index (StandardAnalyzer "body", StringField "id")
 *     large.tsv      query, requested hit count, howMany ("all" for the manager's reduce):
 *                    total hits, relation, then doc:scoreBits per hit
 *     profiler.tsv   query, the wrapped collector: its name, reason, children, and its hits
 *     errors.tsv     the arguments Java refuses, and its message
 * </pre>
 *
 * <p>{@code topDocs(howMany)} past the hit count pops the queue's weakest {@code howMany}
 * (Java's {@code populateResults}), so each one runs on a fresh collector.
 */
public class GenSandboxCollectors {

  static final String[] QUERIES = {"T w1", "T w7", "B w0 w3", "all"};
  static final int[] REQUESTED = {1, 7, 50, 1000};

  static String words(Random r, int n, int vocab) {
    StringBuilder sb = new StringBuilder();
    for (int k = 0; k < n; k++) {
      double x = r.nextDouble();
      int w = (int) (vocab * x * x);
      if (k > 0) sb.append(' ');
      sb.append('w').append(w);
    }
    return sb.toString();
  }

  static Query query(String spec) {
    String[] t = spec.split(" ");
    switch (t[0]) {
      case "T":
        return new TermQuery(new Term("body", t[1]));
      case "B":
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        for (int i = 1; i < t.length; i++) {
          b.add(new TermQuery(new Term("body", t[i])), BooleanClause.Occur.SHOULD);
        }
        return b.build();
      case "all":
        return new MatchAllDocsQuery();
      default:
        throw new AssertionError(spec);
    }
  }

  static String render(TopDocs td) {
    StringBuilder sb = new StringBuilder();
    sb.append(td.totalHits.value()).append('\t').append(td.totalHits.relation()).append('\t');
    for (ScoreDoc sd : td.scoreDocs) {
      // populateResults pops a queue smaller than howMany into null slots.
      if (sd == null) {
        sb.append("null,");
        continue;
      }
      sb.append(sd.doc).append(':').append(Integer.toHexString(Float.floatToRawIntBits(sd.score)))
          .append(',');
    }
    return sb.toString();
  }

  /** One collector (the searcher has a single slice), read with {@code topDocs(howMany)}. */
  static TopDocs topDocs(IndexSearcher searcher, Query q, int requested, int howMany)
      throws IOException {
    return searcher.search(
        q,
        new CollectorManager<LargeNumHitsTopDocsCollector, TopDocs>() {
          @Override
          public LargeNumHitsTopDocsCollector newCollector() {
            return new LargeNumHitsTopDocsCollector(requested);
          }

          @Override
          public TopDocs reduce(Collection<LargeNumHitsTopDocsCollector> cs) {
            if (cs.size() != 1) throw new AssertionError("slices " + cs.size());
            return cs.iterator().next().topDocs(howMany);
          }
        });
  }

  static String profile(ProfilerCollectorResult r) {
    StringBuilder sb = new StringBuilder();
    sb.append(r.getName()).append('|').append(r.getReason()).append('|')
        .append(r.getTime() > 0 ? "timed" : "untimed").append('|')
        .append(r.getProfiledChildren().size());
    return sb.toString();
  }

  static String error(Runnable r) {
    try {
      r.run();
      return "OK";
    } catch (RuntimeException e) {
      return e.getClass().getSimpleName() + ": " + e.getMessage();
    }
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("sandbox_collectors");
    if (Files.exists(root)) {
      try (Stream<Path> walk = Files.walk(root)) {
        for (Path p : walk.sorted((x, y) -> y.compareTo(x)).toList()) Files.delete(p);
      }
    }
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(20261020L);
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      int id = 0;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int size : new int[] {70, 50, 30}) {
          for (int k = 0; k < size; k++) {
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            doc.add(new TextField("body", words(r, 1 + r.nextInt(8), 12), Field.Store.NO));
            w.addDocument(doc);
            id++;
          }
          w.commit();
        }
        for (String del : new String[] {"2", "33", "71", "100", "149"}) {
          w.deleteDocuments(new Term("id", del));
        }
        w.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 3) throw new AssertionError("segments");
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        StringBuilder large = new StringBuilder();
        for (String spec : QUERIES) {
          Query q = query(spec);
          for (int req : REQUESTED) {
            TopDocs all = searcher.search(q, new LargeNumHitsTopDocsCollectorManager(req));
            large.append(spec).append('\t').append(req).append("\tall\t").append(render(all))
                .append('\n');
            for (int howMany : new int[] {0, 3, req, req + 5}) {
              large.append(spec).append('\t').append(req).append('\t').append(howMany)
                  .append('\t').append(render(topDocs(searcher, q, req, howMany))).append('\n');
            }
          }
        }
        Files.writeString(root.resolve("large.tsv"), large);

        StringBuilder prof = new StringBuilder();
        for (String spec : QUERIES) {
          Query q = query(spec);
          ProfilerCollectorResult[] single = new ProfilerCollectorResult[1];
          TopDocs[] inner = new TopDocs[1];
          ProfilerCollectorResult reduced =
              searcher.search(
                  q,
                  new ProfilerCollectorManager("search_top_hits") {
                    @Override
                    protected Collector createCollector() {
                      return new TopScoreDocCollectorManager(5, 1000).newCollector();
                    }

                    @Override
                    public ProfilerCollectorResult reduce(Collection<ProfilerCollector> cs)
                        throws IOException {
                      ProfilerCollector c = cs.iterator().next();
                      single[0] = c.getProfileResult();
                      return super.reduce(cs);
                    }
                  });
          ProfilerCollectorResult largeProfile =
              searcher.search(
                  q,
                  new ProfilerCollectorManager("large") {
                    @Override
                    protected Collector createCollector() {
                      return new LargeNumHitsTopDocsCollector(3);
                    }
                  });
          // The wrapped collector's own results, through a parent with one child.
          LargeNumHitsTopDocsCollector wrapped = new LargeNumHitsTopDocsCollector(4);
          ProfilerCollector child =
              new ProfilerCollector(new LargeNumHitsTopDocsCollector(2), "child", java.util.List.of());
          ProfilerCollector parent =
              new ProfilerCollector(wrapped, "parent", java.util.List.of(child));
          searcher.search(
              q,
              new CollectorManager<ProfilerCollector, Void>() {
                boolean made;

                @Override
                public ProfilerCollector newCollector() {
                  if (made) throw new AssertionError("slices");
                  made = true;
                  return parent;
                }

                @Override
                public Void reduce(Collection<ProfilerCollector> cs) {
                  return null;
                }
              });
          ProfilerCollectorResult tree = parent.getProfileResult();
          inner[0] = wrapped.topDocs();
          prof.append(spec).append('\t').append(profile(single[0])).append('\t')
              .append(profile(reduced)).append('\t').append(profile(largeProfile)).append('\t')
              .append(profile(tree)).append('\t')
              .append(profile(tree.getProfiledChildren().get(0))).append('\t')
              .append(parent.scoreMode()).append('\t').append(render(inner[0])).append('\n');
        }
        Files.writeString(root.resolve("profiler.tsv"), prof);

        StringBuilder err = new StringBuilder();
        err.append("collector 0\t")
            .append(error(() -> new LargeNumHitsTopDocsCollector(0))).append('\n');
        err.append("collector -3\t")
            .append(error(() -> new LargeNumHitsTopDocsCollector(-3))).append('\n');
        err.append("manager 0\t")
            .append(error(() -> new LargeNumHitsTopDocsCollectorManager(0))).append('\n');
        err.append("topDocs -1\t")
            .append(error(() -> new LargeNumHitsTopDocsCollector(5).topDocs(-1))).append('\n');
        Files.writeString(root.resolve("errors.tsv"), err);
      }
    }
  }
}
