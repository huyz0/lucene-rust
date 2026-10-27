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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.simpletext.SimpleTextCodec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.InetAddressPoint;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldCollectorManager;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * Searches over a {@code SimpleTextCodec} index: the routing of every query family through
 * SimpleText segments ({@code lucene-search/src/directory_reader/simpletext_search.rs}).
 *
 * <p>{@code simpletext_search/index}: {@link GenM7Queries}' three-segment, index-sorted schema
 * and documents (same seed), written by {@code SimpleTextCodec} with six deletions -- text,
 * keywords with sorted-set doc values, long points with numeric doc values, the index-sort field,
 * 2-D, 1-D and 16-byte points, float and byte vectors.
 *
 * <pre>
 *   simpletext_search/
 *     index/          the index
 *     searches.tsv    similarity, GenM7Queries' query text, top-20 "doc:scoreBits" -- every
 *                     SCORED query under each of its similarities, every UNSCORED one (doc-values
 *                     rewrites and ranges, the index-sort range, points, IndexOrDocValuesQuery,
 *                     vector thresholds, KNN as a clause; SimpleText's KNN is exhaustive)
 *     spans.tsv       GenSpans' span spec, top-20 "doc:scoreBits" under BM25
 *     sorted.tsv      query text, GenSortedSearch's sort spec, top n, its hits
 *                     ("doc:value:value"), the total and its relation (threshold 1000)
 * </pre>
 */
public final class GenSimpleTextSearch {
  static final int TOP = 20;

  static final String[] SPANS = {
    "t1",
    "near(0,true,t0,t1)",
    "near(2,true,t1,t0,t2)",
    "near(3,false,t2,t1)",
    "or(t3,t5,zz)",
    "first(2,t0)",
    "not(t1,t2,1,1)",
    "containing(near(4,false,t0,t3),t1)",
    "within(t2,near(3,true,t0,t4))",
    "near(1,true,prefix(body,t1),t0)",
    "or(wildcard(body,t?),t20)",
    "near(0,true,t(title,t1),t(title,t2))",
    "range(1,4,t2)",
  };

  static final String[] SORT_QUERIES = {
    "T body t1",
    "B 0 2 0 T body t0 PF body t2 cs",
    "P2R p2 -20 -20 20 20",
    "B 1 0 0 T body t2 F2 NR num 0 80",
  };

  static Sort[] sorts() {
    return new Sort[] {
      new Sort(new SortedNumericSortField("snum", SortField.Type.LONG)),
      new Sort(new SortedNumericSortField("snum", SortField.Type.LONG, true, SortedNumericSelector.Type.MAX)),
      new Sort(withMissing(new SortField("num", SortField.Type.LONG), 7L), SortField.FIELD_DOC),
      new Sort(withMissing(new SortField("num", SortField.Type.LONG, true), -1000L)),
      new Sort(SortField.FIELD_SCORE, new SortedNumericSortField("snum", SortField.Type.LONG)),
      new Sort(SortField.FIELD_DOC),
    };
  }

  static SortField withMissing(SortField f, Object missing) {
    f.setMissingValue(missing);
    return f;
  }

  static String hits(TopDocs td) {
    return GenM7Queries.hits(td);
  }

  static Query parse(String q) {
    GenM7Queries.pos = 0;
    String[] tok = q.split(" ");
    Query query = GenM7Queries.parse(tok);
    if (GenM7Queries.pos != tok.length) throw new AssertionError("trailing tokens in " + q);
    return query;
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]);
    Path out = root.resolve("simpletext_search");
    GenM7Queries.deleteRecursive(out);
    Path index = out.resolve("index");
    Files.createDirectories(index);
    // GenM7Queries.main's documents, field for field and draw for draw.
    Random r = new Random(20261001L);
    try (Directory dir = FSDirectory.open(index)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setCodec(new SimpleTextCodec());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setIndexSort(new Sort(new SortedNumericSortField("snum", SortField.Type.LONG)));
      int id = 0;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int size : new int[] {90, 60, 30}) {
          for (int k = 0; k < size; k++) {
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            doc.add(new TextField("body",
                GenM7Queries.words(r, 1 + r.nextInt(r.nextInt(4) == 0 ? 50 : 12), 40), Field.Store.NO));
            if (r.nextInt(5) != 0) {
              doc.add(new TextField("title", GenM7Queries.words(r, 1 + r.nextInt(5), 12), Field.Store.NO));
            }
            doc.add(new TextField("gram", GenM7Queries.bigrams(r), Field.Store.NO));
            int tags = r.nextInt(4);
            for (int t = 0; t < tags; t++) {
              String tag = String.format("k%02d", r.nextInt(20));
              doc.add(new StringField("tag", tag, Field.Store.NO));
              doc.add(new SortedSetDocValuesField("tag", new BytesRef(tag)));
            }
            if (r.nextInt(6) != 0) {
              long v = r.nextInt(120) - 20;
              doc.add(new LongPoint("num", v));
              doc.add(new NumericDocValuesField("num", v));
            }
            doc.add(new SortedNumericDocValuesField("snum", r.nextInt(100)));
            if (r.nextInt(4) != 0) {
              doc.add(new IntPoint("p2", r.nextInt(101) - 50, r.nextInt(101) - 50));
            }
            if (r.nextInt(3) != 0) {
              doc.add(new IntPoint("i1", r.nextInt(31) - 10));
            }
            if (r.nextInt(3) != 0) {
              doc.add(new InetAddressPoint("ip",
                  GenM7Queries.addr("10.0." + r.nextInt(2) + "." + r.nextInt(100))));
            }
            if (r.nextInt(7) != 0) {
              float[] v = new float[4];
              for (int j = 0; j < 4; j++) v[j] = r.nextFloat() * 2 - 1;
              doc.add(new KnnFloatVectorField("vec", v, VectorSimilarityFunction.EUCLIDEAN));
            }
            if (r.nextInt(5) != 0) {
              byte[] v = new byte[4];
              for (int j = 0; j < 4; j++) v[j] = (byte) (r.nextInt(81) - 40);
              doc.add(new KnnByteVectorField("bvec", v, VectorSimilarityFunction.EUCLIDEAN));
            }
            w.addDocument(doc);
            id++;
          }
          w.commit();
        }
        for (String del : new String[] {"3", "17", "40", "95", "101", "130"}) {
          w.deleteDocuments(new Term("id", del));
        }
        w.commit();
      }

      StringBuilder searches = new StringBuilder();
      StringBuilder spans = new StringBuilder();
      StringBuilder sorted = new StringBuilder();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 3) {
          throw new AssertionError("expected three segments, got " + reader.leaves().size());
        }
        for (Map.Entry<String, Similarity> e : GenM7Queries.sims().entrySet()) {
          IndexSearcher searcher = new IndexSearcher(reader);
          searcher.setQueryCache(null);
          searcher.setSimilarity(e.getValue());
          List<String> qs = new ArrayList<>(List.of(GenM7Queries.SCORED));
          if (e.getKey().equals("bm25")) qs.addAll(List.of(GenM7Queries.UNSCORED));
          for (String q : qs) {
            TopDocs td = searcher.search(parse(q), TOP);
            searches.append(e.getKey()).append('\t').append(q).append('\t').append(hits(td)).append('\n');
          }
        }
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        for (String spec : SPANS) {
          SpanQuery query = GenSpans.query(spec);
          spans.append(spec).append('\t').append(hits(searcher.search(query, TOP))).append('\n');
        }
        for (String q : SORT_QUERIES) {
          for (Sort sort : sorts()) {
            for (int n : new int[] {5, 25}) {
              TopFieldDocs td = searcher.search(parse(q), new TopFieldCollectorManager(sort, n, null, 1000));
              StringBuilder h = new StringBuilder();
              for (ScoreDoc sd : td.scoreDocs) {
                if (h.length() > 0) h.append(',');
                h.append(GenSortedSearch.hit(sort, (FieldDoc) sd));
              }
              sorted.append(q).append('\t').append(GenSortedSearch.spec(sort)).append('\t').append(n)
                  .append('\t').append(h).append('\t').append(td.totalHits.value())
                  .append('\t').append(td.totalHits.relation() == org.apache.lucene.search.TotalHits.Relation.EQUAL_TO
                      ? "eq" : "gte").append('\n');
            }
          }
        }
      }
      Files.writeString(out.resolve("searches.tsv"), searches);
      Files.writeString(out.resolve("spans.tsv"), spans);
      Files.writeString(out.resolve("sorted.tsv"), sorted);
    }
  }
}
