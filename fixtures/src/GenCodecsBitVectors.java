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
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.bitvectors.HnswBitVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * lucene-codecs' {@code HnswBitVectorsFormat}: byte vectors read as bit vectors, the graph built
 * and searched with {@code FlatBitVectorsScorer}'s Hamming similarity. Fields {@code b8} (8
 * bytes, every other document) and {@code tiny} (4 bytes, five documents: no graph) go to one
 * default instance ({@code HnswBitVectorsFormat_0}), {@code b32} (32 bytes, every document) to an
 * {@code HnswBitVectorsFormat(8, 40)} instance ({@code HnswBitVectorsFormat_1}).
 *
 * <pre>
 *   codecs_bitvectors/
 *     flush/index/        one flushed segment
 *     flush/meta.tsv      max_doc
 *     flush/vectors.tsv   field, ord, doc, vector hex -- every vector of the segment
 *     flush/search.tsv    field, k, query hex, then doc:scoreBits per hit
 *                         (LeafReader.searchNearestVectors, every document accepted)
 *     merged/...          the same over three flushed segments, force-merged
 * </pre>
 */
public class GenCodecsBitVectors {
  static final HexFormat HEX = HexFormat.of();
  static final String[] FIELDS = {"b8", "tiny", "b32"};

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("codecs_bitvectors");
    Files.createDirectories(root);
    run(root.resolve("flush"), false);
    run(root.resolve("merged"), true);
  }

  static byte[] random(Random r, int n) {
    byte[] b = new byte[n];
    r.nextBytes(b);
    return b;
  }

  static void run(Path out, boolean merged) throws IOException {
    if (Files.exists(out)) {
      try (var walk = Files.walk(out)) {
        walk.sorted((a, b) -> b.compareTo(a)).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
    Path idx = out.resolve("index");
    Random r = new Random(merged ? 23 : 7);
    KnnVectorsFormat first = new HnswBitVectorsFormat();
    KnnVectorsFormat second = new HnswBitVectorsFormat(8, 40);
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
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return field.equals("b32") ? second : first;
          }
        });
    int numDocs = merged ? 1800 : 1500;
    try (Directory dir = FSDirectory.open(idx); IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = 0; i < numDocs; i++) {
        if (merged && (i == 600 || i == 1200)) w.flush();
        Document d = new Document();
        d.add(new StringField("id", Integer.toString(i), Field.Store.NO));
        if (i % 2 == 0) {
          d.add(new KnnByteVectorField("b8", random(r, 8), VectorSimilarityFunction.EUCLIDEAN));
        }
        if (i % 301 == 0) {
          d.add(new KnnByteVectorField("tiny", random(r, 4), VectorSimilarityFunction.DOT_PRODUCT));
        }
        d.add(
            new KnnByteVectorField(
                "b32", random(r, 32), VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT));
        w.addDocument(d);
      }
      if (merged) {
        w.forceMerge(1);
      }
      w.commit();
    }

    try (Directory dir = FSDirectory.open(idx)) {
      SegmentInfos infos = SegmentInfos.readLatestCommit(dir);
      if (infos.size() != 1) throw new IllegalStateException("expected one segment");
      Files.writeString(out.resolve("meta.tsv"), "max_doc\t" + infos.info(0).info.maxDoc() + "\n");
      List<String> vectors = new ArrayList<>();
      List<String> search = new ArrayList<>();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        for (String field : FIELDS) {
          ByteVectorValues values = leaf.getByteVectorValues(field);
          KnnVectorValues.DocIndexIterator it = values.iterator();
          for (int doc = it.nextDoc(); doc != KnnVectorValues.DocIndexIterator.NO_MORE_DOCS;
              doc = it.nextDoc()) {
            int ord = it.index();
            vectors.add(field + "\t" + ord + "\t" + doc + "\t" + HEX.formatHex(values.vectorValue(ord)));
          }
          int dim = values.dimension();
          for (int q = 0; q < 40; q++) {
            // Half random queries, half stored vectors (an exact hit first).
            byte[] query =
                q % 2 == 0 ? random(r, dim) : values.vectorValue(r.nextInt(values.size())).clone();
            for (int k : new int[] {1, 10, 50}) {
              TopDocs top =
                  leaf.searchNearestVectors(
                      field, query, k, AcceptDocs.fromLiveDocs(null, leaf.maxDoc()), Integer.MAX_VALUE);
              StringBuilder sb = new StringBuilder(field);
              sb.append('\t').append(k).append('\t').append(HEX.formatHex(query));
              for (ScoreDoc sd : top.scoreDocs) {
                sb.append('\t').append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
              }
              search.add(sb.toString());
            }
          }
        }
      }
      Files.write(out.resolve("vectors.tsv"), vectors, StandardCharsets.UTF_8);
      Files.write(out.resolve("search.tsv"), search, StandardCharsets.UTF_8);
    }
  }
}
