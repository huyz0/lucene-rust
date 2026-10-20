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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.bitvectors.HnswBitVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Version;

/**
 * Reverse-direction check for lucene-codecs' {@code HnswBitVectorsFormat}: {@code
 * write_bitvectors_fixture} (Rust) writes two groups ({@code .vec}/{@code .vemf}/{@code
 * .vex}/{@code .vem}), and real Lucene opens each through {@code
 * HnswBitVectorsFormat.fieldsReader} with a hand-built {@link SegmentInfo}/{@link FieldInfos},
 * then:
 *
 * <ul>
 *   <li>runs {@code checkIntegrity} (every file's checksum);
 *   <li>requires every vector, with its ordinal and document, to read back as written;
 *   <li>runs {@code KnnVectorsReader.search} for every recorded query and {@code k} and requires
 *       exactly this port's hits, documents and score bits -- the search walks the Rust-built
 *       graph with Java's bit scorer, so a graph whose neighbours differ from what Java would
 *       traverse to fails it.
 * </ul>
 *
 * <p>What it cannot catch: anything that binds these files into a segment ({@code
 * PerFieldKnnVectorsFormat} attributes, {@code SegmentInfo.files}), since this port's IndexWriter
 * does not route a field to this format; and a Rust graph that differs from Java's but walks to
 * the same hits (the forward test pins the graph bytes of a flush).
 */
public class VerifyCodecsBitVectors {
  public static void main(String[] args) throws Exception {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("manifest.properties"))) {
      int eq = line.indexOf('=');
      if (eq > 0) {
        manifest.put(line.substring(0, eq), line.substring(eq + 1));
      }
    }
    byte[] id = HexFormat.of().parseHex(manifest.get("id_hex"));
    int maxDoc = Integer.parseInt(manifest.get("max_doc"));

    List<FieldInfo> infos = new ArrayList<>();
    Map<String, List<FieldInfo>> bySuffix = new LinkedHashMap<>();
    for (String line : Files.readAllLines(dir.resolve("fields.tsv"))) {
      String[] e = line.split("\t");
      String sim = e[3].replaceAll("([a-z])([A-Z])", "$1_$2").toUpperCase();
      FieldInfo fi =
          new FieldInfo(
              e[0],
              Integer.parseInt(e[1]),
              false,
              false,
              false,
              IndexOptions.NONE,
              DocValuesType.NONE,
              DocValuesSkipIndexType.NONE,
              -1,
              new HashMap<>(),
              0,
              0,
              0,
              Integer.parseInt(e[2]),
              VectorEncoding.BYTE,
              VectorSimilarityFunction.valueOf(sim),
              false,
              false);
      infos.add(fi);
      bySuffix.computeIfAbsent(e[4], k -> new ArrayList<>()).add(fi);
    }
    FieldInfos fis = new FieldInfos(infos.toArray(new FieldInfo[0]));

    int failures = 0;
    int searches = 0;
    try (Directory directory = FSDirectory.open(dir)) {
      SegmentInfo si =
          new SegmentInfo(
              directory,
              Version.LATEST,
              Version.LATEST,
              "_0",
              maxDoc,
              false,
              false,
              new Lucene104Codec(),
              Collections.emptyMap(),
              id,
              new HashMap<>(),
              null);
      for (Map.Entry<String, List<FieldInfo>> group : bySuffix.entrySet()) {
        String suffix = group.getKey();
        SegmentReadState state =
            new SegmentReadState(directory, si, fis, IOContext.DEFAULT, suffix);
        try (KnnVectorsReader reader = new HnswBitVectorsFormat().fieldsReader(state)) {
          reader.checkIntegrity();
          Map<Integer, String> names = new HashMap<>();
          for (FieldInfo fi : group.getValue()) {
            names.put(fi.number, fi.name);
          }
          for (String line : Files.readAllLines(dir.resolve("vectors." + suffix + ".tsv"))) {
            String[] e = line.split("\t");
            ByteVectorValues values = reader.getByteVectorValues(names.get(Integer.parseInt(e[0])));
            int ord = Integer.parseInt(e[1]);
            int doc = Integer.parseInt(e[2]);
            if (values.ordToDoc(ord) != doc
                || !HexFormat.of().formatHex(values.vectorValue(ord)).equals(e[3])) {
              System.out.println(suffix + ": vector " + line + " read back differently");
              failures++;
            }
          }
          for (String line : Files.readAllLines(dir.resolve("search." + suffix + ".tsv"))) {
            String[] e = line.split("\t", -1);
            String field = names.get(Integer.parseInt(e[0]));
            int k = Integer.parseInt(e[1]);
            byte[] query = HexFormat.of().parseHex(e[2]);
            TopKnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE);
            reader.search(field, query, collector, AcceptDocs.fromLiveDocs(null, maxDoc));
            TopDocs top = collector.topDocs();
            StringBuilder got = new StringBuilder();
            for (ScoreDoc sd : top.scoreDocs) {
              got.append('\t').append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
            }
            String expected = line.substring(e[0].length() + e[1].length() + e[2].length() + 2);
            if (!got.toString().equals(expected)) {
              System.out.println(
                  suffix + ": " + field + " k=" + k + " " + e[2] + "\n  got " + got + "\n  want "
                      + expected);
              failures++;
            }
            searches++;
          }
        }
      }
    }
    if (searches == 0) {
      System.out.println("no searches recorded");
      failures++;
    }
    if (failures > 0) {
      System.out.println(failures + " failure(s)");
      System.exit(1);
    }
    System.out.println(
        "HnswBitVectorsFormat groups verified against real Lucene (" + searches + " searches). PASS");
  }
}
