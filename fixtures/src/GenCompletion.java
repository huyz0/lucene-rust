import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.core.KeywordAnalyzer;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.search.suggest.document.Completion104PostingsFormat;
import java.io.IOException;
import org.apache.lucene.search.suggest.document.PrefixCompletionQuery;
import org.apache.lucene.search.suggest.document.RegexCompletionQuery;
import org.apache.lucene.search.suggest.document.SuggestField;
import org.apache.lucene.search.suggest.document.SuggestIndexSearcher;
import org.apache.lucene.search.suggest.document.TopSuggestDocs;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.3d: {@code suggest.document}'s completion postings format. One segment, one suggest field
 * (KeywordAnalyzer, so a term is its surface form); {@code docs.tsv} is the input ({@code doc TAB
 * surface TAB weight}), {@code id.hex} the segment id, {@code _0_Completion104_0.cmp/.lkp} the
 * bytes, {@code lookups.tsv} the answers of {@code PrefixCompletionQuery}.
 */
public class GenCompletion {
  record Doc(String surface, int weight) {}

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("completion");
    Files.createDirectories(out);
    Path idx = Files.createTempDirectory("gencompletion");
    Random r = new Random(2026);
    String[] words = {"cat", "car", "card", "care", "cart", "dog", "door", "dot", "e", "ea", "eat", "zebra"};
    List<Doc> docs = new ArrayList<>();
    for (int i = 0; i < 60; i++) docs.add(new Doc(words[r.nextInt(words.length)], r.nextInt(100)));
    // one surface form on many documents: more than three paths, so a second dedup byte
    for (int i = 0; i < 30; i++) docs.add(new Doc("many", 10 + r.nextInt(5)));
    // One surface form whose dedup order (the builder's heap order) is not doc order: after the
    // collector rejects a hit, a later equal-score hit with a smaller doc would still fit, so
    // "tie" with 4 hits shows whether the lookup stopped (CollectionTerminatedException).
    for (int w : new int[] {1, 9, 1, 1, 1, 5, 1, 1}) docs.add(new Doc("tie", w));
    docs.add(new Doc("heavy", Integer.MAX_VALUE));
    docs.add(new Doc("light", 0));
    docs.add(new Doc("café", 7));
    docs.add(new Doc("日本", 8));

    StringBuilder dt = new StringBuilder();
    for (int i = 0; i < docs.size(); i++)
      dt.append(i).append('\t').append(docs.get(i).surface()).append('\t').append(docs.get(i).weight()).append('\n');
    Files.writeString(out.resolve("docs.tsv"), dt.toString(), StandardCharsets.UTF_8);

    IndexWriterConfig cfg = new IndexWriterConfig(new KeywordAnalyzer());
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(org.apache.lucene.index.NoMergePolicy.INSTANCE);
    cfg.setCodec(new Lucene104Codec() {
      @Override
      public PostingsFormat getPostingsFormatForField(String field) {
        return new Completion104PostingsFormat();
      }
    });
    try (Directory dir = FSDirectory.open(idx); IndexWriter w = new IndexWriter(dir, cfg)) {
      for (Doc d : docs) {
        Document doc = new Document();
        doc.add(new SuggestField("suggest", d.surface(), d.weight()));
        w.addDocument(doc);
      }
      w.commit();
    }
    try (Directory dir = FSDirectory.open(idx)) {
      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) throw new IllegalStateException("segments " + sis.size());
      Files.writeString(out.resolve("id.hex"), HexFormat.of().formatHex(sis.info(0).info.getId()));
      for (String f : sis.info(0).files()) {
        if (f.endsWith(".cmp")) Files.copy(idx.resolve(f), out.resolve("suggest.cmp"), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        if (f.endsWith(".lkp")) Files.copy(idx.resolve(f), out.resolve("suggest.lkp"), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        if (f.endsWith(".cmp")) Files.writeString(out.resolve("suffix.txt"), f.substring(f.indexOf('_', 1) + 1, f.lastIndexOf('.')));
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        SuggestIndexSearcher s = new SuggestIndexSearcher(reader);
        StringBuilder rows = new StringBuilder();
        String[] prefixes = {"c", "ca", "car", "cat", "d", "do", "e", "ea", "m", "many", "h", "l", "z", "caf", "日", "x", "t", "tie", ""};
        for (String p : prefixes) {
          if (p.isEmpty()) continue;
          for (int n : new int[] {1, 3, 4, 10, 200}) {
            for (boolean dedup : new boolean[] {false, true}) {
              TopSuggestDocs top = s.suggest(new PrefixCompletionQuery(new KeywordAnalyzer(), new Term("suggest", p)), n, dedup);
              rows.append(p).append('\t').append(n).append('\t').append(dedup ? 1 : 0).append('\t');
              for (int i = 0; i < top.scoreLookupDocs().length; i++) {
                if (i > 0) rows.append('\u001f');
                var h = top.scoreLookupDocs()[i];
                rows.append(h.key).append('\u001e').append(h.doc).append('\u001e').append((long) h.score);
              }
              rows.append('\n');
            }
          }
        }
        Files.writeString(out.resolve("lookups.tsv"), rows.toString(), StandardCharsets.UTF_8);

        // Regular expressions intersect into several prefix paths, so the collector can fill
        // before the search ends (CollectionTerminatedException).
        StringBuilder re = new StringBuilder();
        String[] regexes = {
          "c[a-z]", "[cd][a-z]", "[a-z]", "ca[rt]", "[a-z][a-z]r", "m[a-z]*", "tie|light", "tie|zebra"
        };
        for (String p : regexes) {
          for (int n : new int[] {1, 2, 3, 4, 10, 200}) {
            for (boolean dedup : new boolean[] {false, true}) {
              TopSuggestDocs top =
                  s.suggest(new RegexCompletionQuery(new Term("suggest", p)), n, dedup);
              re.append(p).append('\t').append(n).append('\t').append(dedup ? 1 : 0).append('\t');
              for (int i = 0; i < top.scoreLookupDocs().length; i++) {
                if (i > 0) re.append('\u001f');
                var h = top.scoreLookupDocs()[i];
                re.append(h.key).append('\u001e').append(h.doc).append('\u001e').append((long) h.score);
              }
              re.append('\n');
            }
          }
        }
        Files.writeString(out.resolve("regex.tsv"), re.toString(), StandardCharsets.UTF_8);
      }
    }
    mergedIndex(out, docs);
    hashMapOrders(out);
    System.out.println("GenCompletion: " + docs.size() + " docs");
  }

  /**
   * The same documents committed after 60 and 90 of them and force-merged into one segment ({@code
   * merged.lkp}/{@code merged.cmp}): {@code CompletionFieldsConsumer.merge} rebuilds the FST from
   * the merged postings, whose doc order is the merge's (largest segment first).
   */
  static void mergedIndex(Path out, List<Doc> docs) throws IOException {
    Path idx = Files.createTempDirectory("gencompletion-merged");
    IndexWriterConfig cfg = new IndexWriterConfig(new KeywordAnalyzer());
    cfg.setUseCompoundFile(false);
    cfg.setCodec(new Lucene104Codec() {
      @Override
      public PostingsFormat getPostingsFormatForField(String field) {
        return new Completion104PostingsFormat();
      }
    });
    try (Directory dir = FSDirectory.open(idx); IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = 0; i < docs.size(); i++) {
        Document doc = new Document();
        doc.add(new SuggestField("suggest", docs.get(i).surface(), docs.get(i).weight()));
        w.addDocument(doc);
        if (i + 1 == 60 || i + 1 == 90) w.commit();
      }
      w.commit();
      w.forceMerge(1);
      w.commit();
    }
    try (Directory dir = FSDirectory.open(idx)) {
      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) throw new IllegalStateException("merged segments " + sis.size());
      for (String f : sis.info(0).files()) {
        if (f.endsWith(".cmp")) Files.copy(idx.resolve(f), out.resolve("merged.cmp"), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        if (f.endsWith(".lkp")) Files.copy(idx.resolve(f), out.resolve("merged.lkp"), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
      }
    }
  }

  /**
   * The iteration order of {@code java.util.HashMap<String, _>} for key lists put in order: what
   * {@code CompletionFieldsConsumer} writes the {@code .cmp} field table in. {@code keys TAB
   * order}, both {@code \u001f}-separated. "Aa" and "BB" share a hash, so their concatenations
   * collide: ten of them fill one bucket past the treeify threshold while the table is below 64,
   * which resizes it instead.
   */
  static void hashMapOrders(Path out) throws IOException {
    List<List<String>> cases = new ArrayList<>();
    List<String> fields = new ArrayList<>();
    for (int i = 0; i < 40; i++) {
      fields.add("f" + i);
      if (i == 3 || i == 12 || i == 13 || i == 25 || i == 39) cases.add(new ArrayList<>(fields));
    }
    cases.add(List.of("suggest", "title", "body", "a", "q", "Q", "zz"));
    String[] ab = {"Aa", "BB"};
    List<String> colliding = new ArrayList<>();
    for (int m = 0; m < 16; m++) {
      StringBuilder b = new StringBuilder();
      for (int bit = 3; bit >= 0; bit--) b.append(ab[(m >> bit) & 1]);
      colliding.add(b.toString());
    }
    for (int k : new int[] {8, 9, 10}) {
      List<String> keys = new ArrayList<>();
      for (int i = 0; i < k; i++) {
        keys.add(colliding.get(i));
        if (i % 3 == 0) keys.add("x" + i);
      }
      cases.add(keys);
    }
    StringBuilder b = new StringBuilder();
    for (List<String> keys : cases) {
      java.util.HashMap<String, Integer> map = new java.util.HashMap<>();
      for (int i = 0; i < keys.size(); i++) map.put(keys.get(i), i);
      b.append(String.join("\u001f", keys)).append('\t');
      b.append(String.join("\u001f", map.keySet())).append('\n');
    }
    Files.writeString(out.resolve("hashmap.tsv"), b.toString(), StandardCharsets.UTF_8);
  }
}
