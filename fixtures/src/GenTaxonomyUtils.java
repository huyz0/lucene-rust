import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.facet.FacetField;
import org.apache.lucene.facet.FacetResult;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.LabelAndValue;
import org.apache.lucene.facet.taxonomy.FacetLabel;
import org.apache.lucene.facet.taxonomy.FastTaxonomyFacetCounts;
import org.apache.lucene.facet.taxonomy.OrdinalMappingLeafReader;
import org.apache.lucene.facet.taxonomy.PrintTaxonomyStats;
import org.apache.lucene.facet.taxonomy.SearcherTaxonomyManager;
import org.apache.lucene.facet.taxonomy.SearcherTaxonomyManager.SearcherAndTaxonomy;
import org.apache.lucene.facet.taxonomy.TaxonomyMergeUtils;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyWriter;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyWriter.MemoryOrdinalMap;
import org.apache.lucene.facet.taxonomy.directory.ReindexingEnrichedDirectoryTaxonomyWriter;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;

/**
 * M13 T13.4d2: the taxonomy utilities. Writes {@code taxonomy_utils/}:
 *
 * <ul>
 *   <li>{@code src/index}, {@code src/taxo}: {@link #SRC_DOCS} documents in two segments (40 and
 *       25, deletions in both) whose facets are a flat, a hierarchical and a multi-valued dim in
 *       its own index field ({@code $tags}), plus a sorted-numeric field {@code num} that is not a
 *       facet field.
 *   <li>{@code dest/index}, {@code dest/taxo}: {@link #DEST_DOCS} documents whose categories were
 *       added in another order (and an extra dim), so the source's ordinals all move.
 *   <li>{@code merge.tsv}: {@code TaxonomyMergeUtils.merge} of src into a copy of dest: the ordinal
 *       map, the merged taxonomy, every live document's facet and {@code num} values per leaf, and
 *       {@code FastTaxonomyFacetCounts.getAllDims} over the merged index.
 *   <li>{@code mapping.tsv}: {@code OrdinalMappingLeafReader} over each source leaf, read with
 *       {@code nextDoc} ({@code $facets}), {@code advanceExact} ({@code $tags}) and {@code advance}
 *       ({@code num}, not remapped).
 *   <li>{@code stats.txt}, {@code stats_tree.txt}: {@code PrintTaxonomyStats.printStats} of the
 *       merged taxonomy without and with the tree.
 *   <li>{@code enriched/taxo}, {@code enriched.tsv}: a {@code
 *       ReindexingEnrichedDirectoryTaxonomyWriter} appending a {@code len} numeric doc value, then
 *       {@code reindexWithNewOrdinalData} with a {@code tag} binary doc value, then one more
 *       category; each stage's ordinal documents.
 *   <li>{@code manager.tsv}: {@code SearcherTaxonomyManager} steps, near-real-time and over
 *       directories, including the {@code replaceTaxonomy} refusal.
 * </ul>
 */
public class GenTaxonomyUtils {
  static final int SRC_DOCS = 65;
  static final int SRC_FIRST = 40;
  static final int DEST_DOCS = 30;

  static FacetsConfig config() {
    FacetsConfig c = new FacetsConfig();
    c.setHierarchical("Publish Date", true);
    c.setMultiValued("Tag", true);
    c.setIndexFieldName("Tag", "$tags");
    return c;
  }

  static Document srcDoc(int i) {
    Document d = new Document();
    d.add(new StringField("q", "all", Field.Store.NO));
    if (i % 9 == 4) d.add(new StringField("del", "y", Field.Store.NO));
    d.add(new FacetField("Author", "a" + (i % 6)));
    d.add(new FacetField("Publish Date", "" + (2010 + i % 3), "" + (1 + i % 4)));
    d.add(new FacetField("Tag", "t" + (i % 4)));
    d.add(new FacetField("Tag", "u" + (i % 5)));
    d.add(new SortedNumericDocValuesField("num", i));
    d.add(new SortedNumericDocValuesField("num", (i * 3) % 7));
    return d;
  }

  static Document destDoc(int i) {
    Document d = new Document();
    d.add(new StringField("q", "all", Field.Store.NO));
    d.add(new FacetField("Other", "o" + (i % 3)));
    d.add(new FacetField("Tag", "u" + (4 - i % 5)));
    d.add(new FacetField("Author", "a" + (5 - i % 4)));
    if (i % 2 == 0) d.add(new FacetField("Publish Date", "" + (2012 - i % 2), "9"));
    d.add(new SortedNumericDocValuesField("num", 1000 + i));
    return d;
  }

  static IndexWriterConfig iwc() {
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setRAMBufferSizeMB(256);
    return cfg;
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("taxonomy_utils");
    if (Files.exists(root)) IOUtils.rm(root);
    FacetsConfig config = config();
    Path srcIndex = root.resolve("src/index");
    Path srcTaxo = root.resolve("src/taxo");
    Path destIndex = root.resolve("dest/index");
    Path destTaxo = root.resolve("dest/taxo");
    for (Path p : List.of(srcIndex, srcTaxo, destIndex, destTaxo)) Files.createDirectories(p);

    try (Directory id = FSDirectory.open(srcIndex);
        Directory td = FSDirectory.open(srcTaxo)) {
      IndexWriterConfig cfg = iwc();
      cfg.setUseCompoundFile(false);
      IndexWriter w = new IndexWriter(id, cfg);
      DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(td);
      for (int i = 0; i < SRC_DOCS; i++) {
        w.addDocument(config.build(tw, srcDoc(i)));
        if (i + 1 == SRC_FIRST) w.flush();
      }
      w.flush();
      w.deleteDocuments(new Term("del", "y"));
      w.commit();
      w.close();
      tw.close();
    }
    try (Directory id = FSDirectory.open(destIndex);
        Directory td = FSDirectory.open(destTaxo)) {
      IndexWriterConfig cfg = iwc();
      cfg.setUseCompoundFile(false);
      IndexWriter w = new IndexWriter(id, cfg);
      DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(td);
      for (int i = 0; i < DEST_DOCS; i++) w.addDocument(config.build(tw, destDoc(i)));
      w.commit();
      w.close();
      tw.close();
    }

    StringBuilder merge = new StringBuilder();
    StringBuilder mapping = new StringBuilder();
    int[] ordMap;
    try (Directory sid = FSDirectory.open(srcIndex);
        Directory std = FSDirectory.open(srcTaxo);
        Directory did = FSDirectory.open(destIndex);
        Directory dtd = FSDirectory.open(destTaxo);
        Directory mid = new ByteBuffersDirectory();
        Directory mtd = new ByteBuffersDirectory()) {
      for (String f : did.listAll()) mid.copyFrom(did, f, f, IOContext.DEFAULT);
      for (String f : dtd.listAll()) mtd.copyFrom(dtd, f, f, IOContext.DEFAULT);
      IndexWriter w = new IndexWriter(mid, iwc());
      DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(mtd);
      MemoryOrdinalMap map = new MemoryOrdinalMap();
      TaxonomyMergeUtils.merge(sid, std, map, w, tw, config);
      ordMap = map.getMap();
      w.close();
      tw.close();
      line(merge, "map", ints(ordMap));
      try (DirectoryTaxonomyReader taxo = new DirectoryTaxonomyReader(mtd);
          DirectoryReader reader = DirectoryReader.open(mid)) {
        for (int ord = 0; ord < taxo.getSize(); ord++) {
          line(merge, "taxo", "" + ord, path(taxo.getPath(ord)));
        }
        dumpIndex(merge, reader);
        for (String field : List.of("$facets", "$tags")) {
          FastTaxonomyFacetCounts counts = new FastTaxonomyFacetCounts(field, reader, taxo, config);
          line(merge, "counts", field, list(counts.getAllDims(10)));
        }
        String plain = stats(taxo, false);
        String tree = stats(taxo, true);
        write(root.resolve("stats.txt"), plain);
        write(root.resolve("stats_tree.txt"), tree);
      }
      try (DirectoryReader src = DirectoryReader.open(sid)) {
        for (LeafReaderContext ctx : src.leaves()) {
          LeafReader r = new OrdinalMappingLeafReader(ctx.reader(), ordMap, config);
          String leaf = "" + ctx.ord;
          line(mapping, "leaf", leaf, "" + r.maxDoc());
          SortedNumericDocValues facets = r.getSortedNumericDocValues("$facets");
          for (int doc = facets.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = facets.nextDoc()) {
            line(mapping, "doc", leaf, "" + doc, "$facets", values(facets));
          }
          SortedNumericDocValues tags = r.getSortedNumericDocValues("$tags");
          for (int doc = 0; doc < r.maxDoc(); doc += 3) {
            if (tags.advanceExact(doc)) line(mapping, "doc", leaf, "" + doc, "$tags", values(tags));
          }
          SortedNumericDocValues num = r.getSortedNumericDocValues("num");
          for (int doc = num.advance(5); doc != DocIdSetIterator.NO_MORE_DOCS; doc = num.advance(doc + 7)) {
            line(mapping, "doc", leaf, "" + doc, "num", values(num));
          }
          line(mapping, "absent", leaf, String.valueOf(r.getSortedNumericDocValues("nope")));
        }
      }
    }
    write(root.resolve("merge.tsv"), merge.toString());
    write(root.resolve("mapping.tsv"), mapping.toString());

    StringBuilder enriched = new StringBuilder();
    Path enrichedTaxo = root.resolve("enriched/taxo");
    Files.createDirectories(enrichedTaxo);
    try (Directory d = FSDirectory.open(enrichedTaxo)) {
      ReindexingEnrichedDirectoryTaxonomyWriter tw =
          new ReindexingEnrichedDirectoryTaxonomyWriter(
              d, (cp, doc) -> doc.add(new NumericDocValuesField("len", cp.length)));
      tw.addCategory(new FacetLabel("A", "x"));
      tw.addCategory(new FacetLabel("A", "y"));
      tw.addCategory(new FacetLabel("B", "p", "q"));
      tw.addCategory(new FacetLabel("C"));
      tw.commit();
      line(enriched, "epoch", "before", "" + tw.getTaxonomyEpoch());
      dumpEnriched(enriched, "before", d);
      tw.reindexWithNewOrdinalData(
          (cp, doc) ->
              doc.add(new BinaryDocValuesField("tag", new org.apache.lucene.util.BytesRef(
                  String.join("/", java.util.Arrays.copyOf(cp.components, cp.length)) + "!"))));
      line(enriched, "epoch", "after", "" + tw.getTaxonomyEpoch());
      dumpEnriched(enriched, "after", d);
      line(enriched, "add", "" + tw.addCategory(new FacetLabel("D", "z")));
      line(enriched, "add", "" + tw.addCategory(new FacetLabel("A", "x")));
      tw.commit();
      dumpEnriched(enriched, "final", d);
      tw.close();
    }
    write(root.resolve("enriched.tsv"), enriched.toString());

    StringBuilder manager = new StringBuilder();
    nrtManager(manager, config);
    dirManager(manager, config);
    write(root.resolve("manager.tsv"), manager.toString());
  }

  static void dumpIndex(StringBuilder sb, IndexReader reader) throws Exception {
    for (LeafReaderContext ctx : reader.leaves()) {
      LeafReader r = ctx.reader();
      String leaf = "" + ctx.ord;
      line(sb, "leaf", leaf, "" + r.maxDoc(), "" + r.numDocs());
      Bits live = r.getLiveDocs();
      for (String field : List.of("$facets", "$tags", "num")) {
        SortedNumericDocValues dv = DocValues.getSortedNumeric(r, field);
        for (int doc = dv.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = dv.nextDoc()) {
          if (live != null && !live.get(doc)) continue;
          line(sb, "doc", leaf, "" + doc, field, values(dv));
        }
      }
    }
  }

  static void dumpEnriched(StringBuilder sb, String stage, Directory d) throws Exception {
    try (DirectoryTaxonomyReader taxo = new DirectoryTaxonomyReader(d)) {
      IndexReader ir = taxo.getInternalIndexReader();
      for (LeafReaderContext ctx : ir.leaves()) {
        LeafReader r = ctx.reader();
        NumericDocValues len = r.getNumericDocValues("len");
        BinaryDocValues tag = r.getBinaryDocValues("tag");
        for (int doc = 0; doc < r.maxDoc(); doc++) {
          int ord = ctx.docBase + doc;
          String l = len != null && len.advanceExact(doc) ? "" + len.longValue() : "-";
          String t = tag != null && tag.advanceExact(doc) ? tag.binaryValue().utf8ToString() : "-";
          line(sb, stage, "" + ord, path(taxo.getPath(ord)), l, t);
        }
      }
      line(sb, stage + "_size", "" + taxo.getSize());
    }
  }

  static Object call(SearcherTaxonomyManager m, String name) throws Exception {
    Method method = SearcherTaxonomyManager.class.getDeclaredMethod(name);
    method.setAccessible(true);
    return method.invoke(m);
  }

  static void step(StringBuilder sb, String mode, String name, SearcherTaxonomyManager m, FacetsConfig config)
      throws Exception {
    SearcherAndTaxonomy sat = m.acquire();
    try {
      IndexReader r = sat.searcher().getIndexReader();
      FastTaxonomyFacetCounts counts =
          new FastTaxonomyFacetCounts(FacetsConfig.DEFAULT_INDEX_FIELD_NAME, r, sat.taxonomyReader(), config);
      line(
          sb,
          mode,
          name,
          "" + r.numDocs(),
          "" + sat.taxonomyReader().getSize(),
          "" + call(m, "isSearcherCurrent"),
          "" + call(m, "isTaxonomyCurrent"),
          list(counts.getAllDims(10)));
    } finally {
      m.release(sat);
    }
  }

  static void nrtManager(StringBuilder sb, FacetsConfig config) throws Exception {
    try (Directory id = new ByteBuffersDirectory();
        Directory td = new ByteBuffersDirectory();
        Directory other = new ByteBuffersDirectory()) {
      try (DirectoryTaxonomyWriter ow = new DirectoryTaxonomyWriter(other)) {
        ow.addCategory(new FacetLabel("Zed", "z"));
        ow.commit();
      }
      IndexWriter w = new IndexWriter(id, iwc());
      DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(td);
      for (int i = 0; i < 3; i++) w.addDocument(config.build(tw, srcDoc(i)));
      SearcherTaxonomyManager m = new SearcherTaxonomyManager(w, null, tw);
      step(sb, "nrt", "init", m, config);
      for (int i = 3; i < 8; i++) w.addDocument(config.build(tw, srcDoc(i)));
      step(sb, "nrt", "added", m, config);
      SearcherAndTaxonomy before = m.acquire();
      line(sb, "nrt", "refresh", "" + m.maybeRefresh());
      step(sb, "nrt", "refreshed", m, config);
      SearcherAndTaxonomy after = m.acquire();
      line(sb, "nrt", "changed", "" + (before != after));
      m.release(before);
      m.release(after);
      line(sb, "nrt", "refresh", "" + m.maybeRefresh());
      SearcherAndTaxonomy again = m.acquire();
      line(sb, "nrt", "changed", "" + (again != after));
      m.release(again);
      // A delete alone: the index reader changes, the taxonomy does not.
      w.deleteDocuments(new Term("del", "y"));
      line(sb, "nrt", "refresh", "" + m.maybeRefresh());
      step(sb, "nrt", "deleted", m, config);
      tw.replaceTaxonomy(other);
      w.addDocument(config.build(tw, srcDoc(9)));
      try {
        m.maybeRefresh();
        line(sb, "nrt", "replace", "no error");
      } catch (IllegalStateException e) {
        line(sb, "nrt", "replace", "ISE:" + e.getMessage());
      }
      m.close();
      w.close();
      tw.close();
    }
  }

  static void dirManager(StringBuilder sb, FacetsConfig config) throws Exception {
    try (Directory id = new ByteBuffersDirectory();
        Directory td = new ByteBuffersDirectory()) {
      IndexWriter w = new IndexWriter(id, iwc());
      DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(td);
      for (int i = 0; i < 4; i++) w.addDocument(config.build(tw, srcDoc(i)));
      tw.commit();
      w.commit();
      SearcherTaxonomyManager m = new SearcherTaxonomyManager(id, td, null);
      step(sb, "dir", "init", m, config);
      line(sb, "dir", "generation", "" + call(m, "getSearcherCommitGeneration"));
      for (int i = 4; i < 10; i++) w.addDocument(config.build(tw, srcDoc(i)));
      line(sb, "dir", "refresh", "" + m.maybeRefresh());
      step(sb, "dir", "uncommitted", m, config);
      tw.commit();
      w.commit();
      step(sb, "dir", "committed", m, config);
      line(sb, "dir", "refresh", "" + m.maybeRefresh());
      step(sb, "dir", "refreshed", m, config);
      line(sb, "dir", "generation", "" + call(m, "getSearcherCommitGeneration"));
      m.close();
      w.close();
      tw.close();
    }
  }

  static String stats(DirectoryTaxonomyReader taxo, boolean tree) throws Exception {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (PrintStream out = new PrintStream(bytes, true, StandardCharsets.UTF_8)) {
      PrintTaxonomyStats.printStats(taxo, out, tree);
    }
    return bytes.toString(StandardCharsets.UTF_8).replace("\r\n", "\n");
  }

  static String path(FacetLabel l) {
    return String.join("/", l.components);
  }

  static String values(SortedNumericDocValues dv) throws Exception {
    StringBuilder sb = new StringBuilder();
    for (int k = 0; k < dv.docValueCount(); k++) {
      if (k > 0) sb.append(',');
      sb.append(dv.nextValue());
    }
    return sb.toString();
  }

  static String ints(int[] a) {
    StringBuilder sb = new StringBuilder();
    for (int k = 0; k < a.length; k++) {
      if (k > 0) sb.append(',');
      sb.append(a[k]);
    }
    return sb.toString();
  }

  static String res(FacetResult r) {
    if (r == null) return "null";
    StringBuilder children = new StringBuilder();
    for (LabelAndValue lv : r.labelValues) {
      if (children.length() > 0) children.append(',');
      children.append(lv.label).append('=').append(lv.value.intValue()).append('=').append(lv.count);
    }
    return r.dim + "|" + String.join("/", r.path) + "|" + r.value.intValue() + "|" + r.childCount + "|" + children;
  }

  static String list(List<FacetResult> l) {
    StringBuilder sb = new StringBuilder();
    for (FacetResult r : l) {
      if (sb.length() > 0) sb.append(';');
      sb.append(res(r));
    }
    return sb.toString();
  }

  static void line(StringBuilder sb, String... cols) {
    sb.append(String.join("\t", cols)).append('\n');
  }

  static void write(Path p, String s) throws Exception {
    Files.writeString(p, s, StandardCharsets.UTF_8);
  }
}
