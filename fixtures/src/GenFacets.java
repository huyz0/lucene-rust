import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleDocValuesField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.facet.FacetResult;
import org.apache.lucene.facet.Facets;
import org.apache.lucene.facet.FacetsCollector;
import org.apache.lucene.facet.FacetsCollectorManager;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.LabelAndValue;
import org.apache.lucene.facet.range.DoubleRange;
import org.apache.lucene.facet.range.DoubleRangeFacetCounts;
import org.apache.lucene.facet.range.LongRange;
import org.apache.lucene.facet.range.LongRangeFacetCounts;
import org.apache.lucene.facet.sortedset.DefaultSortedSetDocValuesReaderState;
import org.apache.lucene.facet.sortedset.SortedSetDocValuesFacetCounts;
import org.apache.lucene.facet.sortedset.SortedSetDocValuesFacetField;
import org.apache.lucene.facet.sortedset.SortedSetDocValuesReaderState;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MultiDocValues;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.OrdinalMap;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.LongValues;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Generates the `facets_index` fixture: a real Lucene 10.5.0 index written
 * through `FacetsConfig.build`, deliberately left as **three segments**
 * (`NoMergePolicy` + a `commit()` per batch) so that per-segment SORTED_SET
 * ordinals genuinely disagree with the global ones and `OrdinalMap` has
 * something to prove.
 *
 * <p>The manifest carries real Lucene's own answers for everything the Rust
 * port claims to reproduce:
 *
 * <ul>
 *   <li>`OrdinalMap`: each segment's local term list, the global term list,
 *       and each segment's local-&gt;global ordinal map, taken straight off
 *       `MultiDocValues.getSortedSetValues(...).mapping`.
 *   <li>`SortedSetDocValuesFacetCounts`: `getAllDims`, `getTopChildren`,
 *       `getAllChildren` and `getSpecificValue` for a flat single-valued dim
 *       ("Author"), a flat multi-valued dim configured with
 *       `requireDimCount` ("Publish Year"), a flat multi-valued dim *without*
 *       it ("Tag", whose `value` Lucene reports as -1), and a hierarchical
 *       dim ("Path").
 *   <li>`LongRangeFacetCounts` over a single-valued NUMERIC field ("price"),
 *       over a genuinely multi-valued SORTED_NUMERIC field ("sizes"), and
 *       `DoubleRangeFacetCounts` over a DOUBLE field ("score").
 * </ul>
 */
public class GenFacets {
  /**
   * Manifest list separator. `\u0001` cannot occur in a facet label
   * (`FacetField.verifyLabel` rejects the whole C0 range except tab) and is not
   * `FacetsConfig`'s own `\u001F`/`\u001E`, so it can never collide.
   */
  static final char SEP = '\u0001';

  /** Docs, as (author, years[], tags[], path, price, sizes[], score). */
  record Doc(
      String author, String[] years, String[] tags, String path, long price, long[] sizes, double score) {}

  static final Doc[] DOCS = {
    new Doc("Lisa", new String[] {"2010", "2012"}, new String[] {"x"}, "a/b", 10, new long[] {1, 5}, -2.5),
    new Doc("Bob", new String[] {"2010"}, new String[] {"x", "y"}, "a/c", 25, new long[] {}, 0.0),
    new Doc("Lisa", new String[] {"2012"}, new String[] {}, "a/b", 25, new long[] {5}, 1.5),
    new Doc("Frank", new String[] {"1999"}, new String[] {"y"}, "d/e", 40, new long[] {2, 2, 9}, 7.25),
    new Doc("Bob", new String[] {"2010", "1999"}, new String[] {"x"}, "a/b", 40, new long[] {9}, -0.5),
    new Doc("Susan", new String[] {"2012"}, new String[] {"z"}, "d/f", 55, new long[] {1, 9}, 3.0),
    new Doc("Bob", new String[] {"2012"}, new String[] {"y", "z"}, "a/c", 70, new long[] {}, -7.0),
    new Doc("Lisa", new String[] {"1999"}, new String[] {"x"}, "d/e", 85, new long[] {3}, 12.0),
    new Doc("Frank", new String[] {"2010"}, new String[] {}, "a/b", 85, new long[] {1, 2, 3}, 0.5),
  };

  /** Segment boundaries: docs [0,3), [3,6), [6,9). */
  static final int[] COMMIT_AFTER = {3, 6, 9};

  static final LongRange[] PRICE_RANGES = {
    new LongRange("cheap", 0, true, 25, false),
    new LongRange("mid", 25, true, 70, false),
    new LongRange("dear", 70, true, Long.MAX_VALUE, true),
    // Deliberately overlaps "mid" and "dear": RangeFacetCounts counts a doc in
    // every range it falls in, and totCount must still count that doc once.
    new LongRange("over40", 40, true, Long.MAX_VALUE, true),
  };

  static final LongRange[] SIZE_RANGES = {
    new LongRange("small", 1, true, 3, false),
    new LongRange("medium", 3, true, 9, false),
    new LongRange("large", 9, true, 100, true),
    // Overlaps "small": doc 8 has sizes {1,2,3} so it lands in "small" twice
    // if the per-range dedup is missing.
    new LongRange("upto3", 1, true, 3, true),
  };

  static final DoubleRange[] SCORE_RANGES = {
    new DoubleRange("negative", Double.NEGATIVE_INFINITY, true, 0.0, false),
    new DoubleRange("zeroToOne", 0.0, true, 1.0, true),
    new DoubleRange("positive", 0.0, false, Double.POSITIVE_INFINITY, true),
  };

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("facets_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    FacetsConfig config = new FacetsConfig();
    config.setMultiValued("Publish Year", true);
    config.setRequireDimCount("Publish Year", true);
    config.setMultiValued("Tag", true);
    config.setHierarchical("Path", true);

    StringBuilder m = new StringBuilder();

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        int next = 0;
        for (int boundary : COMMIT_AFTER) {
          for (; next < boundary; next++) {
            Doc d = DOCS[next];
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(next), Field.Store.NO));
            doc.add(new SortedSetDocValuesFacetField("Author", d.author()));
            for (String y : d.years()) {
              doc.add(new SortedSetDocValuesFacetField("Publish Year", y));
            }
            for (String t : d.tags()) {
              doc.add(new SortedSetDocValuesFacetField("Tag", t));
            }
            String[] parts = d.path().split("/");
            doc.add(new SortedSetDocValuesFacetField("Path", parts));
            doc.add(new NumericDocValuesField("price", d.price()));
            for (long s : d.sizes()) {
              doc.add(new SortedNumericDocValuesField("sizes", s));
            }
            doc.add(new DoubleDocValuesField("score", d.score()));
            w.addDocument(config.build(doc));
          }
          w.commit();
        }
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      m.append("segment_count=").append(sis.size()).append('\n');
      int docBase = 0;
      for (int i = 0; i < sis.size(); i++) {
        SegmentCommitInfo sci = sis.info(i);
        String dvm = null, dvd = null, fnm = null;
        for (String f : sci.info.files()) {
          if (f.endsWith(".dvm")) dvm = f;
          if (f.endsWith(".dvd")) dvd = f;
          if (f.endsWith(".fnm")) fnm = f;
        }
        if (dvm == null || dvd == null || fnm == null) {
          throw new AssertionError("segment " + i + " is missing dv/fnm files: " + sci.info.files());
        }
        dump(dir, dvm, out);
        dump(dir, dvd, out);
        dump(dir, fnm, out);

        m.append("segment.").append(i).append(".name=").append(sci.info.name).append('\n');
        m.append("segment.").append(i).append(".id_hex=").append(hex(sci.info.getId())).append('\n');
        m.append("segment.").append(i).append(".max_doc=").append(sci.info.maxDoc()).append('\n');
        m.append("segment.").append(i).append(".doc_base=").append(docBase).append('\n');
        m.append("segment.").append(i).append(".dvm=").append(dvm).append('\n');
        m.append("segment.").append(i).append(".dvd=").append(dvd).append('\n');
        m.append("segment.").append(i).append(".fnm=").append(fnm).append('\n');

        FieldInfos fis =
            sci.info.getCodec().fieldInfosFormat().read(dir, sci.info, "", IOContext.READONCE);
        StringBuilder fieldNumbers = new StringBuilder();
        for (FieldInfo fi : fis) {
          if (fieldNumbers.length() > 0) fieldNumbers.append(',');
          fieldNumbers.append(fi.name).append(':').append(fi.number);
        }
        m.append("segment.").append(i).append(".field_numbers=").append(fieldNumbers).append('\n');
        docBase += sci.info.maxDoc();
      }
      m.append("max_doc=").append(docBase).append('\n');

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        // --- OrdinalMap ground truth ---------------------------------------
        SortedSetDocValues global = MultiDocValues.getSortedSetValues(reader, "$facets");
        if (!(global instanceof MultiDocValues.MultiSortedSetDocValues multi)) {
          throw new AssertionError("expected a multi-segment SortedSetDocValues");
        }
        OrdinalMap map = multi.mapping;
        m.append("ordmap.global_count=").append(global.getValueCount()).append('\n');
        StringBuilder globalTerms = new StringBuilder();
        for (long ord = 0; ord < global.getValueCount(); ord++) {
          if (ord > 0) globalTerms.append('');
          globalTerms.append(escape(global.lookupOrd(ord).utf8ToString()));
        }
        m.append("ordmap.global_terms=").append(globalTerms).append('\n');

        List<LeafReaderContext> leaves = reader.leaves();
        for (int i = 0; i < leaves.size(); i++) {
          SortedSetDocValues seg = leaves.get(i).reader().getSortedSetDocValues("$facets");
          StringBuilder localTerms = new StringBuilder();
          StringBuilder localToGlobal = new StringBuilder();
          LongValues g = map.getGlobalOrds(i);
          for (long ord = 0; ord < seg.getValueCount(); ord++) {
            if (ord > 0) {
              localTerms.append('');
              localToGlobal.append(',');
            }
            localTerms.append(escape(seg.lookupOrd(ord).utf8ToString()));
            localToGlobal.append(g.get(ord));
          }
          m.append("ordmap.seg.").append(i).append(".count=").append(seg.getValueCount()).append('\n');
          m.append("ordmap.seg.").append(i).append(".terms=").append(localTerms).append('\n');
          m.append("ordmap.seg.").append(i).append(".to_global=").append(localToGlobal).append('\n');
        }

        // --- SortedSetDocValuesFacetCounts ground truth ---------------------
        IndexSearcher searcher = new IndexSearcher(reader);
        FacetsCollector fc =
            searcher.search(new MatchAllDocsQuery(), new FacetsCollectorManager());
        SortedSetDocValuesReaderState state =
            new DefaultSortedSetDocValuesReaderState(reader, config);
        Facets facets = new SortedSetDocValuesFacetCounts(state, fc);

        StringBuilder dims = new StringBuilder();
        for (String dim : state.getDims()) {
          if (dims.length() > 0) dims.append(',');
          dims.append(escape(dim));
        }
        m.append("state.dims=").append(dims).append('\n');

        List<FacetResult> allDims = facets.getAllDims(10);
        m.append("alldims.count=").append(allDims.size()).append('\n');
        for (int i = 0; i < allDims.size(); i++) {
          appendResult(m, "alldims." + i, allDims.get(i));
        }

        for (String dim : new String[] {"Author", "Publish Year", "Tag"}) {
          appendResult(m, "top." + escape(dim), facets.getTopChildren(10, dim));
          appendResult(m, "top2." + escape(dim), facets.getTopChildren(2, dim));
          appendResult(m, "all." + escape(dim), facets.getAllChildren(dim));
        }
        appendResult(m, "top.Path", facets.getTopChildren(10, "Path"));
        appendResult(m, "top.Path.a", facets.getTopChildren(10, "Path", "a"));
        appendResult(m, "all.Path.d", facets.getAllChildren("Path", "d"));

        m.append("specific.Author.Bob=").append(facets.getSpecificValue("Author", "Bob")).append('\n');
        m.append("specific.Author.Nobody=")
            .append(facets.getSpecificValue("Author", "Nobody"))
            .append('\n');
        m.append("specific.Path.a=").append(facets.getSpecificValue("Path", "a")).append('\n');
        m.append("specific.Path.a.b=").append(facets.getSpecificValue("Path", "a", "b")).append('\n');

        // --- Range faceting ground truth -----------------------------------
        Facets priceFacets = new LongRangeFacetCounts("price", fc, PRICE_RANGES);
        appendResult(m, "range.price", priceFacets.getAllChildren("price"));
        appendResult(m, "rangetop.price", priceFacets.getTopChildren(10, "price"));

        Facets sizeFacets = new LongRangeFacetCounts("sizes", fc, SIZE_RANGES);
        appendResult(m, "range.sizes", sizeFacets.getAllChildren("sizes"));

        Facets scoreFacets = new DoubleRangeFacetCounts("score", fc, SCORE_RANGES);
        appendResult(m, "range.score", scoreFacets.getAllChildren("score"));
      }

      appendBuildCases(m);
      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote facets_index/ fixture directory");
  }

  /** Sub-separator inside one manifest list entry. */
  static final char SUB = '';

  /**
   * One `FacetsConfig.build` case: the dim configuration, the facet labels of
   * a single document, and exactly what `build` turned them into.
   */
  record BuildCase(String name, FacetsConfig config, String[][] labels) {}

  /**
   * `dim` config as
   * `dim SUB hierarchical SUB multiValued SUB requireDimCount SUB drillDown SUB indexFieldName`.
   */
  static String describeDim(FacetsConfig config, String dim) {
    FacetsConfig.DimConfig c = config.getDimConfig(dim);
    return escape(dim)
        + SUB + c.hierarchical
        + SUB + c.multiValued
        + SUB + c.requireDimCount
        + SUB + c.drillDownTermsIndexing
        + SUB + escape(c.indexFieldName);
  }

  static FacetsConfig configWith(java.util.function.Consumer<FacetsConfig> setup) {
    FacetsConfig c = new FacetsConfig();
    setup.accept(c);
    return c;
  }

  /**
   * Records what `FacetsConfig.build(Document)` produces for a range of dim
   * configurations -- the write-side half of faceting, and the only reason
   * every read-side key in this manifest decodes the way it does. A port that
   * gets the read side right against a hand-built index proves nothing about
   * whether it would have *written* the same one.
   *
   * <p>Per case: the input labels, the dim configuration in force, the
   * `SortedSetDocValuesField` values `build` emitted (in order, per index
   * field) and the drill-down `StringField` terms beside them.
   */
  static void appendBuildCases(StringBuilder m) throws IOException {
    BuildCase[] cases = {
      new BuildCase("flat_default", new FacetsConfig(), new String[][] {{"Author", "Lisa"}}),
      new BuildCase(
          "flat_multi_require_dim_count",
          configWith(
              c -> {
                c.setMultiValued("Publish Year", true);
                c.setRequireDimCount("Publish Year", true);
              }),
          new String[][] {{"Publish Year", "2010"}, {"Publish Year", "2012"}}),
      new BuildCase(
          "flat_multi_no_dim_count",
          configWith(c -> c.setMultiValued("Tag", true)),
          new String[][] {{"Tag", "x"}, {"Tag", "y"}}),
      new BuildCase(
          "hierarchical",
          configWith(c -> c.setHierarchical("Path", true)),
          new String[][] {{"Path", "a", "b"}}),
      new BuildCase(
          "hierarchical_deep",
          configWith(c -> c.setHierarchical("Path", true)),
          new String[][] {{"Path", "a", "b", "c"}}),
      new BuildCase(
          "drilldown_none",
          configWith(
              c -> {
                c.setHierarchical("Path", true);
                c.setDrillDownTermsIndexing("Path", FacetsConfig.DrillDownTermsIndexing.NONE);
              }),
          new String[][] {{"Path", "a", "b", "c"}}),
      new BuildCase(
          "drilldown_full_path_only",
          configWith(
              c -> {
                c.setHierarchical("Path", true);
                c.setDrillDownTermsIndexing(
                    "Path", FacetsConfig.DrillDownTermsIndexing.FULL_PATH_ONLY);
              }),
          new String[][] {{"Path", "a", "b", "c"}}),
      new BuildCase(
          "drilldown_all_paths_no_dim",
          configWith(
              c -> {
                c.setHierarchical("Path", true);
                c.setDrillDownTermsIndexing(
                    "Path", FacetsConfig.DrillDownTermsIndexing.ALL_PATHS_NO_DIM);
              }),
          new String[][] {{"Path", "a", "b", "c"}}),
      new BuildCase(
          "drilldown_dimension_and_full_path",
          configWith(
              c -> {
                c.setHierarchical("Path", true);
                c.setDrillDownTermsIndexing(
                    "Path", FacetsConfig.DrillDownTermsIndexing.DIMENSION_AND_FULL_PATH);
              }),
          new String[][] {{"Path", "a", "b", "c"}}),
      new BuildCase(
          "custom_index_field",
          configWith(c -> c.setIndexFieldName("Author", "$author")),
          new String[][] {{"Author", "Lisa"}, {"Tag", "x"}}),
      new BuildCase(
          "escaped_component",
          configWith(c -> c.setHierarchical("Path", true)),
          // A component containing a '/', which is NOT FacetsConfig's path
          // delimiter (that is U+001F, which FacetField.verifyLabel forbids
          // inside a label): it must survive as one component, not be split.
          new String[][] {{"Path", "a/b", "c"}}),
    };

    m.append("build_count=").append(cases.length).append('\n');
    for (int i = 0; i < cases.length; i++) {
      BuildCase bc = cases[i];
      String prefix = "build." + i;
      m.append(prefix).append(".name=").append(bc.name()).append('\n');

      StringBuilder dims = new StringBuilder();
      StringBuilder labels = new StringBuilder();
      Document doc = new Document();
      java.util.LinkedHashSet<String> seenDims = new java.util.LinkedHashSet<>();
      for (String[] label : bc.labels()) {
        String dim = label[0];
        String[] path = java.util.Arrays.copyOfRange(label, 1, label.length);
        if (seenDims.add(dim)) {
          if (dims.length() > 0) dims.append(SEP);
          dims.append(describeDim(bc.config(), dim));
        }
        if (labels.length() > 0) labels.append(SEP);
        labels.append(escape(dim));
        for (String p : path) {
          labels.append(SUB).append(escape(p));
        }
        doc.add(new SortedSetDocValuesFacetField(dim, path));
      }
      m.append(prefix).append(".dims=").append(dims).append('\n');
      m.append(prefix).append(".labels=").append(labels).append('\n');

      Document built = bc.config().build(doc);
      StringBuilder ssdv = new StringBuilder();
      StringBuilder terms = new StringBuilder();
      for (org.apache.lucene.index.IndexableField f : built.getFields()) {
        if (f.fieldType().docValuesType() == org.apache.lucene.index.DocValuesType.SORTED_SET) {
          if (ssdv.length() > 0) ssdv.append(SEP);
          ssdv.append(escape(f.name())).append(SUB).append(escape(f.binaryValue().utf8ToString()));
        } else if (f.stringValue() != null) {
          if (terms.length() > 0) terms.append(SEP);
          terms.append(escape(f.name())).append(SUB).append(escape(f.stringValue()));
        }
      }
      m.append(prefix).append(".ssdv=").append(ssdv).append('\n');
      m.append(prefix).append(".terms=").append(terms).append('\n');
    }
  }

  /**
   * `FacetResult` as three manifest keys: the dim value (Lucene's `-1` for a
   * multi-valued dim without `requireDimCount` included verbatim), the child
   * count, and the ordered `label=count` list.
   */
  static void appendResult(StringBuilder m, String prefix, FacetResult r) {
    if (r == null) {
      m.append(prefix).append(".null=true\n");
      return;
    }
    m.append(prefix).append(".value=").append(r.value).append('\n');
    m.append(prefix).append(".child_count=").append(r.childCount).append('\n');
    StringBuilder children = new StringBuilder();
    for (LabelAndValue lv : r.labelValues) {
      if (children.length() > 0) children.append('');
      children.append(escape(lv.label)).append('=').append(lv.value);
    }
    m.append(prefix).append(".children=").append(children).append('\n');
  }

  /** Newlines and the record separators would break the flat manifest format. */
  static String escape(String s) {
    return s.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace(String.valueOf(SEP), "\\u0001")
        .replace(String.valueOf(SUB), "\\u0002")
        .replace("", "\\u001F")
        .replace("", "\\u001E");
  }

  static void dump(Directory dir, String fileName, Path out) throws IOException {
    try (IndexInput in = dir.openInput(fileName, IOContext.READONCE)) {
      byte[] bytes = new byte[(int) in.length()];
      in.readBytes(bytes, 0, bytes.length);
      Files.write(out.resolve(fileName + ".raw"), bytes);
    }
  }

  static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder();
    for (byte x : b) sb.append(String.format("%02x", x));
    return sb.toString();
  }

  static void deleteRecursive(Path p) throws IOException {
    try (var s = Files.walk(p)) {
      s.sorted(java.util.Comparator.reverseOrder())
          .forEach(
              q -> {
                try {
                  Files.delete(q);
                } catch (IOException e) {
                  throw new RuntimeException(e);
                }
              });
    }
  }
}
