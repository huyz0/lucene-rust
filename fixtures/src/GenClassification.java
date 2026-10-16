import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.stream.Stream;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.classification.BM25NBClassifier;
import org.apache.lucene.classification.BooleanPerceptronClassifier;
import org.apache.lucene.classification.CachingNaiveBayesClassifier;
import org.apache.lucene.classification.ClassificationResult;
import org.apache.lucene.classification.Classifier;
import org.apache.lucene.classification.KNearestFuzzyClassifier;
import org.apache.lucene.classification.KNearestNeighborClassifier;
import org.apache.lucene.classification.SimpleNaiveBayesClassifier;
import org.apache.lucene.classification.document.DocumentClassifier;
import org.apache.lucene.classification.document.KNearestNeighborDocumentClassifier;
import org.apache.lucene.classification.document.SimpleNaiveBayesDocumentClassifier;
import org.apache.lucene.classification.utils.ConfusionMatrixGenerator;
import org.apache.lucene.classification.utils.DatasetSplitter;
import org.apache.lucene.classification.utils.DocToDoubleVectorUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermVectors;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * M14 T14.3 ({@code classification}): what Lucene's classifiers make of two indexes.
 *
 * <p>{@code sample/} holds {@code ClassificationTestBase}'s documents (its sample index, then its
 * update, as two segments) analysed by {@code StandardAnalyzer}; {@code random/} 240 documents of
 * a small vocabulary whose word frequencies depend on the class, analysed by {@code
 * WhitespaceAnalyzer}, in one segment; {@code fuzzy/} (see {@link #fuzzy}) the fuzzy classifiers'
 * variant queue overflowing. Both store {@code text}, {@code cat} and {@code bool} with
 * term vectors (positions and offsets), {@code cat} with sorted doc values too.
 *
 * <p>{@code classification.tsv}, one row per line, fields tab-separated, values as hex bits:
 *
 * <ul>
 *   <li>{@code A index classifier input class:score} -- {@code assignClass} ({@code -} for none);
 *   <li>{@code L index classifier input class:score,...} -- {@code getClasses(input)};
 *   <li>{@code D index classifier input class:score | class:score,...} -- a document classifier's
 *       {@code assignClass} and {@code getClasses};
 *   <li>{@code M index classifier field n accuracy precision recall f1 matrix per-class} -- the
 *       confusion matrix;
 *   <li>{@code V doc dense sparse} -- {@code DocToDoubleVectorUtils} over {@code random/};
 *   <li>{@code S split part doc tv field=value...} -- the documents {@code DatasetSplitter} writes
 *       under each of {@link #SPLITS}, whether each has term vectors, and its stored fields;
 *   <li>{@code T split part doc field axes term:freq:position/start-end...} -- after its {@code S}
 *       row, each term vector of that document (see {@link #vectorRow}).
 * </ul>
 *
 * Java orders classes of equal score, and sums {@code NearestFuzzyQuery}'s clauses, by {@code
 * BytesRef} hash, seeded by {@code StringHelper.GOOD_FAST_HASH_SEED}: {@code gen-fixtures.sh} pins
 * it with {@code -Dtests.seed=C1A55}, and the Rust test hashes with the same seed.
 */
public class GenClassification {

  static final String[] SAMPLE = {
    "politics",
    "The traveling press secretary for Mitt Romney lost his cool and cursed at reporters who"
        + " attempted to ask questions of the Republican presidential candidate in a public plaza"
        + " near the Tomb of the Unknown Soldier in Warsaw Tuesday.",
    "politics",
    "Mitt Romney seeks to assure Israel and Iran, as well as Jewish voters in the United States,"
        + " that he will be tougher against Iran's nuclear ambitions than President Barack Obama.",
    "politics",
    "And there's a threshold question that he has to answer for the American people and that's"
        + " whether he is prepared to be commander-in-chief,\" she continued. \"As we look to the"
        + " past events, we know that this raises some questions about his preparedness and we'll"
        + " see how the rest of his trip goes.\"",
    "politics",
    "Still, when it comes to gun policy, many congressional Democrats have \"decided to keep quiet"
        + " and not go there,\" said Alan Lizotte, dean and professor at the State University of"
        + " New York at Albany's School of Criminal Justice.",
    "technology",
    "Standing amongst the thousands of people at the state Capitol, Jorstad, director of"
        + " technology at the University of Wisconsin-La Crosse, documented the historic moment and"
        + " shared it with the world through the Internet.",
    "technology",
    "So, about all those experts and analysts who've spent the past year or so saying Facebook was"
        + " going to make a phone. A new expert has stepped forward to say it's not going to"
        + " happen.",
    "technology",
    "More than 400 million people trust Google with their e-mail, and 50 million store files in"
        + " the cloud using the Dropbox service. People manage their bank accounts, pay bills, trade"
        + " stocks and generally transfer or store huge volumes of personal data online.",
    null,
    "unlabeled doc",
  };

  static final String[] UPDATE = {
    "politics",
    "Warren Bennis says John F. Kennedy grasped a key lesson about the presidency that few have"
        + " followed.",
    "politics",
    "Julian Zelizer says Bill Clinton is still trying to shape his party, years after the White"
        + " House, while George W. Bush opts for a much more passive role.",
    "politics",
    "Crossfire: Sen. Tim Scott passes on Sen. Lindsey Graham endorsement",
    "politics",
    "Illinois becomes 16th state to allow same-sex marriage.",
    "technology",
    "Apple is developing iPhones with curved-glass screens and enhanced sensors that detect"
        + " different levels of pressure, according to a new report.",
    "technology",
    "The Xbox One is Microsoft's first new gaming console in eight years. It's a quality piece of"
        + " hardware but it's also noteworthy because Microsoft is using it to make a statement.",
    "technology",
    "Google says it will replace a Google Maps image after a California father complained it shows"
        + " the body of his teen-age son, who was shot to death in 2009.",
    null,
    "second unlabeled doc",
  };

  static final String[] SAMPLE_INPUTS = {
    "Here are some interesting questions and answers about Mitt Romney.. If you don't know the"
        + " answer to the question about Mitt Romney, then simply click on the answer below the"
        + " question section.",
    "Much is made of what the likes of Facebook, Google and Apple know about users. Truth is,"
        + " Amazon may know more.",
    "Much is made of what the likes of Facebook, Google and Apple know about users. Truth is,"
        + " Amazon may know more. This technology observation is extracted from the internet.",
    "More than 400 million people trust Google with their e-mail, and 50 million store files in"
        + " the cloud using the Dropbox service. People manage their bank accounts, pay bills, trade"
        + " stocks and generally transfer or store huge volumes of personal data online. traveling"
        + " seeks raises some questions Republican presidential. ",
    "nothing here is in the index zzz",
    "",
  };

  static final int CLASSES = 4;
  static final int VOCABULARY = 40;

  /** A word of {@code random/}: each class favours its own quarter of the vocabulary. */
  static String word(Random r, int cls) {
    int w =
        r.nextInt(3) == 0
            ? r.nextInt(VOCABULARY)
            : cls * (VOCABULARY / CLASSES) + r.nextInt(VOCABULARY / CLASSES);
    // a few near-miss spellings, for the fuzzy classifier
    return (r.nextInt(25) == 0 ? "x" : "w") + w;
  }

  static String randomText(Random r, int cls, int words) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < words; i++) b.append(i == 0 ? "" : " ").append(word(r, cls));
    return b.toString();
  }

  static FieldType textType() {
    FieldType ft = new FieldType(TextField.TYPE_STORED);
    ft.setStoreTermVectors(true);
    ft.setStoreTermVectorOffsets(true);
    ft.setStoreTermVectorPositions(true);
    ft.freeze();
    return ft;
  }

  static Document doc(FieldType ft, String text, String cat) {
    Document doc = new Document();
    doc.add(new Field("text", text, ft));
    if (cat != null) {
      doc.add(new Field("cat", cat, ft));
      doc.add(new SortedDocValuesField("cat", new BytesRef(cat)));
      doc.add(new Field("bool", String.valueOf(!cat.equals("technology") && !cat.equals("c2") && !cat.equals("c3")), ft));
    }
    return doc;
  }

  static void recreate(Path dirPath) throws IOException {
    if (Files.exists(dirPath)) {
      try (Stream<Path> st = Files.walk(dirPath)) {
        st.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(dirPath);
  }

  static IndexWriterConfig config(Analyzer a) {
    IndexWriterConfig cfg = new IndexWriterConfig(a);
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    return cfg;
  }

  static void writeSample(Path dirPath) throws IOException {
    recreate(dirPath);
    FieldType ft = textType();
    try (Directory dir = FSDirectory.open(dirPath);
        IndexWriter w = new IndexWriter(dir, config(new StandardAnalyzer()))) {
      for (String[] batch : new String[][] {SAMPLE, UPDATE}) {
        for (int i = 0; i < batch.length; i += 2) w.addDocument(doc(ft, batch[i + 1], batch[i]));
        w.commit();
      }
    }
  }

  static List<String> randomInputs = new ArrayList<>();

  static void writeRandom(Path dirPath) throws IOException {
    recreate(dirPath);
    FieldType ft = textType();
    Random r = new Random(0xc1a55L);
    try (Directory dir = FSDirectory.open(dirPath);
        IndexWriter w = new IndexWriter(dir, config(new WhitespaceAnalyzer()))) {
      for (int i = 0; i < 240; i++) {
        int cls = r.nextInt(CLASSES);
        // a few documents without a class, and one class missing from the bool field's view
        String cat = i % 41 == 0 ? null : "c" + cls;
        w.addDocument(doc(ft, randomText(r, cls, 3 + r.nextInt(12)), cat));
      }
      w.commit();
    }
    for (int i = 0; i < 24; i++) randomInputs.add(randomText(r, i % CLASSES, 2 + r.nextInt(10)));
    randomInputs.add("w1 w1 w1 x39");
    randomInputs.add("unknown words only");
  }

  static String hex(double d) {
    return Long.toHexString(Double.doubleToRawLongBits(d));
  }

  static String esc(String s) {
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n");
  }

  static String cls(Object c) {
    return c instanceof BytesRef b ? b.utf8ToString() : String.valueOf(c);
  }

  static String result(ClassificationResult<?> r) {
    return r == null ? "-" : esc(cls(r.assignedClass())) + ":" + hex(r.score());
  }

  static String results(List<? extends ClassificationResult<?>> rs) {
    if (rs == null) return "null";
    StringBuilder b = new StringBuilder();
    for (ClassificationResult<?> r : rs) b.append(b.length() == 0 ? "" : ",").append(result(r));
    return b.toString();
  }

  /** A classifier configuration, by the name the Rust test builds it from. */
  static Classifier<?> classifier(String name, IndexReader reader, Analyzer a) throws IOException {
    Query q = new TermQuery(new Term("bool", "true"));
    return switch (name) {
      case "snb" -> new SimpleNaiveBayesClassifier(reader, a, null, "cat", "text");
      case "snb-q" -> new SimpleNaiveBayesClassifier(reader, a, q, "cat", "text");
      case "snb-bool" -> new SimpleNaiveBayesClassifier(reader, a, null, "bool", "text");
      case "cnb" -> new CachingNaiveBayesClassifier(reader, a, null, "cat", "text");
      case "cnb-q" -> new CachingNaiveBayesClassifier(reader, a, q, "cat", "text");
      case "bm25nb" -> new BM25NBClassifier(reader, a, null, "cat", "text");
      case "bm25nb-q" -> new BM25NBClassifier(reader, a, q, "cat", "text");
      case "knn1" -> new KNearestNeighborClassifier(reader, null, a, null, 1, 0, 0, "cat", "text");
      case "knn3" -> new KNearestNeighborClassifier(reader, null, a, null, 3, 0, 0, "cat", "text");
      case "knn3-classic" ->
          new KNearestNeighborClassifier(
              reader, new ClassicSimilarity(), a, null, 3, 0, 0, "cat", "text");
      case "knn5-df" -> new KNearestNeighborClassifier(reader, null, a, null, 5, 1, 1, "cat", "text");
      case "knn3-boost" ->
          new KNearestNeighborClassifier(reader, null, a, null, 3, 0, 0, "cat", "text^2.5");
      case "knn3-q" -> new KNearestNeighborClassifier(reader, null, a, q, 3, 0, 0, "cat", "text");
      case "fuzzy3" -> new KNearestFuzzyClassifier(reader, null, a, null, 3, "cat", "text");
      case "fuzzy5-classic" ->
          new KNearestFuzzyClassifier(reader, new ClassicSimilarity(), a, null, 5, "cat", "text");
      case "perceptron" -> new BooleanPerceptronClassifier(reader, a, null, 1, null, "bool", "text");
      case "perceptron-b3" ->
          new BooleanPerceptronClassifier(reader, a, null, 3, 5.0, "bool", "text");
      default -> throw new IllegalArgumentException(name);
    };
  }

  static final String[] CLASSIFIERS = {
    "snb", "snb-q", "snb-bool", "cnb", "cnb-q", "bm25nb", "bm25nb-q", "knn1", "knn3", "knn3-classic",
    "knn5-df", "knn3-boost", "knn3-q", "fuzzy3", "fuzzy5-classic", "perceptron", "perceptron-b3"
  };

  static void classify(StringBuilder rows, String index, IndexReader reader, Analyzer a, List<String> inputs)
      throws IOException {
    for (String name : CLASSIFIERS) {
      Classifier<?> c = classifier(name, reader, a);
      for (int i = 0; i < inputs.size(); i++) {
        String in = inputs.get(i);
        rows.append("A\t").append(index).append('\t').append(name).append('\t').append(i).append('\t')
            .append(result(c.assignClass(in))).append('\n');
        rows.append("L\t").append(index).append('\t').append(name).append('\t').append(i).append('\t')
            .append(results(c.getClasses(in))).append('\n');
      }
      // the confusion matrix over the index the classifier learnt from
      String field = name.startsWith("perceptron") || name.equals("snb-bool") ? "bool" : "cat";
      ConfusionMatrixGenerator.ConfusionMatrix m =
          ConfusionMatrixGenerator.getConfusionMatrix(reader, c, field, "text", -1);
      rows.append("M\t").append(index).append('\t').append(name).append('\t').append(field).append('\t')
          .append(m.getNumberOfEvaluatedDocs()).append('\t').append(hex(m.getAccuracy())).append('\t')
          .append(hex(m.getPrecision())).append('\t').append(hex(m.getRecall())).append('\t')
          .append(hex(m.getF1Measure())).append('\t');
      TreeMap<String, TreeMap<String, Long>> sorted = new TreeMap<>();
      m.getLinearizedMatrix().forEach((k, v) -> sorted.put(k, new TreeMap<>(v)));
      StringBuilder mx = new StringBuilder();
      StringBuilder per = new StringBuilder();
      for (var e : sorted.entrySet()) {
        for (var v : e.getValue().entrySet()) {
          mx.append(mx.length() == 0 ? "" : ",").append(e.getKey()).append('>').append(v.getKey())
              .append('=').append(v.getValue());
        }
        per.append(per.length() == 0 ? "" : ",").append(e.getKey()).append('=')
            .append(hex(m.getPrecision(e.getKey()))).append('/').append(hex(m.getRecall(e.getKey())))
            .append('/').append(hex(m.getF1Measure(e.getKey())));
      }
      rows.append(mx).append('\t').append(per).append('\n');
    }
  }

  static void documents(StringBuilder rows, String index, IndexReader reader, Analyzer a, List<String> inputs)
      throws IOException {
    Map<String, Analyzer> f2a = new HashMap<>();
    f2a.put("text", a);
    f2a.put("title", a);
    FieldType ft = textType();
    for (String name : new String[] {"knn-doc", "knn-doc-boost", "snb-doc", "snb-doc-boost"}) {
      boolean boost = name.endsWith("-boost");
      DocumentClassifier<BytesRef> c =
          name.startsWith("knn")
              ? new KNearestNeighborDocumentClassifier(
                  reader, null, null, 3, 0, 0, "cat", f2a, boost ? new String[] {"text^2", "title"} : new String[] {"text"})
              : new SimpleNaiveBayesDocumentClassifier(
                  reader, null, "cat", f2a, boost ? new String[] {"text^2"} : new String[] {"text"});
      for (int i = 0; i + 1 < inputs.size(); i++) {
        Document d = new Document();
        d.add(new Field("text", inputs.get(i), ft));
        d.add(new Field("text", inputs.get(i + 1), ft));
        d.add(new Field("title", inputs.get(inputs.size() - 1 - i), ft));
        rows.append("D\t").append(index).append('\t').append(name).append('\t').append(i).append('\t')
            .append(result(c.assignClass(d))).append(" | ").append(results(c.getClasses(d))).append('\n');
      }
    }
  }

  /** {@code DatasetSplitter} runs: name, index, term vectors, class field, field names. */
  /**
   * A {@code T} row: one document's term vector of {@code field}, its axes, then each term with its
   * frequency and every occurrence's position and offsets.
   */
  static void vectorRow(StringBuilder rows, String key, String field, Terms terms)
      throws IOException {
    rows.append("T\t").append(key).append('\t').append(field).append('\t')
        .append(terms.hasPositions() ? 'p' : '-')
        .append(terms.hasOffsets() ? 'o' : '-')
        .append(terms.hasPayloads() ? 'y' : '-');
    TermsEnum te = terms.iterator();
    for (BytesRef term = te.next(); term != null; term = te.next()) {
      rows.append('\t').append(esc(term.utf8ToString())).append(':').append(te.totalTermFreq());
      PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
      pe.nextDoc();
      for (int i = 0; i < pe.freq(); i++) {
        int position = pe.nextPosition();
        rows.append(':');
        if (terms.hasPositions()) {
          rows.append(position);
        }
        if (terms.hasOffsets()) {
          rows.append('/').append(pe.startOffset()).append('-').append(pe.endOffset());
        }
      }
    }
    rows.append('\n');
  }

  static final String[][] SPLITS = {
    {"sample", "sample", "-", "cat", "text,cat"},
    {"random", "random", "-", "cat", "text,cat"},
    {"random-tv", "random", "tv", "cat", ""},
    {"random-bool", "random", "-", "bool", "text,bool"},
  };

  /**
   * {@code fuzzy/}: one token with more than {@code MAX_VARIANTS_PER_TERM} (50) variants within one
   * edit, so {@code NearestFuzzyQuery}'s variant queue fills and {@code FuzzyTermsEnum} drops its
   * edit budget as the queue's bottom rises.
   */
  static void fuzzy(StringBuilder rows, Path dirPath) throws IOException {
    recreate(dirPath);
    FieldType ft = textType();
    List<String> words = new ArrayList<>();
    for (char c = 'a'; c <= 'z'; c++) {
      words.add("abcd" + c); // an insertion at the end
      words.add("abc" + c); // a substitution of the last letter
      words.add("ab" + c + "d"); // a substitution of the third
      words.add("ab" + c + c + "d"); // two edits: outside the budget once it drops
    }
    words.add("abd");
    words.add("abcd");
    words.add("bacd");
    try (Directory dir = FSDirectory.open(dirPath);
        IndexWriter w = new IndexWriter(dir, config(new WhitespaceAnalyzer()))) {
      for (int i = 0; i < words.size(); i++) {
        String text = words.get(i) + " " + words.get((i * 7 + 3) % words.size());
        w.addDocument(doc(ft, text, "f" + (i % 3)));
      }
      w.commit();
    }
    String[] inputs = {"abcd", "abce abdd", "abzzd", "abcdz abcd"};
    try (Directory dir = FSDirectory.open(dirPath);
        DirectoryReader r = DirectoryReader.open(dir)) {
      for (String name : new String[] {"fuzzy3", "fuzzy5-classic"}) {
        Classifier<?> c = classifier(name, r, new WhitespaceAnalyzer());
        for (int i = 0; i < inputs.length; i++) {
          rows.append("A\tfuzzy\t").append(name).append('\t').append(i).append('\t')
              .append(result(c.assignClass(inputs[i]))).append('\n');
          rows.append("L\tfuzzy\t").append(name).append('\t').append(i).append('\t')
              .append(results(c.getClasses(inputs[i]))).append('\n');
        }
      }
    }
  }

  static String vector(Double[] v) {
    if (v == null) return "null";
    StringBuilder b = new StringBuilder();
    for (Double d : v) b.append(b.length() == 0 ? "" : ",").append(d == null ? "null" : hex(d));
    return b.toString();
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("classification");
    Files.createDirectories(out);
    writeSample(out.resolve("sample"));
    writeRandom(out.resolve("random"));
    StringBuilder rows = new StringBuilder();
    try (Directory sd = FSDirectory.open(out.resolve("sample"));
        DirectoryReader sample = DirectoryReader.open(sd);
        Directory rd = FSDirectory.open(out.resolve("random"));
        DirectoryReader random = DirectoryReader.open(rd)) {
      classify(rows, "sample", sample, new StandardAnalyzer(), List.of(SAMPLE_INPUTS));
      classify(rows, "random", random, new WhitespaceAnalyzer(), randomInputs);
      documents(rows, "sample", sample, new StandardAnalyzer(), List.of(SAMPLE_INPUTS));
      documents(rows, "random", random, new WhitespaceAnalyzer(), randomInputs);

      LeafReader leaf = random.leaves().get(0).reader();
      TermVectors tvs = leaf.termVectors();
      Terms fieldTerms = leaf.terms("text");
      for (int doc = 0; doc < leaf.maxDoc(); doc++) {
        Terms docTerms = tvs.get(doc, "text");
        rows.append("V\t").append(doc).append('\t')
            .append(vector(DocToDoubleVectorUtils.toDenseLocalFreqDoubleArray(docTerms))).append('\t')
            .append(vector(DocToDoubleVectorUtils.toSparseLocalFreqDoubleArray(docTerms, fieldTerms)))
            .append('\n');
      }

      // DatasetSplitter closes the reader it splits: give it one of its own
      for (String[] spec : SPLITS) {
        String name = spec[0];
        String index = spec[1];
        boolean termVectors = spec[2].equals("tv");
        String classField = spec[3];
        String[] fieldNames = spec[4].isEmpty() ? new String[0] : spec[4].split(",");
        Directory train = new ByteBuffersDirectory();
        Directory test = new ByteBuffersDirectory();
        Directory cv = new ByteBuffersDirectory();
        try (Directory src = FSDirectory.open(out.resolve(index))) {
          DirectoryReader r = DirectoryReader.open(src);
          Analyzer a = index.equals("sample") ? new StandardAnalyzer() : new WhitespaceAnalyzer();
          new DatasetSplitter(0.2, 0.1)
              .split(r, train, test, cv, a, termVectors, classField, fieldNames);
        } catch (IOException e) {
          // a class field without sorted doc values: GroupingSearch's TermGroupSelector rejects it
          rows.append("S\t").append(name).append("\t!")
              .append(e.getCause().getClass().getSimpleName()).append('\n');
          continue;
        }
        String[] names = {"train", "test", "cv"};
        Directory[] dirs = {train, test, cv};
        for (int s = 0; s < 3; s++) {
          try (DirectoryReader r = DirectoryReader.open(dirs[s])) {
            StoredFields sf = r.storedFields();
            TermVectors tv = r.termVectors();
            for (int doc = 0; doc < r.maxDoc(); doc++) {
              Document d = sf.document(doc);
              rows.append("S\t").append(name).append('\t').append(names[s]).append('\t').append(doc)
                  .append('\t').append(tv.get(doc) == null ? "-" : "tv");
              for (IndexableField f : d.getFields()) {
                rows.append('\t').append(f.name()).append('=').append(esc(f.stringValue()));
              }
              rows.append('\n');
              Fields vectors = tv.get(doc);
              if (vectors != null) {
                for (String field : vectors) {
                  vectorRow(rows, name + "\t" + names[s] + "\t" + doc, field, vectors.terms(field));
                }
              }
            }
          }
          dirs[s].close();
        }
      }
    }
    fuzzy(rows, out.resolve("fuzzy"));
    Files.writeString(out.resolve("classification.tsv"), rows.toString(), StandardCharsets.UTF_8);
  }
}
