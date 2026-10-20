import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.payloads.DelimitedPayloadTokenFilter;
import org.apache.lucene.analysis.payloads.IdentityEncoder;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.IntField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoDeletionPolicy;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.luke.models.analysis.Analysis;
import org.apache.lucene.luke.models.analysis.AnalysisImpl;
import org.apache.lucene.luke.models.analysis.CustomAnalyzerConfig;
import org.apache.lucene.luke.models.commits.Commit;
import org.apache.lucene.luke.models.commits.CommitsImpl;
import org.apache.lucene.luke.models.commits.Segment;
import org.apache.lucene.luke.models.documents.DocValues;
import org.apache.lucene.luke.models.documents.DocumentField;
import org.apache.lucene.luke.models.documents.DocumentsImpl;
import org.apache.lucene.luke.models.documents.TermPosting;
import org.apache.lucene.luke.models.documents.TermVectorEntry;
import org.apache.lucene.luke.models.overview.OverviewImpl;
import org.apache.lucene.luke.models.overview.TermCountsOrder;
import org.apache.lucene.luke.models.overview.TermStats;
import org.apache.lucene.luke.models.search.MLTConfig;
import org.apache.lucene.luke.models.search.QueryParserConfig;
import org.apache.lucene.luke.models.search.SearchImpl;
import org.apache.lucene.luke.models.search.SearchResults;
import org.apache.lucene.luke.models.search.SimilarityConfig;
import org.apache.lucene.luke.models.tools.IndexToolsImpl;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.NumericUtils;

/**
 * M14.i: what Luke's index-inspector models report, through a command line that is the twin of
 * {@code lucene-tools Luke}. Writes {@code tools/luke}:
 *
 * <ul>
 *   <li>{@code index}, three commits kept by {@code NoDeletionPolicy} (three segments, user
 *       data on each, a deleted document in the second), its fields modelled on Luke's own
 *       {@code DocumentsTestBase}: stored text without norms, text with positions and offsets,
 *       term vectors, payloads, points, every doc-values type and a vector field;
 *   <li>{@code news}, three messages in the 20 Newsgroups format {@code createNewIndex} reads;
 *   <li>{@code cases.tsv}, one command per line: {@code id, command, args...}. An argument may
 *       name {@code {fixture}} (this directory, read only) or {@code {tmp}} (one scratch
 *       directory the commands share, in order). {@code copy a b} copies a directory;
 *   <li>{@code out/<id>.out}, what the command printed. A model's exception ends the command
 *       with the line {@code error: <simple class name>: <message>}.
 * </ul>
 *
 * Every command runs the Luke models themselves ({@code OverviewImpl}, {@code DocumentsImpl},
 * {@code CommitsImpl}, {@code SearchImpl}, {@code AnalysisImpl}, {@code IndexToolsImpl}); this
 * file only prints what they return. Where a model returns a hash map, its entries are printed
 * by key.
 *
 * <p>Read by {@code crates/lucene-tools/tests/luke_fixtures.rs}, which runs the same lines
 * through {@code lucene-tools Luke}.
 */
public class GenToolsLuke {

  private static final String[][] CASES = {
    {"overview", "overview", "{fixture}/index"},
    {"overview-top", "overview", "{fixture}/index", "author", "3", "text", "5", "title", "2"},
    {"overview-top-cached", "overview", "{fixture}/index", "text", "4", "text", "2", "text", "6"},
    {"overview-top-negative", "overview", "{fixture}/index", "text", "-1"},
    {"overview-top-unknown", "overview", "{fixture}/index", "nofield", "3"},
    {"documents", "documents", "{fixture}/index"},
    {"terms-author", "terms", "{fixture}/index", "author"},
    {"terms-notes", "terms", "{fixture}/index", "notes"},
    {"terms-title", "terms", "{fixture}/index", "title"},
    {"terms-seek", "terms", "{fixture}/index", "text", "seek", "sister", "seek", "zzz", "seek", "b"},
    {"terms-none", "terms", "{fixture}/index", "subject", "seek", "x"},
    {"commits", "commits", "{fixture}/index"},
    {"commits-missing", "commit", "{fixture}/index", "99"},
    {"fields", "fields", "{fixture}/index"},
    {"search-classic", "search", "{fixture}/index", "-field", "text", "sister OR alice"},
    {"search-standard", "search", "{fixture}/index", "-standard", "-field", "text", "sister alice"},
    {"search-and", "search", "{fixture}/index", "-and", "-field", "text", "sister alice"},
    {"search-phrase", "search", "{fixture}/index", "-slop", "2", "-field", "author", "\"jane austen\""},
    {"search-rewrite", "search", "{fixture}/index", "-rewrite", "-field", "text", "sis*"},
    {"search-leading-wildcard", "search", "{fixture}/index", "-leading-wildcard", "-field", "text", "*ster"},
    {"search-leading-refused", "search", "{fixture}/index", "-field", "text", "*ster"},
    {"search-syntax", "search", "{fixture}/index", "-field", "text", "(alice"},
    {"search-syntax-standard", "search", "{fixture}/index", "-standard", "-field", "text", "(alice"},
    {
      "search-pages",
      "search",
      "{fixture}/index",
      "-page",
      "2",
      "-field",
      "text",
      "-nav",
      "n,n,n,p,p,p,n",
      "the OR of OR a OR sister"
    },
    {"search-classic-sim", "search", "{fixture}/index", "-classic-sim", "-no-discount", "-field", "text", "sister"},
    {"search-bm25-params", "search", "{fixture}/index", "-k1", "2.0", "-b", "0.5", "-field", "text", "sister"},
    {"search-load", "search", "{fixture}/index", "-load", "title,id", "-field", "text", "sister"},
    {"search-sort-int", "search", "{fixture}/index", "-sort", "year", "INT", "-field", "id", "*:*"},
    {"search-sort-int-rev", "search", "{fixture}/index", "-sort", "year", "INT", "-reverse", "-field", "id", "*:*"},
    {"search-sort-numeric", "search", "{fixture}/index", "-sort", "rating", "LONG", "-field", "id", "*:*"},
    {"search-sort-string", "search", "{fixture}/index", "-sort", "lang", "STRING", "-page", "3", "-nav", "n", "-field", "id", "*:*"},
    {"search-sort-set", "search", "{fixture}/index", "-sort", "subject", "STRING", "-reverse", "-field", "id", "*:*"},
    {"search-sort-none", "search", "{fixture}/index", "-sort", "year", "STRING", "-field", "id", "*:*"},
    {"search-explain", "search", "{fixture}/index", "-explain", "1", "-field", "text", "sister alice"},
    {"search-explain-miss", "search", "{fixture}/index", "-explain", "0", "-field", "text", "sister"},
    {"search-negative-page", "search", "{fixture}/index", "-page", "-1", "-field", "text", "sister"},
    {"search-empty", "search", "{fixture}/index", "-nav", "n,p", "-field", "text", "nomatch"},
    {"mlt", "mlt", "{fixture}/index", "1", "text", "1", "1"},
    {"mlt-two-fields", "mlt", "{fixture}/index", "2", "text,author", "1", "1"},
    {"mlt-defaults", "mlt", "{fixture}/index", "2", "text"},
    {"components", "components"},
    {"analyze-standard", "analyze", "-analyzer", "org.apache.lucene.analysis.standard.StandardAnalyzer", "The Quick-Brown fox's 2 e-mails"},
    {"analyze-whitespace", "analyze", "-analyzer", "org.apache.lucene.analysis.core.WhitespaceAnalyzer", "a  B c"},
    {"analyze-unknown", "analyze", "-analyzer", "org.example.NoSuchAnalyzer", "x"},
    {
      "analyze-custom",
      "analyze",
      "-tokenizer",
      "standard",
      "-cf",
      "htmlStrip",
      "-tf",
      "lowercase",
      "-tf",
      "stop",
      "<b>The</b> Quick brown Fox"
    },
    {
      "analyze-custom-params",
      "analyze",
      "-tokenizer",
      "whitespace",
      "maxTokenLen=4",
      "-cf",
      "patternReplace",
      "pattern=a",
      "replacement=o",
      "-cf",
      "patternReplace",
      "pattern=o+",
      "replacement=0",
      "-tf",
      "lowercase",
      "-tf",
      "keywordRepeat",
      "-tf",
      "porterStem",
      "-tf",
      "removeDuplicates",
      "Cats running abcdefgh"
    },
    {"analyze-custom-bad", "analyze", "-tokenizer", "noSuchTokenizer", "x"},
    {"analyze-step-not-custom", "analyze", "-step", "-analyzer", "org.apache.lucene.analysis.core.WhitespaceAnalyzer", "x"},
    {"export", "export", "{fixture}/index", "author", ",", "{tmp}"},
    {"export-tab", "export", "{fixture}/index", "id", "\\t", "{tmp}"},
    {"export-no-terms", "export", "{fixture}/index", "subject", ",", "{tmp}"},
    {"check", "check", "{fixture}/index"},
    {"copy-optimize", "copy", "{fixture}/index", "{tmp}/opt"},
    {"optimize", "optimize", "{tmp}/opt", "1"},
    {"optimize-overview", "overview", "{tmp}/opt"},
    {"optimize-documents", "documents", "{tmp}/opt"},
    {"copy-expunge", "copy", "{fixture}/index", "{tmp}/exp"},
    {"expunge", "optimize", "{tmp}/exp", "expunge"},
    {"expunge-overview", "overview", "{tmp}/exp"},
    {"copy-delete", "copy", "{fixture}/index", "{tmp}/del"},
    {"delete", "delete", "{tmp}/del", "id", "2"},
    {"delete-overview", "overview", "{tmp}/del"},
    {"delete-documents", "documents", "{tmp}/del"},
    {"copy-add", "copy", "{fixture}/index", "{tmp}/add"},
    {"add", "add", "{tmp}/add", "summary", "A Tale of Two Cities"},
    {"add-overview", "overview", "{tmp}/add"},
    {"add-terms", "terms", "{tmp}/add", "summary"},
    {"create", "create", "{tmp}/news", "{fixture}/news"},
    {"create-overview", "overview", "{tmp}/news", "body", "5", "from", "10"},
    {"create-stored", "stored", "{tmp}/news", "messageId"},
    {"create-again", "create", "{tmp}/news", "{fixture}/news"},
    {"create-empty", "create", "{tmp}/empty"},
    {"create-empty-overview", "overview", "{tmp}/empty"},
  };

  private static final String[][] BOOKS = {
    {
      "1",
      "Pride and Prejudice",
      "Jane Austen",
      "It is a truth universally acknowledged, that a single man in possession of a good fortune,"
          + " must be in want of a wife.",
      "Fiction|Love stories",
      "en",
      "1813",
      "quick|fast brown fox|animal"
    },
    {
      "2",
      "Alice's Adventures in Wonderland",
      "Lewis Carroll",
      "Alice was beginning to get very tired of sitting by her sister on the bank, and of having"
          + " nothing to do: once or twice she had peeped into the book her sister was reading,"
          + " but it had no pictures or conversations in it, 'and what is the use of a book,'"
          + " thought Alice 'without pictures or conversations?'",
      "Fantasy literature",
      "en",
      "1865",
      "rabbit|animal hole"
    },
    {
      "3",
      "Frankenstein; Or, The Modern Prometheus",
      "Mary Wollstonecraft Shelley",
      "You will rejoice to hear that no disaster has accompanied the commencement of an"
          + " enterprise which you have regarded with such evil forebodings. I arrived here"
          + " yesterday, and my first task is to assure my dear sister of my welfare and"
          + " increasing confidence in the success of my undertaking.",
      "Science fiction|Horror tales|Monsters",
      "en",
      "1818",
      ""
    },
    {
      "4",
      "A Doll's House : a play",
      "Henrik Ibsen",
      "",
      "Drama",
      "no",
      "1879",
      "door|exit"
    },
    {
      "5",
      "The Adventures of Sherlock Holmes",
      "Arthur Conan Doyle",
      "To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any"
          + " other name. In his eyes she eclipses and predominates the whole of her sex.",
      "Fiction|Detective and mystery stories",
      "en",
      "1892",
      "pipe violin|music"
    },
    {
      "6",
      "Faust",
      "Johann Wolfgang von Goethe",
      "Ihr naht euch wieder, schwankende Gestalten, die früh sich einst dem trüben Blick"
          + " gezeigt.",
      "Drama|Tragedy",
      "de",
      "1808",
      "devil|myth pact"
    },
  };

  private static final String[][] NEWS = {
    {
      "comp.graphics/38001",
      "From: alice@example.com (Alice Smith)\n"
          + "Subject: Re: Ray tracing on the cheap\n"
          + "Organization: Example University\n"
          + "Lines: 3\n"
          + "Newsgroups: comp.graphics,comp.sys.mac\n"
          + "Message-ID: <1001@example.com>\n"
          + "Date: 5 Apr 93 10:00:00 GMT\n"
          + "\n"
          + "Has anyone tried a ray tracer\n"
          + "on a small machine?\n"
          + "Thanks.\n"
    },
    {
      "sci.space/60001",
      "From: bob@space.example.org\n"
          + "Subject: Orbital mechanics question\n"
          + "Lines: many\n"
          + "Newsgroups: sci.space\n"
          + "Message-ID: <2001@space.example.org>\n"
          + "this header line has no colon\n"
          + "\n"
          + "What is the delta-v to reach low orbit?\n"
    },
    {
      "sci.space/60002",
      "From: carol@example.net (Carol)\n"
          + "Subject: Re: Orbital mechanics question\n"
          + "Organization: None\n"
          + "Lines: 2\n"
          + "Newsgroups: sci.space\n"
          + "Message-ID: <2002@example.net>\n"
          + "Date: 6 Apr 93 12:30:00 GMT\n"
          + "\n"
          + "About 9.4 km/s, counting losses.\n"
          + "See the FAQ: http://example.net/faq\n"
    },
  };

  public static void main(String[] args) throws Exception {
    Path data = Path.of(args[0]).resolve("tools").resolve("luke");
    deleteTree(data);
    Files.createDirectories(data.resolve("out"));
    writeIndex(data.resolve("index"));
    for (String[] m : NEWS) {
      Path file = data.resolve("news").resolve(m[0]);
      Files.createDirectories(file.getParent());
      Files.writeString(file, m[1], StandardCharsets.UTF_8);
    }

    Path tmp = Files.createTempDirectory("gen-tools-luke");
    try (PrintWriter tsv =
        new PrintWriter(
            Files.newBufferedWriter(data.resolve("cases.tsv"), StandardCharsets.UTF_8))) {
      for (String[] c : CASES) {
        tsv.print(String.join("\t", c) + "\n");
        String[] real = new String[c.length - 2];
        for (int i = 2; i < c.length; i++) {
          real[i - 2] =
              c[i].replace("{fixture}", data.toString()).replace("{tmp}", tmp.toString());
        }
        StringBuilder out = new StringBuilder();
        try {
          run(c[1], real, out);
        } catch (Exception e) {
          out.append("error: ")
              .append(e.getClass().getSimpleName())
              .append(": ")
              .append(e.getMessage())
              .append('\n');
        }
        String text =
            out.toString().replace(data.toString(), "{fixture}").replace(tmp.toString(), "{tmp}");
        Files.writeString(data.resolve("out").resolve(c[0] + ".out"), text, StandardCharsets.UTF_8);
      }
    } finally {
      deleteTree(tmp);
    }
  }

  // ------------------------------------------------------------------ the index

  private static Analyzer analyzer() {
    Analyzer notes =
        new Analyzer() {
          @Override
          protected TokenStreamComponents createComponents(String fieldName) {
            Tokenizer t = new WhitespaceTokenizer();
            return new TokenStreamComponents(
                t, new DelimitedPayloadTokenFilter(t, '|', new IdentityEncoder()));
          }
        };
    return new PerFieldAnalyzerWrapper(new StandardAnalyzer(), Map.of("notes", notes));
  }

  private static void writeIndex(Path path) throws IOException {
    FieldType titleType = new FieldType();
    titleType.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    titleType.setStored(true);
    titleType.setTokenized(true);
    titleType.setOmitNorms(true);

    FieldType authorType = new FieldType();
    authorType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    authorType.setStored(true);
    authorType.setTokenized(true);

    FieldType textType = new FieldType();
    textType.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    textType.setTokenized(true);
    textType.setStoreTermVectors(true);

    FieldType notesType = new FieldType();
    notesType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    notesType.setTokenized(true);
    notesType.setStored(true);
    notesType.setStoreTermVectors(true);
    notesType.setStoreTermVectorPositions(true);
    notesType.setStoreTermVectorOffsets(true);
    notesType.setStoreTermVectorPayloads(true);

    FieldType downloadsType = new FieldType();
    downloadsType.setDimensions(1, Integer.BYTES);
    downloadsType.setStored(true);

    IndexWriterConfig config =
        new IndexWriterConfig(analyzer())
            .setIndexDeletionPolicy(NoDeletionPolicy.INSTANCE)
            .setMergePolicy(NoMergePolicy.INSTANCE);
    int[] downloads = {28533, 18712, 14737, 14629, 12828, 9001};
    float[] ratings = {4.5f, 4.25f, 3.75f, 3.5f, 4.75f, 4.0f};
    try (Directory dir = FSDirectory.open(path);
        IndexWriter writer = new IndexWriter(dir, config)) {
      for (int i = 0; i < BOOKS.length; i++) {
        String[] b = BOOKS[i];
        Document doc = new Document();
        doc.add(new StringField("id", b[0], Field.Store.YES));
        doc.add(new Field("title", b[1], titleType));
        doc.add(new Field("author", b[2], authorType));
        doc.add(new Field("text", b[3], textType));
        for (String s : b[4].split("\\|")) {
          doc.add(new SortedSetDocValuesField("subject", new BytesRef(s)));
        }
        doc.add(new SortedDocValuesField("lang", new BytesRef(b[5])));
        doc.add(new IntField("year", Integer.parseInt(b[6]), Field.Store.YES));
        if (!b[7].isEmpty()) {
          doc.add(new Field("notes", b[7], notesType));
        }
        doc.add(new Field("downloads", packInt(downloads[i]), downloadsType));
        doc.add(new NumericDocValuesField("rating", (long) (ratings[i] * 100)));
        doc.add(new StoredField("rating", ratings[i]));
        byte[] isbn = {(byte) 0x97, (byte) 0x80, (byte) i, (byte) (0xF0 + i)};
        doc.add(new BinaryDocValuesField("isbn", new BytesRef(isbn)));
        doc.add(new StoredField("isbn", isbn));
        doc.add(new StoredField("pages", 100L + 37L * i));
        doc.add(new StoredField("weight", 0.5 + i / 4.0));
        if (i != 2) {
          float[] v = {1f + i, 2f - i, 0.5f * i};
          doc.add(new KnnFloatVectorField("vec", v, VectorSimilarityFunction.COSINE));
        }
        writer.addDocument(doc);
        if (i == 2 || i == 4) {
          Map<String, String> userData = new HashMap<>();
          userData.put("step", i == 2 ? "1" : "2");
          userData.put("source", "GenToolsLuke");
          writer.setLiveCommitData(userData.entrySet());
          writer.commit();
        }
      }
      writer.deleteDocuments(new Term("id", "4"));
      Map<String, String> userData = new HashMap<>();
      userData.put("step", "3");
      userData.put("note", "a deletion");
      userData.put("source", "GenToolsLuke");
      writer.setLiveCommitData(userData.entrySet());
      writer.commit();
    }
  }

  private static BytesRef packInt(int value) {
    byte[] dest = new byte[Integer.BYTES];
    NumericUtils.intToSortableBytes(value, dest, 0);
    return new BytesRef(dest);
  }

  // ------------------------------------------------------------------ the commands

  private static void run(String command, String[] args, StringBuilder out) throws Exception {
    switch (command) {
      case "copy":
        copyTree(Path.of(args[0]), Path.of(args[1]));
        return;
      case "components":
        components(out);
        return;
      case "analyze":
        analyze(args, out);
        return;
      case "create":
        create(args, out);
        return;
      default:
    }
    try (Directory dir = FSDirectory.open(Path.of(args[0]));
        DirectoryReader reader = DirectoryReader.open(dir)) {
      switch (command) {
        case "overview" -> overview(reader, args, out);
        case "documents" -> documents(reader, out);
        case "terms" -> terms(reader, args, out);
        case "stored" -> stored(reader, args[1], out);
        case "commits" -> commits(reader, args[0], out);
        case "commit" -> {
          CommitsImpl commits = new CommitsImpl(reader, args[0]);
          long gen = Long.parseLong(args[1]);
          out.append("commit ").append(gen).append(": ").append(present(commits.getCommit(gen)));
          out.append('\n');
          out.append("files: ").append(commits.getFiles(gen).size()).append('\n');
          out.append("segments: ").append(commits.getSegments(gen).size()).append('\n');
          out.append("attributes: ").append(commits.getSegmentAttributes(gen, "_0")).append('\n');
          out.append("diagnostics: ").append(commits.getSegmentDiagnostics(gen, "_0")).append('\n');
          out.append("codec: ").append(present(commits.getSegmentCodec(gen, "_0"))).append('\n');
        }
        case "fields" -> fields(reader, out);
        case "search" -> search(reader, args, out);
        case "mlt" -> mlt(reader, args, out);
        case "export" -> {
          String delimiter = args[2].equals("\\t") ? "\t" : args[2];
          String file =
              new IndexToolsImpl(reader, false, false).exportTerms(args[3], args[1], delimiter);
          String name = Path.of(file).getFileName().toString();
          out.append("file: ").append(name.replaceAll("_[0-9]+\\.out$", "_{ms}.out")).append('\n');
          out.append(Files.readString(Path.of(file), StandardCharsets.UTF_8));
          Files.delete(Path.of(file));
        }
        case "check" -> {
          CheckIndex.Status st = new IndexToolsImpl(reader, false, false).checkIndex(null);
          out.append("clean: ").append(st.clean).append('\n');
          out.append("segments: ").append(st.numSegments).append('\n');
          out.append("bad segments: ").append(st.numBadSegments).append('\n');
          out.append("lost documents: ").append(st.totLoseDocCount).append('\n');
        }
        case "optimize" -> {
          boolean expunge = args[1].equals("expunge");
          int max = expunge ? 1 : Integer.parseInt(args[1]);
          new IndexToolsImpl(reader, false, false).optimize(expunge, max, null);
        }
        case "delete" ->
            new IndexToolsImpl(reader, false, false)
                .deleteDocuments(new TermQuery(new Term(args[1], args[2])));
        case "add" -> {
          Document doc = new Document();
          doc.add(new TextField(args[1], args[2], Field.Store.YES));
          new IndexToolsImpl(reader, false, false).addDocument(doc, new StandardAnalyzer());
        }
        default -> throw new IllegalArgumentException("unknown command " + command);
      }
    }
  }

  private static String present(Optional<?> o) {
    return o.map(String::valueOf).orElse("--");
  }

  private static String sorted(Map<String, ?> map) {
    return new TreeMap<>(map).toString();
  }

  private static void overview(DirectoryReader reader, String[] args, StringBuilder out) {
    OverviewImpl o = new OverviewImpl(reader, args[0]);
    out.append("fields: ").append(o.getNumFields()).append('\n');
    out.append("documents: ").append(o.getNumDocuments()).append('\n');
    out.append("terms: ").append(o.getNumTerms()).append('\n');
    out.append("deletions: ").append(o.hasDeletions()).append('\n');
    out.append("deleted documents: ").append(o.getNumDeletedDocs()).append('\n');
    out.append("optimized: ").append(present(o.isOptimized())).append('\n');
    out.append("version: ").append(present(o.getIndexVersion())).append('\n');
    out.append("format: ").append(present(o.getIndexFormat())).append('\n');
    out.append("commit: ").append(present(o.getCommitDescription())).append('\n');
    out.append("user data: ").append(present(o.getCommitUserData())).append('\n');
    out.append("field names: ").append(o.getFieldNames()).append('\n');
    for (TermCountsOrder order : TermCountsOrder.values()) {
      out.append("term counts ").append(order).append(": ");
      out.append(o.getSortedTermCounts(order)).append('\n');
    }
    for (int i = 1; i + 1 < args.length; i += 2) {
      String field = args[i];
      int n = Integer.parseInt(args[i + 1]);
      out.append("top ").append(field).append(' ').append(n).append(":\n");
      List<TermStats> top = o.getTopTerms(field, n);
      for (TermStats t : top) {
        out.append("  ").append(t).append('\n');
      }
    }
  }

  private static void documents(DirectoryReader reader, StringBuilder out) {
    DocumentsImpl d = new DocumentsImpl(reader);
    out.append("max doc: ").append(d.getMaxDoc()).append('\n');
    out.append("field names: ").append(d.getFieldNames()).append('\n');
    for (int doc = 0; doc < d.getMaxDoc(); doc++) {
      out.append("doc ").append(doc).append(" live=").append(d.isLive(doc)).append('\n');
      for (DocumentField f : d.getDocumentFields(doc)) {
        out.append("  ").append(f);
        out.append(" hasPayloads=").append(f.hasPayloads());
        out.append(" hasNorms=").append(f.hasNorms());
        out.append(" norm=").append(f.getNorm());
        out.append(" string=").append(f.getStringValue());
        out.append(" binary=").append(f.getBinaryValue());
        out.append(" numeric=").append(f.getNumericValue());
        out.append(" pointNumBytes=").append(f.getPointNumBytes());
        out.append(" vectorSimilarity=").append(f.getVectorSimilarity());
        out.append('\n');
      }
      for (String field : d.getFieldNames()) {
        List<TermVectorEntry> tv = d.getTermVectors(doc, field);
        if (!tv.isEmpty()) {
          out.append("  term vector ").append(field).append(":\n");
          for (TermVectorEntry e : tv) {
            out.append("    ").append(e).append('\n');
          }
        }
        Optional<DocValues> dv = d.getDocValues(doc, field);
        if (dv.isPresent()) {
          out.append("  doc values ").append(field).append(": ").append(dv.get()).append('\n');
        }
      }
    }
  }

  private static void terms(DirectoryReader reader, String[] args, StringBuilder out) {
    DocumentsImpl d = new DocumentsImpl(reader);
    Optional<Term> term = d.firstTerm(args[1]);
    out.append("first term: ").append(present(term)).append('\n');
    out.append("current field: ").append(d.getCurrentField()).append('\n');
    while (term.isPresent()) {
      postings(d, out);
      term = d.nextTerm();
      out.append("next term: ").append(present(term)).append('\n');
    }
    out.append("doc freq: ").append(present(d.getDocFreq())).append('\n');
    out.append("first doc: ").append(present(d.firstTermDoc())).append('\n');
    out.append("positions: ").append(d.getTermPositions()).append('\n');
    for (int i = 2; i + 1 < args.length; i += 2) {
      term = d.seekTerm(args[i + 1]);
      out.append("seek ").append(args[i + 1]).append(": ").append(present(term)).append('\n');
      if (term.isPresent()) {
        postings(d, out);
      }
    }
  }

  private static void postings(DocumentsImpl d, StringBuilder out) {
    out.append("  doc freq: ").append(present(d.getDocFreq())).append('\n');
    Optional<Integer> doc = d.firstTermDoc();
    while (doc.isPresent()) {
      out.append("  doc ").append(doc.get()).append(": ");
      out.append(d.getTermPositions().stream().map(String::valueOf).collect(Collectors.joining(", ")));
      out.append('\n');
      doc = d.nextTermDoc();
    }
    out.append("  next doc: ").append(present(d.nextTermDoc())).append('\n');
  }

  private static void stored(DirectoryReader reader, String sortField, StringBuilder out)
      throws IOException {
    DocumentsImpl d = new DocumentsImpl(reader);
    List<List<String>> docs = new ArrayList<>();
    for (int doc = 0; doc < d.getMaxDoc(); doc++) {
      List<String> lines = new ArrayList<>();
      String key = "";
      for (DocumentField f : d.getDocumentFields(doc)) {
        if (f.isStored()) {
          lines.add(f.getName() + "=" + f.getStringValue());
          if (f.getName().equals(sortField)) {
            key = f.getStringValue();
          }
        }
      }
      lines.add(0, key);
      docs.add(lines);
    }
    docs.sort(Comparator.comparing(l -> l.get(0)));
    for (List<String> lines : docs) {
      out.append("doc ").append(lines.get(0)).append('\n');
      for (String line : lines.subList(1, lines.size())) {
        out.append("  ").append(line).append('\n');
      }
    }
  }

  private static void commits(DirectoryReader reader, String path, StringBuilder out) {
    CommitsImpl commits = new CommitsImpl(reader, path);
    for (Commit c : commits.listCommits()) {
      long gen = c.getGeneration();
      out.append("commit ").append(gen);
      out.append(" deleted=").append(c.isDeleted());
      out.append(" segments=").append(c.getSegCount());
      out.append(" user data=").append(c.getUserData()).append('\n');
      out.append("  same commit: ").append(commits.getCommit(gen).isPresent()).append('\n');
      for (var f : commits.getFiles(gen)) {
        out.append("  file ").append(f.getFileName()).append(' ').append(f.getDisplaySize());
        out.append('\n');
      }
      for (Segment s : commits.getSegments(gen)) {
        out.append("  segment ").append(s.getName());
        out.append(" maxDoc=").append(s.getMaxDoc());
        out.append(" delGen=").append(s.getDelGen());
        out.append(" delCount=").append(s.getDelCount());
        out.append(" version=").append(s.getLuceneVer());
        out.append(" codec=").append(s.getCodecName());
        out.append(" size=").append(s.getDisplaySize());
        out.append(" compound=").append(s.isUseCompoundFile()).append('\n');
        out.append("    attributes: ");
        out.append(sorted(commits.getSegmentAttributes(gen, s.getName()))).append('\n');
        Map<String, String> diagnostics =
            new TreeMap<>(commits.getSegmentDiagnostics(gen, s.getName()));
        // what differs between runs and machines
        for (String k : List.of("timestamp", "os", "os.arch", "os.version", "java.vendor",
            "java.runtime.version")) {
          diagnostics.computeIfPresent(k, (key, v) -> "*");
        }
        out.append("    diagnostics: ").append(diagnostics).append('\n');
        out.append("    codec: ");
        out.append(present(commits.getSegmentCodec(gen, s.getName()).map(x -> x.getName())));
        out.append('\n');
      }
    }
  }

  private static void fields(DirectoryReader reader, StringBuilder out) {
    SearchImpl s = new SearchImpl(reader);
    out.append("sortable: ").append(s.getSortableFieldNames()).append('\n');
    out.append("searchable: ").append(s.getSearchableFieldNames()).append('\n');
    out.append("range searchable: ").append(new TreeSet<>(s.getRangeSearchableFieldNames()));
    out.append('\n');
    for (String field : s.getFieldNames()) {
      out.append("sort types ").append(field).append(": ").append(s.guessSortTypes(field));
      out.append('\n');
      for (String type : List.of("INT", "LONG", "FLOAT", "DOUBLE", "STRING", "STRING_VAL", "DOC")) {
        for (boolean reverse : new boolean[] {false, true}) {
          Optional<SortField> sf = s.getSortType(field, type, reverse);
          if (sf.isPresent()) {
            out.append("  ").append(type).append(reverse ? " reverse" : "").append(": ");
            out.append(sf.get()).append('\n');
          }
        }
      }
    }
    try {
      s.guessSortTypes("nofield");
    } catch (Exception e) {
      out.append("nofield: ").append(e.getClass().getSimpleName()).append(": ");
      out.append(e.getMessage()).append('\n');
    }
  }

  private static void search(DirectoryReader reader, String[] args, StringBuilder out) {
    SearchImpl s = new SearchImpl(reader);
    QueryParserConfig.Builder qp = new QueryParserConfig.Builder();
    SimilarityConfig.Builder sim = new SimilarityConfig.Builder();
    String field = "text";
    int pageSize = 10;
    boolean rewrite = false;
    boolean exact = false;
    Set<String> load = null;
    String sortField = null;
    String sortType = null;
    boolean reverse = false;
    String nav = "";
    int explain = -1;
    int i = 1;
    for (; i < args.length - 1; i++) {
      switch (args[i]) {
        case "-field" -> field = args[++i];
        case "-standard" -> qp.useClassicParser(false);
        case "-and" -> qp.defaultOperator(QueryParserConfig.Operator.AND);
        case "-slop" -> qp.phraseSlop(Integer.parseInt(args[++i]));
        case "-leading-wildcard" -> qp.allowLeadingWildcard(true);
        case "-rewrite" -> rewrite = true;
        case "-classic-sim" -> sim.useClassicSimilarity(true);
        case "-no-discount" -> sim.discountOverlaps(false);
        case "-k1" -> sim.k1(Float.parseFloat(args[++i]));
        case "-b" -> sim.b(Float.parseFloat(args[++i]));
        case "-page" -> pageSize = Integer.parseInt(args[++i]);
        case "-exact" -> exact = true;
        case "-load" -> load = Set.of(args[++i].split(","));
        case "-sort" -> {
          sortField = args[++i];
          sortType = args[++i];
        }
        case "-reverse" -> reverse = true;
        case "-nav" -> nav = args[++i];
        case "-explain" -> explain = Integer.parseInt(args[++i]);
        default -> throw new IllegalArgumentException("unknown option " + args[i]);
      }
    }
    String expression = args[i];
    Query query = s.parseQuery(expression, field, new StandardAnalyzer(), qp.build(), rewrite);
    out.append("query: ").append(query).append('\n');
    Sort sort = null;
    if (sortField != null) {
      Optional<SortField> sf = s.getSortType(sortField, sortType, reverse);
      out.append("sort: ").append(present(sf)).append('\n');
      sort = sf.map(Sort::new).orElse(null);
    }
    SearchResults r =
        sort == null
            ? s.search(query, sim.build(), load, pageSize, exact)
            : s.search(query, sim.build(), sort, load, pageSize, exact);
    results("page", r, out);
    for (String step : nav.isEmpty() ? new String[0] : nav.split(",")) {
      Optional<SearchResults> page = step.equals("n") ? s.nextPage() : s.prevPage();
      if (page.isPresent()) {
        results(step.equals("n") ? "next" : "previous", page.get(), out);
      } else {
        out.append(step.equals("n") ? "next" : "previous").append(": --\n");
      }
    }
    out.append("current query: ").append(s.getCurrentQuery()).append('\n');
    if (explain >= 0) {
      out.append("explain ").append(explain).append(":\n").append(s.explain(query, explain));
    }
  }

  private static void results(String label, SearchResults r, StringBuilder out) {
    out.append(label).append(": total=").append(r.getTotalHits());
    out.append(" offset=").append(r.getOffset()).append(" size=").append(r.size()).append('\n');
    for (SearchResults.Doc d : r.getHits()) {
      out.append("  doc=").append(d.getDocId()).append(" score=").append(d.getScore());
      for (var e : new TreeMap<>(d.getFieldValues()).entrySet()) {
        out.append(' ').append(e.getKey()).append('=').append(Arrays.toString(e.getValue()));
      }
      out.append('\n');
    }
  }

  private static void mlt(DirectoryReader reader, String[] args, StringBuilder out) {
    SearchImpl s = new SearchImpl(reader);
    MLTConfig.Builder b = new MLTConfig.Builder().fields(List.of(args[2].split(",")));
    if (args.length > 3) {
      b.minDocFreq(Integer.parseInt(args[3])).minTermFreq(Integer.parseInt(args[4]));
    }
    Query q = s.mltQuery(Integer.parseInt(args[1]), b.build(), new StandardAnalyzer());
    out.append("query: ").append(q).append('\n');
  }

  private static void components(StringBuilder out) {
    AnalysisImpl a = new AnalysisImpl();
    out.append("char filters: ").append(a.getAvailableCharFilters()).append('\n');
    out.append("tokenizers: ").append(a.getAvailableTokenizers()).append('\n');
    out.append("token filters: ").append(a.getAvailableTokenFilters()).append('\n');
  }

  private static Map<String, String> params(String[] args, int from) {
    Map<String, String> params = new LinkedHashMap<>();
    for (int i = from; i < args.length && args[i].contains("=") && !args[i].startsWith("-"); i++) {
      String[] kv = args[i].split("=", 2);
      params.put(kv[0], kv[1]);
    }
    return params;
  }

  private static void analyze(String[] args, StringBuilder out) {
    AnalysisImpl a = new AnalysisImpl();
    String text = args[args.length - 1];
    boolean step = false;
    CustomAnalyzerConfig.Builder custom = null;
    for (int i = 0; i < args.length - 1; i++) {
      switch (args[i]) {
        case "-step" -> step = true;
        case "-analyzer" -> a.createAnalyzerFromClassName(args[++i]);
        case "-tokenizer" -> {
          Map<String, String> p = params(args, i + 2);
          custom = new CustomAnalyzerConfig.Builder(args[++i], p);
          i += p.size();
        }
        case "-cf" -> {
          Map<String, String> p = params(args, i + 2);
          custom.addCharFilterConfig(args[++i], p);
          i += p.size();
        }
        case "-tf" -> {
          Map<String, String> p = params(args, i + 2);
          custom.addTokenFilterConfig(args[++i], p);
          i += p.size();
        }
        default -> throw new IllegalArgumentException("unknown option " + args[i]);
      }
    }
    if (custom != null) {
      a.buildCustomAnalyzer(custom.build());
      step = true;
    }
    out.append("analyzer: ").append(a.currentAnalyzer().getClass().getName()).append('\n');
    tokens(a.analyze(text), "", out);
    if (step) {
      Analysis.StepByStepResult r = a.analyzeStepByStep(text);
      for (Analysis.CharfilteredText t : r.getCharfilteredTexts()) {
        out.append("char filter ").append(t.getName()).append(": ").append(t.getText());
        out.append('\n');
      }
      for (Analysis.NamedTokens t : r.getNamedTokens()) {
        out.append("stage ").append(t.getName()).append(":\n");
        tokens(t.getTokens(), "  ", out);
      }
    }
  }

  /**
   * The tokens, each attribute by its implementation class. Lucene's core attributes outside
   * {@code PackedTokenAttributeImpl} are left out: which of them a chain adds is not something
   * the Rust port, which always holds all of them, can know.
   */
  private static void tokens(List<Analysis.Token> tokens, String indent, StringBuilder out) {
    Set<String> core =
        Set.of(
            "FlagsAttributeImpl",
            "KeywordAttributeImpl",
            "PayloadAttributeImpl",
            "SentenceAttributeImpl",
            "BoostAttributeImpl");
    for (Analysis.Token t : tokens) {
      out.append(indent).append("token ").append(t.getTerm()).append('\n');
      for (Analysis.TokenAttribute att : t.getAttributes()) {
        if (core.contains(att.getAttClass())) {
          continue;
        }
        out.append(indent).append("  ").append(att.getAttClass()).append(' ');
        out.append(sorted(att.getAttValues())).append('\n');
      }
    }
  }

  private static void create(String[] args, StringBuilder out) throws IOException {
    Files.createDirectories(Path.of(args[0]));
    try (Directory dir = FSDirectory.open(Path.of(args[0]))) {
      IndexToolsImpl tools = new IndexToolsImpl(dir, false, false);
      if (args.length > 1) {
        tools.createNewIndex(args[1]);
      } else {
        tools.createNewIndex();
      }
    }
    out.append("created\n");
  }

  // ------------------------------------------------------------------ files

  private static void copyTree(Path from, Path to) throws IOException {
    Files.createDirectories(to);
    try (Stream<Path> files = Files.list(from)) {
      for (Path f : files.toList()) {
        Files.copy(f, to.resolve(f.getFileName()));
      }
    }
  }

  private static void deleteTree(Path root) throws IOException {
    if (!Files.exists(root)) {
      return;
    }
    try (Stream<Path> walk = Files.walk(root)) {
      for (Path p : walk.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(p);
      }
    }
  }
}
