import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.function.Supplier;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import opennlp.tools.chunker.ChunkSample;
import opennlp.tools.chunker.ChunkerFactory;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.lemmatizer.DictionaryLemmatizer;
import opennlp.tools.lemmatizer.LemmaSample;
import opennlp.tools.lemmatizer.LemmatizerFactory;
import opennlp.tools.lemmatizer.LemmatizerME;
import opennlp.tools.lemmatizer.LemmatizerModel;
import opennlp.tools.postag.POSDictionary;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSSample;
import opennlp.tools.postag.POSTaggerFactory;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorFactory;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.sentdetect.SentenceSample;
import opennlp.tools.tokenize.TokenSample;
import opennlp.tools.tokenize.TokenizerFactory;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.CollectionObjectStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.StringList;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.featuregen.FeatureGeneratorUtil;
import opennlp.tools.util.model.BaseModel;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.opennlp.tools.NLPChunkerOp;
import org.apache.lucene.analysis.opennlp.tools.NLPLemmatizerOp;
import org.apache.lucene.analysis.opennlp.tools.NLPPOSTaggerOp;
import org.apache.lucene.analysis.opennlp.tools.NLPSentenceDetectorOp;
import org.apache.lucene.analysis.opennlp.tools.NLPTokenizerOp;

/**
 * M12 T12.5: Lucene's analysis-opennlp over OpenNLP 2.5.3, compared by {@code
 * crates/lucene-analysis-opennlp/tests/opennlp_fixtures.rs}.
 *
 * <p>The models are trained here, by OpenNLP's own trainers, on a seeded synthetic English corpus
 * (a small grammar over a tagged, lemmatized, chunked vocabulary) -- Lucene's tests do the same
 * with their own tiny models. Each is re-packed with a fixed timestamp and stored (not deflated)
 * entries so that the bytes do not depend on the clock or the JDK's zlib: {@code en-sent.bin}
 * (maxent, with an abbreviation dictionary), {@code en-token.bin} (maxent, alphanumeric
 * optimization, abbreviations), {@code en-pos-maxent.bin} (Penn tags, a tag dictionary), {@code
 * en-pos-ud.bin} (UD tags, which Lucene's PENN request maps back), {@code en-pos-perceptron.bin},
 * {@code en-chunker.bin} (maxent), {@code en-chunker-perceptron.bin}, {@code en-lemmatizer.bin} and
 * {@code en-lemmas.dict} (the dictionary lemmatizer's text file).
 *
 * <ul>
 *   <li>{@code opennlp_text.txt}: the texts, one per line (sentences outside the training data:
 *       unknown words, abbreviations, numbers, several sentences a line).
 *   <li>{@code tools.tsv}: the NLP*Op outputs per line -- sentence spans, token spans per sentence,
 *       POS tags (three models), chunks (two), lemmas (dictionary, model, both).
 *   <li>{@code o_*.tsv}: {@code CustomAnalyzer} chains over the texts (AnalysisRows rows), and
 *       {@code factory_specs.tsv} naming them; a refused configuration is a {@code B} row.
 * </ul>
 */
public class GenAnalysisOpenNLP {
  // word, Penn tag, UD tag, lemma
  static final String[][] DET = {{"the", "DT", "DET", "the"}, {"a", "DT", "DET", "a"},
      {"this", "DT", "DET", "this"}, {"every", "DT", "DET", "every"}};
  static final String[][] ADJ = {{"big", "JJ", "ADJ", "big"}, {"small", "JJ", "ADJ", "small"},
      {"quick", "JJ", "ADJ", "quick"}, {"old", "JJ", "ADJ", "old"}, {"happy", "JJ", "ADJ", "happy"},
      {"bigger", "JJR", "ADJ", "big"}, {"oldest", "JJS", "ADJ", "old"}};
  static final String[][] NOUN_SG = {{"dog", "NN", "NOUN", "dog"}, {"cat", "NN", "NOUN", "cat"},
      {"house", "NN", "NOUN", "house"}, {"city", "NN", "NOUN", "city"},
      {"teacher", "NN", "NOUN", "teacher"}, {"child", "NN", "NOUN", "child"},
      {"car", "NN", "NOUN", "car"}, {"book", "NN", "NOUN", "book"}, {"river", "NN", "NOUN", "river"},
      {"garden", "NN", "NOUN", "garden"}, {"ねこ", "NN", "NOUN", "ねこ"},
      {"テレビ", "NN", "NOUN", "テレビ"}};
  static final String[][] NOUN_PL = {{"dogs", "NNS", "NOUN", "dog"}, {"cats", "NNS", "NOUN", "cat"},
      {"houses", "NNS", "NOUN", "house"}, {"cities", "NNS", "NOUN", "city"},
      {"teachers", "NNS", "NOUN", "teacher"}, {"children", "NNS", "NOUN", "child"},
      {"cars", "NNS", "NOUN", "car"}, {"books", "NNS", "NOUN", "book"}};
  static final String[][] PROPER = {{"John", "NNP", "PROPN", "John"}, {"Mary", "NNP", "PROPN", "Mary"},
      {"London", "NNP", "PROPN", "London"}, {"Paris", "NNP", "PROPN", "Paris"}};
  static final String[][] PRON_SG = {{"he", "PRP", "PRON", "he"}, {"she", "PRP", "PRON", "she"},
      {"it", "PRP", "PRON", "it"}};
  static final String[][] PRON_PL = {{"they", "PRP", "PRON", "they"}, {"we", "PRP", "PRON", "we"}};
  // verb lemma: 3sg, past, base/non-3sg
  static final String[][] VERBS = {{"run", "runs", "ran", "run"}, {"see", "sees", "saw", "see"},
      {"like", "likes", "liked", "like"}, {"read", "reads", "read", "read"},
      {"give", "gives", "gave", "give"}, {"have", "has", "had", "have"}, {"be", "is", "was", "are"},
      {"find", "finds", "found", "find"}};
  static final String[][] MODAL = {{"can", "MD", "AUX", "can"}, {"will", "MD", "AUX", "will"}};
  static final String[][] ADV = {{"quickly", "RB", "ADV", "quickly"}, {"often", "RB", "ADV", "often"},
      {"never", "RB", "ADV", "never"}};
  static final String[][] PREP = {{"in", "IN", "ADP", "in"}, {"on", "IN", "ADP", "on"},
      {"with", "IN", "ADP", "with"}, {"near", "IN", "ADP", "near"}, {"under", "IN", "ADP", "under"}};
  static final String[][] CONJ = {{"and", "CC", "CCONJ", "and"}, {"but", "CC", "CCONJ", "but"}};
  static final String[][] NUM = {{"3", "CD", "NUM", "3"}, {"42", "CD", "NUM", "42"},
      {"1999", "CD", "NUM", "1999"}, {"3/4", "CD", "NUM", "3/4"}, {"12-5", "CD", "NUM", "12-5"},
      {"1,000", "CD", "NUM", "1,000"}, {"3.5", "CD", "NUM", "3.5"}};

  /** One token of a generated sentence. */
  record Tok(String word, String penn, String ud, String lemma, String chunk, boolean glued) {}

  static String[] pick(Random r, String[][] pool) {
    return pool[r.nextInt(pool.length)];
  }

  static void add(List<Tok> s, String[] w, String chunk) {
    s.add(new Tok(w[0], w[1], w[2], w[3], chunk, false));
  }

  /** A noun phrase; returns whether it is plural. */
  static boolean np(Random r, List<Tok> s, boolean capitalize) {
    int kind = r.nextInt(6);
    boolean plural;
    List<Tok> np = new ArrayList<>();
    if (kind == 0) {
      boolean pl = r.nextBoolean();
      add(np, pick(r, pl ? PRON_PL : PRON_SG), "B-NP");
      plural = pl;
    } else if (kind == 1) {
      add(np, pick(r, PROPER), "B-NP");
      plural = false;
    } else if (kind == 2) {
      add(np, pick(r, NUM), "B-NP");
      add(np, pick(r, NOUN_PL), "I-NP");
      plural = true;
    } else {
      plural = r.nextInt(3) == 0;
      if (!plural || r.nextBoolean()) {
        add(np, plural ? new String[] {"the", "DT", "DET", "the"} : pick(r, DET), "B-NP");
      }
      if (r.nextInt(3) == 0) {
        add(np, pick(r, ADJ), np.isEmpty() ? "B-NP" : "I-NP");
      }
      add(np, pick(r, plural ? NOUN_PL : NOUN_SG), np.isEmpty() ? "B-NP" : "I-NP");
    }
    if (capitalize) {
      Tok f = np.get(0);
      if (!f.penn().equals("NNP") && !f.penn().equals("CD")) {
        np.set(0, new Tok(Character.toUpperCase(f.word().charAt(0)) + f.word().substring(1),
            f.penn(), f.ud(), f.lemma(), f.chunk(), false));
      }
    }
    s.addAll(np);
    return plural;
  }

  static void vp(Random r, List<Tok> s, boolean plural) {
    String[] v = pick(r, VERBS);
    int form = r.nextInt(4);
    if (form == 0) {
      add(s, pick(r, MODAL), "B-VP");
      add(s, new String[] {v[0], "VB", v[0].equals("be") ? "AUX" : "VERB", v[0]}, "I-VP");
    } else if (form == 1) {
      add(s, new String[] {v[2], "VBD", v[0].equals("be") ? "AUX" : "VERB", v[0]}, "B-VP");
    } else {
      if (r.nextInt(4) == 0) {
        add(s, pick(r, ADV), "B-ADVP");
      }
      if (plural) {
        add(s, new String[] {v[3], "VBP", v[0].equals("be") ? "AUX" : "VERB", v[0]}, "B-VP");
      } else {
        add(s, new String[] {v[1], "VBZ", v[0].equals("be") ? "AUX" : "VERB", v[0]}, "B-VP");
      }
    }
  }

  static List<Tok> sentence(Random r) {
    List<Tok> s = new ArrayList<>();
    boolean plural = np(r, s, true);
    vp(r, s, plural);
    int shape = r.nextInt(5);
    if (shape == 0 || shape == 1) {
      np(r, s, false);
    }
    if (shape == 1 || shape == 2) {
      add(s, pick(r, PREP), "B-PP");
      np(r, s, false);
    }
    if (shape == 3) {
      add(s, pick(r, ADV), "B-ADVP");
    }
    if (shape == 4) {
      np(r, s, false);
      add(s, new String[] {",", ",", "PUNCT", ","}, "O");
      add(s, pick(r, CONJ), "O");
      boolean pl = np(r, s, false);
      vp(r, s, pl);
    }
    String end = r.nextInt(8) == 0 ? "?" : r.nextInt(8) == 0 ? "!" : ".";
    s.add(new Tok(end, ".", "PUNCT", end, "O", true));
    // commas are glued to the word before them
    for (int i = 0; i < s.size(); i++) {
      Tok t = s.get(i);
      if (t.word().equals(",")) {
        s.set(i, new Tok(",", ",", "PUNCT", ",", "O", true));
      }
    }
    return s;
  }

  static String text(List<Tok> s) {
    StringBuilder b = new StringBuilder();
    for (Tok t : s) {
      if (b.length() > 0 && !t.glued()) b.append(' ');
      b.append(t.word());
    }
    return b.toString();
  }

  static Span[] tokenSpans(List<Tok> s) {
    List<Span> spans = new ArrayList<>();
    int pos = 0;
    for (Tok t : s) {
      if (pos > 0 && !t.glued()) pos++;
      spans.add(new Span(pos, pos + t.word().length()));
      pos += t.word().length();
    }
    return spans.toArray(new Span[0]);
  }

  static TrainingParameters params(String algorithm, int iterations) {
    TrainingParameters p = new TrainingParameters();
    p.put(TrainingParameters.ALGORITHM_PARAM, algorithm);
    p.put(TrainingParameters.ITERATIONS_PARAM, iterations);
    p.put(TrainingParameters.CUTOFF_PARAM, 1);
    p.put(TrainingParameters.THREADS_PARAM, 1);
    return p;
  }

  /** The model's zip, re-packed: stored entries, sorted, epoch timestamps, no clock in the manifest. */
  static void write(BaseModel model, Path file) throws IOException {
    ByteArrayOutputStream raw = new ByteArrayOutputStream();
    model.serialize(raw);
    TreeMap<String, byte[]> entries = new TreeMap<>();
    try (ZipInputStream zin = new ZipInputStream(new ByteArrayInputStream(raw.toByteArray()))) {
      ZipEntry e;
      while ((e = zin.getNextEntry()) != null) {
        byte[] data = zin.readAllBytes();
        if (e.getName().equals("manifest.properties")) {
          StringBuilder m = new StringBuilder();
          for (String line : new String(data, StandardCharsets.ISO_8859_1).split("\n")) {
            if (line.startsWith("#")) continue;
            if (line.startsWith("Timestamp=")) line = "Timestamp=0";
            m.append(line).append('\n');
          }
          String[] lines = m.toString().split("\n");
          Arrays.sort(lines);
          data = (String.join("\n", lines) + "\n").getBytes(StandardCharsets.ISO_8859_1);
        }
        entries.put(e.getName(), data);
      }
    }
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipOutputStream zout = new ZipOutputStream(out)) {
      for (Map.Entry<String, byte[]> e : entries.entrySet()) {
        ZipEntry z = new ZipEntry(e.getKey());
        z.setMethod(ZipEntry.STORED);
        z.setSize(e.getValue().length);
        CRC32 crc = new CRC32();
        crc.update(e.getValue());
        z.setCrc(crc.getValue());
        z.setTime(315532800000L + 12 * 3600 * 1000L); // 1980-01-01 12:00 UTC, within DOS time
        zout.putNextEntry(z);
        zout.write(e.getValue());
        zout.closeEntry();
      }
    }
    Files.write(file, out.toByteArray());
  }

  /**
   * A copy of a written model with one manifest line replaced (unless {@code from} is null) and
   * entry {@code entry} set to {@code data} (removed when null).
   */
  static void repack(Path in, Path outFile, String from, String to, String entry, byte[] data)
      throws IOException {
    TreeMap<String, byte[]> entries = new TreeMap<>();
    try (ZipInputStream zin = new ZipInputStream(Files.newInputStream(in))) {
      ZipEntry e;
      while ((e = zin.getNextEntry()) != null) entries.put(e.getName(), zin.readAllBytes());
    }
    if (from != null) {
      String m = new String(entries.get("manifest.properties"), StandardCharsets.ISO_8859_1);
      entries.put("manifest.properties", m.replace(from, to).getBytes(StandardCharsets.ISO_8859_1));
    }
    if (data == null) entries.remove(entry); else entries.put(entry, data);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipOutputStream zout = new ZipOutputStream(out)) {
      for (Map.Entry<String, byte[]> e : entries.entrySet()) {
        ZipEntry z = new ZipEntry(e.getKey());
        z.setMethod(ZipEntry.STORED);
        z.setSize(e.getValue().length);
        CRC32 crc = new CRC32();
        crc.update(e.getValue());
        z.setCrc(crc.getValue());
        z.setTime(315532800000L + 12 * 3600 * 1000L);
        zout.putNextEntry(z);
        zout.write(e.getValue());
        zout.closeEntry();
      }
    }
    Files.write(outFile, out.toByteArray());
  }

  static String spans(Span[] spans) {
    StringBuilder b = new StringBuilder();
    for (Span s : spans) {
      if (b.length() > 0) b.append(' ');
      b.append(s.getStart()).append('-').append(s.getEnd());
    }
    return b.toString();
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_opennlp");
    Files.createDirectories(out);
    Random r = new Random(0x6f6e6c70L);
    List<List<Tok>> corpus = new ArrayList<>();
    for (int i = 0; i < 700; i++) corpus.add(sentence(r));

    Dictionary abbreviations = new Dictionary(false);
    for (String a : new String[] {"Mr.", "Dr.", "Mrs.", "e.g.", "etc."}) abbreviations.put(new StringList(a));

    // sentences: documents of 1..6 sentences; some open with an abbreviation and a name
    List<SentenceSample> sentSamples = new ArrayList<>();
    for (int i = 0; i + 6 <= corpus.size(); i += 6) {
      StringBuilder doc = new StringBuilder();
      List<Span> sents = new ArrayList<>();
      for (int j = 0; j < 1 + (i / 6) % 6; j++) {
        String s = text(corpus.get(i + j));
        if ((i + j) % 7 == 0) s = (j % 2 == 0 ? "Mr. " : "Dr. ") + s.substring(0, 1).toLowerCase() + s.substring(1);
        if (doc.length() > 0) doc.append(' ');
        sents.add(new Span(doc.length(), doc.length() + s.length()));
        doc.append(s);
      }
      sentSamples.add(new SentenceSample(doc.toString(), sents.toArray(new Span[0])));
    }
    SentenceModel sentModel = SentenceDetectorME.train("eng", new CollectionObjectStream<>(sentSamples),
        new SentenceDetectorFactory("eng", true, abbreviations, null), params("MAXENT", 60));
    write(sentModel, out.resolve("en-sent.bin"));

    List<TokenSample> tokSamples = new ArrayList<>();
    for (List<Tok> s : corpus) tokSamples.add(new TokenSample(text(s), tokenSpans(s)));
    // possessives and contractions split off
    for (String[] pc : new String[][] {{"John's", "John", "'s"}, {"Mary's", "Mary", "'s"},
        {"don't", "do", "n't"}, {"can't", "ca", "n't"}, {"(dogs)", "(", "dogs", ")"}}) {
      for (int k = 0; k < 8; k++) {
        String t = "the " + pc[0] + " book";
        List<Span> sp = new ArrayList<>();
        sp.add(new Span(0, 3));
        int p = 4;
        for (int q = 1; q < pc.length; q++) {
          sp.add(new Span(p, p + pc[q].length()));
          p += pc[q].length();
        }
        sp.add(new Span(p + 1, p + 5));
        tokSamples.add(new TokenSample(t, sp.toArray(new Span[0])));
      }
    }
    TokenizerModel tokModel = TokenizerME.train(new CollectionObjectStream<>(tokSamples),
        new TokenizerFactory("eng", abbreviations, true, null), params("MAXENT", 60));
    write(tokModel, out.resolve("en-token.bin"));
    // the defaults: no end-of-sentence characters, abbreviations or token-end
    // flag; no alphanumeric shortcut
    write(SentenceDetectorME.train("eng", new CollectionObjectStream<>(sentSamples),
        new SentenceDetectorFactory("eng", false, null, null), params("MAXENT", 60)),
        out.resolve("en-sent-default.bin"));
    write(TokenizerME.train(new CollectionObjectStream<>(tokSamples),
        new TokenizerFactory("eng", null, false, null), params("MAXENT", 60)),
        out.resolve("en-token-noopt.bin"));

    List<POSSample> penn = new ArrayList<>();
    List<POSSample> ud = new ArrayList<>();
    List<ChunkSample> chunks = new ArrayList<>();
    List<LemmaSample> lemmas = new ArrayList<>();
    TreeMap<String, String> lemmaDict = new TreeMap<>();
    for (List<Tok> s : corpus) {
      String[] w = s.stream().map(Tok::word).toArray(String[]::new);
      String[] p = s.stream().map(Tok::penn).toArray(String[]::new);
      String[] u = s.stream().map(Tok::ud).toArray(String[]::new);
      String[] c = s.stream().map(Tok::chunk).toArray(String[]::new);
      String[] l = s.stream().map(Tok::lemma).toArray(String[]::new);
      penn.add(new POSSample(w, p));
      ud.add(new POSSample(w, u));
      chunks.add(new ChunkSample(w, p, c));
      lemmas.add(new LemmaSample(w, p, l));
      for (Tok t : s) {
        // the dictionary knows the nouns and verbs, the model the rest
        if (t.penn().startsWith("NN") || t.penn().startsWith("VB")) {
          lemmaDict.put(t.word().toLowerCase() + "\t" + t.penn(), t.lemma());
        }
      }
    }
    POSDictionary tagDict = new POSDictionary(false);
    tagDict.put("the", "DT");
    tagDict.put("read", "VBD", "VBP", "VB");
    tagDict.put("can", "MD");
    POSTaggerFactory posFactory = new POSTaggerFactory(null, null, tagDict);
    write(POSTaggerME.train("eng", new CollectionObjectStream<>(penn), params("MAXENT", 60), posFactory),
        out.resolve("en-pos-maxent.bin"));
    write(POSTaggerME.train("eng", new CollectionObjectStream<>(ud), params("MAXENT", 60),
        new POSTaggerFactory(null, null, null)), out.resolve("en-pos-ud.bin"));
    write(POSTaggerME.train("eng", new CollectionObjectStream<>(penn), params("PERCEPTRON", 40),
        new POSTaggerFactory(null, null, null)), out.resolve("en-pos-perceptron.bin"));
    // an OpenNLP 1.5 POS model (DefaultPOSContextGenerator over an n-gram
    // dictionary) and one without its feature descriptor (the default one)
    Dictionary ngrams = new Dictionary(true);
    for (String w : new String[] {"the", "dog", "John", "runs", "ねこ"}) ngrams.put(new StringList(w));
    ByteArrayOutputStream ngramBytes = new ByteArrayOutputStream();
    ngrams.serialize(ngramBytes);
    repack(out.resolve("en-pos-maxent.bin"), out.resolve("en-pos-v15.bin"), "OpenNLP-Version=2.5.3",
        "OpenNLP-Version=1.5.0", "ngram.dictionary", ngramBytes.toByteArray());
    repack(out.resolve("en-pos-perceptron.bin"), out.resolve("en-pos-nofeat.bin"), null, null,
        "generator.featuregen", null);
    write(ChunkerME.train("eng", new CollectionObjectStream<>(chunks), params("MAXENT", 60),
        new ChunkerFactory()), out.resolve("en-chunker.bin"));
    write(ChunkerME.train("eng", new CollectionObjectStream<>(chunks), params("PERCEPTRON", 40),
        new ChunkerFactory()), out.resolve("en-chunker-perceptron.bin"));
    write(LemmatizerME.train("eng", new CollectionObjectStream<>(lemmas), params("MAXENT", 60),
        new LemmatizerFactory()), out.resolve("en-lemmatizer.bin"));
    StringBuilder dict = new StringBuilder();
    for (Map.Entry<String, String> e : lemmaDict.entrySet()) {
      dict.append(e.getKey()).append('\t').append(e.getValue()).append('\n');
    }
    dict.append("saw\tNN\tsaw#see\n");
    Files.writeString(out.resolve("en-lemmas.dict"), dict.toString(), StandardCharsets.UTF_8);

    // --- the texts the outputs are made over ---------------------------
    List<String> lines = new ArrayList<>(Arrays.asList(
        "The big dog runs.",
        "Mary likes the old house near the river.",
        "The purple elephant reads a newspaper near Berlin.",
        "Mr. Smith saw 42 cats in London. They were happy! Did the child find it?",
        "Dr. Jones gives John's book to the teacher, but the children never read.",
        "We will see the oldest cities and the bigger gardens quickly.",
        "e.g. this is a test etc. and more.",
        "  Leading and trailing spaces.  ",
        "No final punctuation here",
        "",
        "A",
        "?!.",
        "The cars don't run on the river (dogs) can't swim.",
        "Houses, cities, gardens: books!",
        "She has 3 cars. He had 1999 books. It was a cat.",
        "THE DOG RUNS QUICKLY.",
        "the dog. the cat. the house.",
        "Teachers often find small children in old gardens.",
        "Every teacher will give a book to every child.",
        "Running, jumped, happiest, cities, mice, geese.",
        "Paris and London are cities, Paris is old.",
        "The ねこ sees 3/4 of the テレビ. We have 1,000 cars and 3.5 books!",
        "John reads 12-5 books with ねこ and テレビ quickly.",
        "The 😀 dog runs. A 𠀀 cat ran!",
        "The\u00A0dog runs.\u2003It was\u3000old.",
        "&amp; the dog; runs.",
        "Mr.Smith saw A. B. Jones etc.. Then e.g.ok.",
        "He said \"no\" [maybe] {x} (y) 'z' `w` 3! x? y.z"));
    for (int i = 0; i < 40; i++) {
      StringBuilder b = new StringBuilder();
      int n = 1 + r.nextInt(3);
      for (int j = 0; j < n; j++) {
        if (b.length() > 0) b.append(' ');
        b.append(text(sentence(r)));
      }
      lines.add(b.toString());
    }
    StringBuilder longLine = new StringBuilder();
    while (longLine.length() < 2500) {
      if (longLine.length() > 0) longLine.append(' ');
      longLine.append(text(sentence(r)));
    }
    lines.add(longLine.toString());
    Files.writeString(out.resolve("opennlp_text.txt"), String.join("\n", lines) + "\n", StandardCharsets.UTF_8);

    NLPSentenceDetectorOp sentOp = new NLPSentenceDetectorOp(new SentenceModel(out.resolve("en-sent.bin")));
    NLPTokenizerOp tokOp = new NLPTokenizerOp(new TokenizerModel(out.resolve("en-token.bin")));
    NLPSentenceDetectorOp sentOp1 =
        new NLPSentenceDetectorOp(new SentenceModel(out.resolve("en-sent-default.bin")));
    NLPTokenizerOp tokOp1 = new NLPTokenizerOp(new TokenizerModel(out.resolve("en-token-noopt.bin")));
    NLPPOSTaggerOp[] posOps = {
      new NLPPOSTaggerOp(new POSModel(out.resolve("en-pos-maxent.bin"))),
      new NLPPOSTaggerOp(new POSModel(out.resolve("en-pos-ud.bin"))),
      new NLPPOSTaggerOp(new POSModel(out.resolve("en-pos-perceptron.bin"))),
      new NLPPOSTaggerOp(new POSModel(out.resolve("en-pos-v15.bin"))),
      new NLPPOSTaggerOp(new POSModel(out.resolve("en-pos-nofeat.bin")))
    };
    NLPChunkerOp[] chunkOps = {
      new NLPChunkerOp(new ChunkerModel(out.resolve("en-chunker.bin"))),
      new NLPChunkerOp(new ChunkerModel(out.resolve("en-chunker-perceptron.bin")))
    };
    DictionaryLemmatizer dl;
    try (InputStream in = Files.newInputStream(out.resolve("en-lemmas.dict"))) {
      dl = new DictionaryLemmatizer(in);
    }
    LemmatizerModel lm = new LemmatizerModel(out.resolve("en-lemmatizer.bin"));
    NLPLemmatizerOp[] lemOps = {
      new NLPLemmatizerOp(dl, null), new NLPLemmatizerOp(null, lm), new NLPLemmatizerOp(dl, lm)
    };
    StringBuilder tools = new StringBuilder();
    for (int ln = 0; ln < lines.size(); ln++) {
      String line = lines.get(ln);
      Span[] sents = sentOp.splitSentences(line);
      tools.append("S\t").append(ln).append('\t').append(spans(sents)).append('\n');
      tools.append("S1\t").append(ln).append('\t').append(spans(sentOp1.splitSentences(line))).append('\n');
      for (Span s : sents) {
        String sentence = line.substring(s.getStart(), s.getEnd());
        Span[] toks = tokOp.getTerms(sentence);
        tools.append("T\t").append(ln).append('\t').append(spans(toks)).append('\n');
        tools.append("T1\t").append(ln).append('\t').append(spans(tokOp1.getTerms(sentence))).append('\n');
        String[] words = Span.spansToStrings(toks, sentence);
        if (words.length == 0) continue;
        for (int m = 0; m < posOps.length; m++) {
          String[] tags = posOps[m].getPOSTags(words);
          tools.append("P").append(m).append('\t').append(ln).append('\t').append(String.join(" ", tags)).append('\n');
        }
        String[] tags = posOps[0].getPOSTags(words);
        for (int m = 0; m < chunkOps.length; m++) {
          tools.append("C").append(m).append('\t').append(ln).append('\t')
              .append(String.join(" ", chunkOps[m].getChunks(words, tags, null))).append('\n');
        }
        for (int m = 0; m < lemOps.length; m++) {
          tools.append("L").append(m).append('\t').append(ln).append('\t')
              .append(String.join(" ", lemOps[m].lemmatize(words, tags))).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("tools.tsv"), tools.toString(), StandardCharsets.UTF_8);

    // FeatureGeneratorUtil.tokenFeature of hand-picked tokens
    String[] classTokens = {"ひらがな", "カタカナ", "ー", "ひら・がな", "カタ〜カナ", "dog", "Dog", "DOG", "D",
        "D.", "Ä.", "A.\n", "A.\r\n", "A.\u2028", "A.\n\n", "42", "1999", "123", "3/4", "12-5", "1,000",
        "3.5", "A1", "a1b2", "-", "...", "😀", "𠀀", "𐌰", "\uD82C\uDC01", "\uD82C\uDC00", "𐞀", "Ⅻ",
        "ǅ", "ʰ", "漢字", "ひら漢", "café", "naïve", "", "\uD800", "a\uDC00", "١٢", "١٢٣٤", "Straße", "ⓐ",
        "Ⓐ", "ﾃﾚﾋﾞ", "x-1", "Ab.", "a.", "AB.", "Ö.", "ÉA"};
    StringBuilder tc = new StringBuilder();
    for (String t : classTokens) {
      tc.append(AnalysisRows.esc(t)).append('\t').append(FeatureGeneratorUtil.tokenFeature(t)).append('\n');
    }
    Files.writeString(out.resolve("token_classes.tsv"), tc.toString(), StandardCharsets.UTF_8);

    // feature generator descriptors over two sentences, every previous tag given
    String fg = "opennlp.tools.util.featuregen.";
    String[] descriptors = {
      "<featureGenerators name=\"all\">"
          + "<generator class=\"" + fg + "WindowFeatureGeneratorFactory\">"
          + "<int name=\"prevLength\">1</int><int name=\"nextLength\">3</int>"
          + "<generator class=\"" + fg + "TokenFeatureGeneratorFactory\"><bool name=\"lowercase\">false</bool></generator>"
          + "<generator class=\"" + fg + "TokenClassFeatureGeneratorFactory\"><bool name=\"wordAndClass\">false</bool></generator>"
          + "</generator>"
          + "<generator class=\"" + fg + "SuffixFeatureGeneratorFactory\"><int name=\"length\">2</int></generator>"
          + "<generator class=\"" + fg + "PrefixFeatureGeneratorFactory\"><int name=\"length\">6</int></generator>"
          + "<generator class=\"" + fg + "SentenceFeatureGeneratorFactory\"/>"
          + "<generator class=\"" + fg + "TokenClassFeatureGeneratorFactory\"/>"
          + "<generator class=\"" + fg + "DefinitionFeatureGeneratorFactory\"/>"
          + "<generator class=\"" + fg + "PosTaggerFeatureGeneratorFactory\"/>"
          + "<generator class=\"" + fg + "WindowFeatureGeneratorFactory\">"
          + "<int name=\"prevLength\">2</int><int name=\"nextLength\">0</int>"
          + "<generator class=\"" + fg + "PosTaggerFeatureGeneratorFactory\"/>"
          + "</generator>"
          + "</featureGenerators>",
      "<featureGenerators cache=\"true\"><!-- one -->"
          + "<generator class=\"" + fg + "TokenFeatureGeneratorFactory\"/></featureGenerators>",
    };
    String[][] fgSentences = {{"The", "ÄÖÜ", "A.", "12", "ひらがな", "x"}, {"Solo"}};
    StringBuilder fgOut = new StringBuilder();
    for (int d = 0; d < descriptors.length; d++) {
      opennlp.tools.util.featuregen.AdaptiveFeatureGenerator g =
          opennlp.tools.util.featuregen.GeneratorFactory.create(
              new ByteArrayInputStream(descriptors[d].getBytes(StandardCharsets.UTF_8)), key -> null);
      for (int si = 0; si < fgSentences.length; si++) {
        String[] toks = fgSentences[si];
        String[] tags = new String[toks.length];
        for (int i = 0; i < tags.length; i++) tags[i] = "T" + i;
        for (int i = 0; i < toks.length; i++) {
          List<String> feats = new ArrayList<>();
          g.createFeatures(feats, toks, i, tags);
          fgOut.append('D').append(d).append('\t').append(si).append('\t').append(i);
          for (String f : feats) fgOut.append('\t').append(AnalysisRows.esc(f));
          fgOut.append('\n');
        }
      }
    }
    Files.writeString(out.resolve("featuregen.tsv"), fgOut.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("featuregen_descriptors.txt"), String.join("\n", descriptors) + "\n",
        StandardCharsets.UTF_8);

    String tok = "t:openNlp,sentenceModel=en-sent.bin,tokenizerModel=en-token.bin";
    String pos = " f:openNlppos,posTaggerModel=en-pos-maxent.bin";
    String[] specs = {
      tok,
      tok + pos,
      tok + " f:openNlppos,posTaggerModel=en-pos-ud.bin",
      tok + " f:openNlppos,posTaggerModel=en-pos-perceptron.bin",
      tok + pos + " f:openNlpChunker,chunkerModel=en-chunker.bin",
      tok + pos + " f:openNlpChunker,chunkerModel=en-chunker-perceptron.bin",
      tok + pos + " f:openNlpLemmatizer,dictionary=en-lemmas.dict",
      tok + pos + " f:openNlpLemmatizer,lemmatizerModel=en-lemmatizer.bin",
      tok + pos + " f:openNlpLemmatizer,dictionary=en-lemmas.dict,lemmatizerModel=en-lemmatizer.bin",
      tok + pos + " f:keywordRepeat f:openNlpLemmatizer,dictionary=en-lemmas.dict,lemmatizerModel=en-lemmatizer.bin f:removeDuplicates",
      tok + pos + " f:openNlpChunker,chunkerModel=en-chunker.bin f:typeAsPayload",
      tok + pos + " f:lowercase f:openNlpLemmatizer,lemmatizerModel=en-lemmatizer.bin",
      // refused
      "t:openNlp,sentenceModel=en-sent.bin",
      "t:openNlp,tokenizerModel=en-token.bin",
      tok + " f:openNlppos",
      tok + " f:openNlppos,posTaggerModel=missing.bin",
      tok + pos + " f:openNlpChunker,chunkerModel=en-pos-maxent.bin",
      tok + " f:openNlppos,posTaggerModel=en-pos-maxent.bin,bogus=1",
      "t:openNlp,sentenceModel=en-token.bin,tokenizerModel=en-token.bin",
      "f:openNlpLemmatizer",
      // the default and older models
      "t:openNlp,sentenceModel=en-sent-default.bin,tokenizerModel=en-token-noopt.bin"
          + " f:openNlppos,posTaggerModel=en-pos-v15.bin",
      tok + " f:openNlppos,posTaggerModel=en-pos-nofeat.bin f:openNlpLemmatizer,lemmatizerModel=en-lemmatizer.bin",
      // a chunker without a model fails on the first sentence
      tok + pos + " f:openNlpChunker",
    };
    StringBuilder sp = new StringBuilder();
    for (int i = 0; i < specs.length; i++) {
      String name = String.format("o_%02d", i);
      sp.append(name).append('\t').append(specs[i]).append('\n');
      String rows;
      try (Analyzer a = factory(out, specs[i]).get()) {
        rows = AnalysisRows.rows(a, lines);
      } catch (RuntimeException ex) {
        Throwable c = ex.getCause() != null ? ex.getCause() : ex;
        rows = "B\t" + c.getClass().getSimpleName() + "\n";
      }
      Files.writeString(out.resolve(name + ".tsv"), rows, StandardCharsets.UTF_8);
    }
    Files.writeString(out.resolve("factory_specs.tsv"), sp.toString(), StandardCharsets.UTF_8);
  }

  static Supplier<Analyzer> factory(Path dir, String spec) {
    return () -> {
      try {
        CustomAnalyzer.Builder b = CustomAnalyzer.builder(dir);
        for (String part : spec.split(" ")) {
          String[] kv = part.split(":", 2);
          String[] nameArgs = kv[1].split(",");
          List<String> args = new ArrayList<>();
          for (int i = 1; i < nameArgs.length; i++) {
            String[] a = nameArgs[i].split("=", 2);
            args.add(a[0]);
            args.add(a[1]);
          }
          String[] argv = args.toArray(new String[0]);
          switch (kv[0]) {
            case "t" -> b.withTokenizer(nameArgs[0], argv);
            default -> b.addTokenFilter(nameArgs[0], argv);
          }
        }
        return b.build();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    };
  }
}
