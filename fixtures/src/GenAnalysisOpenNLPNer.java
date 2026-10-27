import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.namefind.BilouCodec;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSSample;
import opennlp.tools.postag.POSTaggerFactory;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.util.CollectionObjectStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.StringList;
import opennlp.tools.util.featuregen.AdaptiveFeatureGenerator;
import opennlp.tools.util.featuregen.BrownCluster;
import opennlp.tools.util.featuregen.GeneratorFactory;
import opennlp.tools.util.featuregen.WordClusterDictionary;
import org.apache.lucene.analysis.opennlp.tools.NLPNERTaggerOp;

/**
 * M14: Lucene's {@code NLPNERTaggerOp} over OpenNLP 2.5.3's {@code NameFinderME}, compared by
 * {@code crates/lucene-analysis-opennlp/tests/opennlp_ner_fixtures.rs}.
 *
 * <p>The name finder models are trained here by OpenNLP on a seeded synthetic corpus of sentences
 * naming people, places and organizations, and re-packed as {@link GenAnalysisOpenNLP#write} does
 * (stored entries, fixed timestamps), into {@code analysis_opennlp_ner/}:
 *
 * <ul>
 *   <li>{@code ner-maxent.bin}: GIS, BIO, the default feature descriptor (stored in the model);
 *       {@code ner-nofeat.bin} the same without it (OpenNLP's default one is used); {@code
 *       ner-badcodec.bin} naming a codec class that does not exist (BIO is used); {@code
 *       ner-beam1.bin} with beam size 1.
 *   <li>{@code ner-untyped.bin}: perceptron, names without a type ({@code default-start}).
 *   <li>{@code ner-bilou.bin}: perceptron, the BILOU codec and a descriptor naming most other
 *       generators, over resources packed in the model: a dictionary ({@code names.dict}) and a
 *       nested POS model ({@code en-pos-ner.bin}, its bytes as written here). OpenNLP's cluster
 *       serializers close the zip stream, so no package can hold {@code word.clusters} or {@code
 *       brown.clusters}.
 *   <li>{@code ner-seq.bin}: the perceptron sequence trainer.
 *   <li>{@code ner.tsv}: every model's names (span, type, probability bits) and token probabilities
 *       over {@code ner_text.txt} -- documents ({@code #doc}) of space-separated tokenized sentences
 *       ({@code s:}), adaptive data cleared between documents.
 *   <li>{@code ner_featuregen.tsv}: each generator's features over three sentences (the resources
 *       above from a map), adaptive data updated after each and cleared before a repeat.
 * </ul>
 */
public class GenAnalysisOpenNLPNer {
  static final String[][] PERSON = {{"John"}, {"Mary"}, {"Anna", "Lee"}, {"Jean-Luc", "Picard"},
      {"O'Brien"}, {"Smith"}, {"Ilse", "Müller"}, {"Kenji"}};
  static final String[][] PLACE = {{"London"}, {"Paris"}, {"New", "York"}, {"São", "Paulo"},
      {"東京"}, {"Berlin"}};
  static final String[][] ORG = {{"Acme", "Corp"}, {"IBM"}, {"Red", "Cross"}, {"UN"},
      {"Globex", "Inc."}};
  static final String[] VERB = {"visited", "met", "left", "praised", "called", "joined"};
  static final String[] FILLER = {"yesterday", "today", "often", "quietly", "again"};
  static final String[] NOUN = {"report", "office", "meeting", "city", "team", "car"};

  /** A sentence under construction: its tokens and typed names. */
  static final class Sent {
    final List<String> toks = new ArrayList<>();
    final List<String> tags = new ArrayList<>();
    final List<Span> names = new ArrayList<>();

    void word(String w, String tag) {
      toks.add(w);
      tags.add(tag);
    }

    void name(String[] n, String type) {
      int start = toks.size();
      for (String w : n) word(w, "NNP");
      names.add(new Span(start, toks.size(), type));
    }
  }

  static String[] pick(Random r, String[][] pool) {
    return pool[r.nextInt(pool.length)];
  }

  static String pick(Random r, String[] pool) {
    return pool[r.nextInt(pool.length)];
  }

  static Sent sentence(Random r) {
    Sent s = new Sent();
    switch (r.nextInt(6)) {
      case 0 -> {
        s.name(pick(r, PERSON), "person");
        s.word(pick(r, VERB), "VBD");
        s.name(pick(r, PLACE), "location");
        if (r.nextBoolean()) s.word(pick(r, FILLER), "RB");
      }
      case 1 -> {
        s.name(pick(r, PERSON), "person");
        s.word("works", "VBZ");
        s.word("for", "IN");
        s.name(pick(r, ORG), "organization");
        s.word("in", "IN");
        s.name(pick(r, PLACE), "location");
      }
      case 2 -> {
        s.word("The", "DT");
        s.name(pick(r, ORG), "organization");
        s.word(pick(r, VERB), "VBD");
        s.name(pick(r, PERSON), "person");
      }
      case 3 -> {
        s.word(r.nextBoolean() ? "Yesterday" : "Today", "NN");
        s.name(pick(r, PERSON), "person");
        s.word("and", "CC");
        s.name(pick(r, PERSON), "person");
        s.word("met", "VBD");
        s.word("in", "IN");
        s.name(pick(r, PLACE), "location");
      }
      case 4 -> {
        s.word("A", "DT");
        s.word(pick(r, NOUN), "NN");
        s.word("from", "IN");
        s.name(pick(r, ORG), "organization");
        s.word("reached", "VBD");
        s.name(pick(r, PLACE), "location");
        s.word("in", "IN");
        s.word(Integer.toString(1990 + r.nextInt(30)), "CD");
      }
      default -> {
        s.word("the", "DT");
        s.word(pick(r, NOUN), "NN");
        s.word("was", "VBD");
        s.word(pick(r, FILLER), "RB");
        s.word("empty", "JJ");
      }
    }
    s.word(".", ".");
    return s;
  }

  static NameSample sample(Sent s, boolean typed, boolean clear) {
    Span[] names = s.names.stream()
        .map(n -> typed ? n : new Span(n.getStart(), n.getEnd()))
        .toArray(Span[]::new);
    return new NameSample(s.toks.toArray(new String[0]), names, clear);
  }

  /** A model's zip with entry {@code entry} replaced by {@code data}. */
  static void replaceEntry(Path file, String entry, byte[] data) throws IOException {
    TreeMap<String, byte[]> entries = new TreeMap<>();
    try (ZipInputStream zin = new ZipInputStream(Files.newInputStream(file))) {
      ZipEntry e;
      while ((e = zin.getNextEntry()) != null) entries.put(e.getName(), zin.readAllBytes());
    }
    entries.put(entry, data);
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
    Files.write(file, out.toByteArray());
  }

  static String hex(double d) {
    return Long.toHexString(Double.doubleToLongBits(d));
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_opennlp_ner");
    Files.createDirectories(out);
    Random r = new Random(0x6e65724cL);
    List<Sent> corpus = new ArrayList<>();
    for (int i = 0; i < 600; i++) corpus.add(sentence(r));

    // a POS model over the same sentences, nested in the BILOU model
    List<POSSample> pos = new ArrayList<>();
    for (Sent s : corpus) {
      pos.add(new POSSample(s.toks.toArray(new String[0]), s.tags.toArray(new String[0])));
    }
    POSModel posModel = POSTaggerME.train("eng", new CollectionObjectStream<>(pos),
        GenAnalysisOpenNLP.params("MAXENT", 40), new POSTaggerFactory(null, null, null));
    GenAnalysisOpenNLP.write(posModel, out.resolve("en-pos-ner.bin"));
    byte[] posBytes = Files.readAllBytes(out.resolve("en-pos-ner.bin"));
    posModel = new POSModel(new ByteArrayInputStream(posBytes));

    List<NameSample> typed = new ArrayList<>();
    List<NameSample> untyped = new ArrayList<>();
    for (int i = 0; i < corpus.size(); i++) {
      typed.add(sample(corpus.get(i), true, i % 10 == 0));
      untyped.add(sample(corpus.get(i), false, i % 10 == 0));
    }

    TokenNameFinderModel maxent = NameFinderME.train("eng", null,
        new CollectionObjectStream<>(typed), GenAnalysisOpenNLP.params("MAXENT", 80),
        new TokenNameFinderFactory());
    GenAnalysisOpenNLP.write(maxent, out.resolve("ner-maxent.bin"));
    GenAnalysisOpenNLP.repack(out.resolve("ner-maxent.bin"), out.resolve("ner-nofeat.bin"), null,
        null, "generator.featuregen", null);
    GenAnalysisOpenNLP.repack(out.resolve("ner-maxent.bin"), out.resolve("ner-badcodec.bin"),
        "sequenceCodecImplName=opennlp.tools.namefind.BioCodec",
        "sequenceCodecImplName=opennlp.tools.namefind.NoSuchCodec", "generator.featuregen",
        entry(out.resolve("ner-maxent.bin"), "generator.featuregen"));
    GenAnalysisOpenNLP.repack(out.resolve("ner-maxent.bin"), out.resolve("ner-beam1.bin"),
        "BeamSize=3", "BeamSize=1", "generator.featuregen",
        entry(out.resolve("ner-maxent.bin"), "generator.featuregen"));

    GenAnalysisOpenNLP.write(NameFinderME.train("eng", null,
        new CollectionObjectStream<>(untyped), GenAnalysisOpenNLP.params("PERCEPTRON", 40),
        new TokenNameFinderFactory()), out.resolve("ner-untyped.bin"));
    GenAnalysisOpenNLP.write(NameFinderME.train("eng", null,
        new CollectionObjectStream<>(typed), GenAnalysisOpenNLP.params("PERCEPTRON_SEQUENCE", 20),
        new TokenNameFinderFactory()), out.resolve("ner-seq.bin"));

    // the BILOU model: every other generator, over packed resources
    Dictionary dict = new Dictionary(false);
    for (String[] e : new String[][] {{"New", "York"}, {"John", "Smith"}, {"Paris"},
        {"Acme", "Corp"}, {"red", "cross", "society"}}) {
      dict.put(new StringList(e));
    }
    String words = "john c12\nlondon c7 0.5\nnew c3\nyork c3\nibm c9\n東京 c7\nvisited c1\n";
    String brown = "0110\tJohn\t10\n011011101\tMary\t7\n0111001011010101\tLondon\t100\n"
        + "1\tParis\t3\n0101011\tvisited\t20\n110010\tIBM\t9\n10\tthe\n1111111111111111111111\tBerlin\t50\n";
    ByteArrayOutputStream dictBytes = new ByteArrayOutputStream();
    dict.serialize(dictBytes);
    Files.write(out.resolve("names.dict"), dictBytes.toByteArray());
    Files.writeString(out.resolve("word.clusters"), words, StandardCharsets.UTF_8);
    Files.writeString(out.resolve("brown.clusters"), brown, StandardCharsets.UTF_8);
    // OpenNLP 2.5.3's cluster serializers close the zip stream they are
    // handed, so a model package can hold neither (it neither writes nor
    // loads): the BILOU model packs the dictionary and the POS model, the
    // generator rows below take the clusters from a map
    Map<String, Object> resources = new HashMap<>();
    resources.put("names.dict", dict);
    resources.put("pos.bin", posModel);
    Map<String, Object> allResources = new HashMap<>(resources);
    allResources.put("word.clusters",
        new WordClusterDictionary(new ByteArrayInputStream(words.getBytes(StandardCharsets.UTF_8))));
    allResources.put("brown.clusters",
        new BrownCluster(new ByteArrayInputStream(brown.getBytes(StandardCharsets.UTF_8))));
    String fg = "opennlp.tools.util.featuregen.";
    String bilouDescriptor = "<featureGenerators cache=\"true\" name=\"all\">"
        + "<generator class=\"" + fg + "WindowFeatureGeneratorFactory\">"
        + "<int name=\"prevLength\">1</int><int name=\"nextLength\">1</int>"
        + "<generator class=\"" + fg + "TokenClassFeatureGeneratorFactory\"/>"
        + "<generator class=\"" + fg + "TokenFeatureGeneratorFactory\"/>"
        + "</generator>"
        + "<generator class=\"" + fg + "DefinitionFeatureGeneratorFactory\"/>"
        + "<generator class=\"" + fg + "PreviousMapFeatureGeneratorFactory\"/>"
        + "<generator class=\"" + fg + "BigramNameFeatureGeneratorFactory\"/>"
        + "<generator class=\"" + fg + "TrigramNameFeatureGeneratorFactory\"/>"
        + "<generator class=\"" + fg + "DocumentBeginFeatureGeneratorFactory\"/>"
        + "<generator class=\"" + fg + "CharacterNgramFeatureGeneratorFactory\">"
        + "<int name=\"min\">2</int><int name=\"max\">3</int></generator>"
        + "<generator class=\"" + fg + "TokenPatternFeatureGeneratorFactory\"/>"
        + "<generator class=\"" + fg + "DictionaryFeatureGeneratorFactory\">"
        + "<str name=\"dict\">names.dict</str></generator>"
        + "<generator class=\"" + fg + "POSTaggerNameFeatureGeneratorFactory\">"
        + "<str name=\"model\">pos.bin</str></generator>"
        + "<generator class=\"" + fg + "PrefixFeatureGeneratorFactory\"/>"
        + "<generator class=\"" + fg + "SuffixFeatureGeneratorFactory\"><int name=\"length\">3</int></generator>"
        + "<generator class=\"" + fg + "SentenceFeatureGeneratorFactory\"/>"
        + "<generator class=\"" + fg + "AggregatedFeatureGeneratorFactory\">"
        + "<generator class=\"" + fg + "PosTaggerFeatureGeneratorFactory\"/>"
        + "</generator>"
        + "</featureGenerators>";
    TokenNameFinderModel bilou = NameFinderME.train("eng", null,
        new CollectionObjectStream<>(typed), GenAnalysisOpenNLP.params("PERCEPTRON", 40),
        new TokenNameFinderFactory(bilouDescriptor.getBytes(StandardCharsets.UTF_8), resources,
            new BilouCodec()));
    GenAnalysisOpenNLP.write(bilou, out.resolve("ner-bilou.bin"));
    // the nested POS model as written above, not as re-serialized (with the clock)
    replaceEntry(out.resolve("ner-bilou.bin"), "pos.bin", posBytes);

    // --- the texts: documents of tokenized sentences ----------------------
    List<List<String[]>> docs = new ArrayList<>();
    docs.add(List.of(
        "John visited London yesterday .".split(" "),
        "Mary works for IBM in Paris .".split(" "),
        "John met Mary in London .".split(" "),
        "London .".split(" ")));
    docs.add(List.of(
        "Yesterday Ilse Müller and Kenji met in 東京 .".split(" "),
        "The Red Cross praised Jean-Luc Picard".split(" "),
        "A car from Globex Inc. reached São Paulo in 2003 .".split(" "),
        new String[0],
        new String[] {"Smith"},
        "Zork Blarg visited Qux Quux and Wibble met Flob in Narnia .".split(" ")));
    docs.add(List.of(
        "JOHN VISITED LONDON .".split(" "),
        "john visited london .".split(" "),
        "New York New York Acme Corp John Smith Paris Paris .".split(" "),
        "the report was quietly empty .".split(" "),
        "O'Brien called O'Brien again and again .".split(" "),
        "Anna Lee joined the UN team in Berlin in 1999 , then left Berlin for Paris .".split(" ")));
    for (int d = 0; d < 6; d++) {
      List<String[]> doc = new ArrayList<>();
      int n = 1 + r.nextInt(5);
      for (int i = 0; i < n; i++) doc.add(sentence(r).toks.toArray(new String[0]));
      docs.add(doc);
    }
    StringBuilder text = new StringBuilder();
    for (List<String[]> doc : docs) {
      text.append("#doc\n");
      for (String[] s : doc) text.append("s:").append(String.join(" ", s)).append('\n');
    }
    Files.writeString(out.resolve("ner_text.txt"), text.toString(), StandardCharsets.UTF_8);

    String[] models = {"ner-maxent.bin", "ner-nofeat.bin", "ner-badcodec.bin", "ner-beam1.bin",
        "ner-untyped.bin", "ner-bilou.bin", "ner-seq.bin"};
    StringBuilder ner = new StringBuilder();
    for (int m = 0; m < models.length; m++) {
      TokenNameFinderModel model = new TokenNameFinderModel(out.resolve(models[m]));
      NLPNERTaggerOp op = new NLPNERTaggerOp(model);
      NameFinderME me = new NameFinderME(model);
      for (int d = 0; d < docs.size(); d++) {
        for (int si = 0; si < docs.get(d).size(); si++) {
          String[] sentence = docs.get(d).get(si);
          Span[] names = op.getNames(sentence.clone());
          ner.append('N').append(m).append('\t').append(d).append('\t').append(si).append('\t');
          for (int k = 0; k < names.length; k++) {
            Span s = names[k];
            if (k > 0) ner.append(' ');
            ner.append(s.getStart()).append('-').append(s.getEnd()).append(':').append(s.getType())
                .append(':').append(hex(s.getProb()));
          }
          ner.append('\n');
          // the same through NameFinderME, with each token's probability
          me.find(sentence.clone());
          ner.append('Q').append(m).append('\t').append(d).append('\t').append(si).append('\t');
          double[] probs = me.probs();
          for (int k = 0; k < probs.length; k++) {
            if (k > 0) ner.append(' ');
            ner.append(hex(probs[k]));
          }
          ner.append('\n');
        }
        op.reset();
        me.clearAdaptiveData();
      }
    }
    Files.writeString(out.resolve("ner.tsv"), ner.toString(), StandardCharsets.UTF_8);

    // --- each generator alone, over the resources -------------------------
    String[] gens = {
      "<generator class=\"" + fg + "BigramNameFeatureGeneratorFactory\"/>",
      "<generator class=\"" + fg + "TrigramNameFeatureGeneratorFactory\"/>",
      "<generator class=\"" + fg + "PreviousMapFeatureGeneratorFactory\"/>",
      "<generator class=\"" + fg + "DocumentBeginFeatureGeneratorFactory\"/>",
      "<generator class=\"" + fg + "CharacterNgramFeatureGeneratorFactory\">"
          + "<int name=\"min\">2</int><int name=\"max\">4</int></generator>",
      "<generator class=\"" + fg + "CharacterNgramFeatureGeneratorFactory\">"
          + "<int name=\"min\">0</int><int name=\"max\">1</int></generator>",
      "<generator class=\"" + fg + "CharacterNgramFeatureGeneratorFactory\">"
          + "<int name=\"min\">3</int><int name=\"max\">2147483647</int></generator>",
      "<generator class=\"" + fg + "TokenPatternFeatureGeneratorFactory\"/>",
      "<generator class=\"" + fg + "DictionaryFeatureGeneratorFactory\">"
          + "<str name=\"dict\">names.dict</str></generator>",
      "<generator class=\"" + fg + "WordClusterFeatureGeneratorFactory\">"
          + "<str name=\"dict\">word.clusters</str><bool name=\"lowerCase\">true</bool></generator>",
      "<generator class=\"" + fg + "WordClusterFeatureGeneratorFactory\">"
          + "<str name=\"dict\">word.clusters</str><bool name=\"lowerCase\">false</bool></generator>",
      "<generator class=\"" + fg + "BrownClusterTokenFeatureGeneratorFactory\">"
          + "<str name=\"dict\">brown.clusters</str></generator>",
      "<generator class=\"" + fg + "BrownClusterTokenClassFeatureGeneratorFactory\">"
          + "<str name=\"dict\">brown.clusters</str></generator>",
      "<generator class=\"" + fg + "BrownClusterBigramFeatureGeneratorFactory\">"
          + "<str name=\"dict\">brown.clusters</str></generator>",
      "<generator class=\"" + fg + "POSTaggerNameFeatureGeneratorFactory\">"
          + "<str name=\"model\">pos.bin</str></generator>",
      "<generator class=\"" + fg + "AggregatedFeatureGeneratorFactory\">"
          + "<generator class=\"" + fg + "DefinitionFeatureGeneratorFactory\"/>"
          + "<int name=\"x\">1</int>"
          + "<generator class=\"" + fg + "TokenFeatureGeneratorFactory\"/>"
          + "</generator>",
      "<generator class=\"" + fg + "WindowFeatureGeneratorFactory\">"
          + "<int name=\"prevLength\">1</int><int name=\"nextLength\">1</int>"
          + "<generator class=\"" + fg + "CachedFeatureGeneratorFactory\">"
          + "<generator class=\"" + fg + "PreviousMapFeatureGeneratorFactory\"/>"
          + "</generator></generator>",
    };
    String[][] fgSentences = {
      {"John", "Smith", "visited", "New", "York", "in", "1999", "."},
      {"Jean-Luc", "O'Brien", "e-mail:", "x@y.com", "3.5kg", "ÉCOLE", "東京", "naïve", "Ⅻ", "áb",
          "  ", "", "Red", "Cross", "Society"},
      {"London"},
    };
    StringBuilder fgOut = new StringBuilder();
    StringBuilder fgDescriptors = new StringBuilder();
    for (int d = 0; d < gens.length; d++) {
      String descriptor = "<featureGenerators>" + gens[d] + "</featureGenerators>";
      fgDescriptors.append(descriptor).append('\n');
      AdaptiveFeatureGenerator g = GeneratorFactory.create(
          new ByteArrayInputStream(descriptor.getBytes(StandardCharsets.UTF_8)),
          allResources::get);
      // the three sentences, then the first again after clearing the adaptive data
      int[] order = {0, 1, 2, -1, 0};
      for (int pass = 0; pass < order.length; pass++) {
        if (order[pass] < 0) {
          g.clearAdaptiveData();
          continue;
        }
        String[] toks = fgSentences[order[pass]].clone();
        String[] tags = new String[toks.length];
        String[] outcomes = new String[toks.length];
        for (int i = 0; i < tags.length; i++) {
          tags[i] = "T" + i;
          outcomes[i] = "O" + pass + "-" + i;
        }
        for (int i = 0; i < toks.length; i++) {
          List<String> feats = new ArrayList<>();
          g.createFeatures(feats, toks, i, tags);
          fgOut.append('G').append(d).append('\t').append(pass).append('\t').append(i);
          for (String f : feats) fgOut.append('\t').append(AnalysisRows.esc(f));
          fgOut.append('\n');
        }
        g.updateAdaptiveData(toks, outcomes);
      }
    }
    Files.writeString(out.resolve("ner_featuregen.tsv"), fgOut.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("ner_featuregen_descriptors.txt"), fgDescriptors.toString(),
        StandardCharsets.UTF_8);
  }

  /** One entry of a model package. */
  static byte[] entry(Path file, String name) throws IOException {
    try (ZipInputStream zin = new ZipInputStream(Files.newInputStream(file))) {
      ZipEntry e;
      while ((e = zin.getNextEntry()) != null) {
        if (e.getName().equals(name)) return zin.readAllBytes();
      }
    }
    throw new IOException("no " + name + " in " + file);
  }
}
