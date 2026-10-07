import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;
import morfologik.fsa.FSA;
import morfologik.fsa.builders.CFSA2Serializer;
import morfologik.fsa.builders.FSA5Serializer;
import morfologik.fsa.builders.FSABuilder;
import morfologik.stemming.Dictionary;
import morfologik.stemming.DictionaryIterator;
import morfologik.stemming.DictionaryLookup;
import morfologik.stemming.EncoderType;
import morfologik.stemming.ISequenceEncoder;
import morfologik.stemming.WordData;
import morfologik.stemming.polish.PolishStemmer;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter;
import org.apache.lucene.analysis.morfologik.MorfologikAnalyzer;
import org.apache.lucene.analysis.morfologik.MorfologikFilter;
import org.apache.lucene.analysis.morfologik.MorphosyntacticTagsAttribute;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.uk.UkrainianMorfologikAnalyzer;

/**
 * M12 T12.3: Lucene's analysis-morfologik module and the Morfologik 2.1.9 lookups it runs.
 *
 * <p>Writes {@code analysis_morfologik/}:
 *
 * <ul>
 *   <li>{@code lookups_polish.tsv}, {@code lookups_ukrainian.tsv}: words through the real dictionaries' {@code
 *       DictionaryLookup} -- {@code word<TAB>stem<TAB>tag<TAB>stem<TAB>tag...} ({@code ~} for
 *       {@code null}) -- for every 400th (Polish) / 600th (Ukrainian) surface form of the
 *       dictionary itself, its upper-cased and capitalised variants, and words that are not in it.
 *   <li>{@code dicts/<name>.dict} + {@code .info}: small dictionaries built here with Morfologik's
 *       own {@code FSABuilder} and serializers over entries written for this project -- every
 *       sequence encoder, FSA5 and CFSA2 (with and without {@code NUMBERS}), UTF-8 and ISO-8859-1,
 *       input/output conversions, tagless entries (whose lookups read a reused buffer's stale tag)
 *       -- and {@code dicts/<name>.tsv}: every probe word through each.
 *   <li>{@code <chain>.tsv}: analyzers and filter chains over {@code corpus/analysis-stempel.txt}
 *       (Polish) and {@code corpus/analysis-ukrainian.txt}: {@link AnalysisRows}' token rows with the
 *       tags appended ({@code TAGS<TAB>line<TAB>tag|tag...} after each token, {@code -} for none).
 *   <li>{@code factory_*.tsv}: {@code morfologik} through {@code CustomAnalyzer}.
 * </ul>
 *
 * Runs with lucene-analysis-morfologik and Morfologik's jars on its own classpath ({@code
 * generator_classpath} in {@code scripts/gen-fixtures.sh}). Deterministic. Read by {@code
 * crates/lucene-analysis-morfologik/tests/morfologik_fixtures.rs}.
 */
public class GenAnalysisMorfologik {

  static String esc(CharSequence s) {
    return s == null ? "~" : AnalysisRows.esc(s.toString());
  }

  static String lookupRow(DictionaryLookup lookup, String w) {
    StringBuilder b = new StringBuilder(AnalysisRows.esc(w));
    try {
      for (WordData wd : lookup.lookup(w)) {
        b.append('\t').append(esc(wd.getStem())).append('\t').append(esc(wd.getTag()));
      }
    } catch (Exception e) {
      b.append("\t!").append(e.getClass().getSimpleName());
    }
    return b.append('\n').toString();
  }

  static List<String> sample(Dictionary d, int every, int limit) {
    List<String> out = new ArrayList<>();
    int n = 0;
    Iterator<WordData> it = new DictionaryIterator(d, d.metadata.getDecoder(), true);
    while (it.hasNext() && out.size() < limit) {
      WordData wd = it.next();
      if (n++ % every == 0) out.add(wd.getWord().toString());
    }
    return out;
  }

  static void lookups(Path file, Dictionary d, List<String> words) throws Exception {
    LinkedHashSet<String> probe = new LinkedHashSet<>();
    for (String w : words) {
      probe.add(w);
      probe.add(w.toUpperCase(Locale.ROOT));
      if (!w.isEmpty()) probe.add(Character.toUpperCase(w.charAt(0)) + w.substring(1));
      probe.add(w + "x");
    }
    probe.addAll(Arrays.asList("", "+", ";", "a+b", "a;b", "123", "😀", "\uD800", "zzzzzz"));
    DictionaryLookup lookup = new DictionaryLookup(d);
    StringBuilder m = new StringBuilder();
    for (String w : probe) m.append(lookupRow(lookup, w));
    Files.writeString(file, m.toString(), StandardCharsets.UTF_8);
  }

  /** Entries {@code surface<TAB>base<TAB>tag} (an empty tag: none). */
  static final String[] ENTRIES = {
    "kotami\tkot\tsubst:pl:inst:m2", "kota\tkot\tsubst:sg:gen:m2+subst:sg:acc:m2", "koty\tkot\tsubst:pl:nom:m2",
    "kot\tkot\tsubst:sg:nom:m2", "psa\tpies\tsubst:sg:gen:m2", "psami\tpies\tsubst:pl:inst:m2",
    "najlepszy\tdobry\tadj:sg:nom:m1:sup", "lepszy\tdobry\tadj:sg:nom:m1:com", "nielepszy\tdobry\tadj:neg",
    "dobra\tdobro\tsubst:pl:nom:n|subst:pl:acc:n", "dobra\tdobry\tadj:sg:nom:f:pos", "zamek\tzamek\t",
    "zamku\tzamek\t", "zamki\tzamek\tsubst:pl", "Łódź\tŁódź\tsubst:sg:nom:f", "łodzi\tłódź\tsubst:sg:gen:f",
    "ab\tab\t", "abc\tb\tx", "xyz\t\ttagonly", "rozmawiać\tmówić\tverb:inf", "przeczytaj\tczytać\timpt",
    "zżółkły\tżółknąć\tppas", "aaaa\ta\tq", "a\taaaa\tq", "colour\tcolor\tsubst+british", "kotek\tkot\t"
  };

  static byte[] entryBytes(String surface, String base, String tag, ISequenceEncoder enc, Charset cs, byte sep) {
    ByteBuffer s = ByteBuffer.wrap(surface.getBytes(cs));
    ByteBuffer t = ByteBuffer.wrap(base.getBytes(cs));
    ByteBuffer e = enc.encode(null, s, t);
    ByteArrayOutputStream b = new ByteArrayOutputStream();
    b.writeBytes(surface.getBytes(cs));
    b.write(sep);
    b.write(e.array(), e.position(), e.remaining());
    b.write(sep);
    b.writeBytes(tag.getBytes(cs));
    return b.toByteArray();
  }

  static void ownDictionary(Path dir, String name, EncoderType type, Charset cs, char sep, String format,
      String extraInfo, List<String> probe) throws Exception {
    ISequenceEncoder enc = type.get();
    List<byte[]> seqs = new ArrayList<>();
    for (String e : ENTRIES) {
      String[] f = e.split("\t", -1);
      if (!cs.newEncoder().canEncode(f[0] + f[1] + f[2])) continue;
      seqs.add(entryBytes(f[0], f[1], f[2], enc, cs, (byte) sep));
    }
    seqs.sort(FSABuilder.LEXICAL_ORDERING);
    List<byte[]> unique = new ArrayList<>();
    for (byte[] s : seqs) if (unique.isEmpty() || !Arrays.equals(unique.get(unique.size() - 1), s)) unique.add(s);
    FSA fsa = FSABuilder.build(unique.toArray(new byte[0][]));
    ByteArrayOutputStream os = new ByteArrayOutputStream();
    switch (format) {
      case "cfsa2" -> new CFSA2Serializer().serialize(fsa, os);
      case "cfsa2n" -> new CFSA2Serializer().withNumbers().serialize(fsa, os);
      case "fsa5" -> new FSA5Serializer().serialize(fsa, os);
      case "fsa5n" -> new FSA5Serializer().withNumbers().serialize(fsa, os);
      default -> throw new AssertionError(format);
    }
    String info = "fsa.dict.separator=" + sep + "\nfsa.dict.encoding=" + cs.name() + "\nfsa.dict.encoder="
        + type.name() + "\n" + extraInfo;
    Files.write(dir.resolve(name + ".dict"), os.toByteArray());
    Files.writeString(dir.resolve(name + ".info"), info, StandardCharsets.UTF_8);
    Dictionary d = Dictionary.read(new ByteArrayInputStream(os.toByteArray()),
        new ByteArrayInputStream(info.getBytes(StandardCharsets.UTF_8)));
    DictionaryLookup lookup = new DictionaryLookup(d);
    StringBuilder m = new StringBuilder();
    for (String w : probe) m.append(lookupRow(lookup, w));
    Files.writeString(dir.resolve(name + ".tsv"), m.toString(), StandardCharsets.UTF_8);
  }

  /** AnalysisRows' rows with each token's tags after it. */
  static String rowsWithTags(Analyzer a, List<String> lines) throws Exception {
    StringBuilder m = new StringBuilder();
    for (int ln = 0; ln < lines.size(); ln++) {
      try (TokenStream ts = a.tokenStream("f", lines.get(ln))) {
        CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
        OffsetAttribute off = ts.addAttribute(OffsetAttribute.class);
        PositionIncrementAttribute inc = ts.addAttribute(PositionIncrementAttribute.class);
        KeywordAttribute kw = ts.addAttribute(KeywordAttribute.class);
        MorphosyntacticTagsAttribute tags = ts.addAttribute(MorphosyntacticTagsAttribute.class);
        ts.reset();
        while (ts.incrementToken()) {
          m.append("T\t").append(ln).append('\t').append(AnalysisRows.esc(term.toString())).append('\t')
              .append(off.startOffset()).append('\t').append(off.endOffset()).append('\t')
              .append(inc.getPositionIncrement()).append('\t').append(kw.isKeyword() ? 1 : 0).append('\t');
          List<StringBuilder> t = tags.getTags();
          if (t == null) {
            m.append('-');
          } else {
            List<String> s = new ArrayList<>();
            for (StringBuilder sb : t) s.add(AnalysisRows.esc(sb.toString()));
            m.append('[').append(String.join("|", s)).append(']');
          }
          m.append('\n');
        }
        ts.end();
        m.append("E\t").append(ln).append('\t').append(off.startOffset()).append('\t').append(off.endOffset())
            .append('\t').append(inc.getPositionIncrement()).append('\n');
      } catch (Exception e) {
        m.append("X\t").append(ln).append('\t').append(e.getClass().getSimpleName()).append('\n');
      }
    }
    return m.toString();
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_morfologik");
    Path dicts = out.resolve("dicts");
    Files.createDirectories(dicts);
    String corpusDir = System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus");

    Dictionary polish = new PolishStemmer().getDictionary();
    lookups(out.resolve("lookups_polish.tsv"), polish, sample(polish, 400, 12000));
    URL ukUrl = UkrainianMorfologikAnalyzer.class.getClassLoader().getResource("ua/net/nlp/ukrainian.dict");
    Dictionary ukrainian = Dictionary.read(ukUrl);
    lookups(out.resolve("lookups_ukrainian.tsv"), ukrainian, sample(ukrainian, 600, 10000));

    List<String> probe = new ArrayList<>();
    for (String e : ENTRIES) probe.add(e.split("\t", -1)[0]);
    probe.addAll(Arrays.asList("", "KOTAMI", "kotam", "kotamix", "x", "zamek", "dobra", "nie", "ŁÓDŹ", "a+b",
        "a;b", "a|b", "colour", "zamki", "zamku", "kotek", "zamek"));
    String[][] own = {
      {"suffix_cfsa2", "SUFFIX", "UTF-8", "+", "cfsa2", ""},
      {"suffix_fsa5", "SUFFIX", "UTF-8", "+", "fsa5", ""},
      {"prefix_cfsa2n", "PREFIX", "UTF-8", ";", "cfsa2n", ""},
      {"prefix_fsa5n", "PREFIX", "UTF-8", "+", "fsa5n", ""},
      {"infix_cfsa2", "INFIX", "UTF-8", "+", "cfsa2", ""},
      {"none_fsa5", "NONE", "UTF-8", "+", "fsa5", ""},
      {"suffix_latin1", "SUFFIX", "ISO-8859-1", "+", "cfsa2", ""},
      {"conversions", "SUFFIX", "UTF-8", "+", "cfsa2",
          "fsa.dict.input-conversion=x kot, colour colour\nfsa.dict.output-conversion=kot KOT\n"},
    };
    for (String[] o : own) {
      ownDictionary(dicts, o[0], EncoderType.valueOf(o[1]), Charset.forName(o[2]), o[3].charAt(0), o[4], o[5], probe);
    }

    List<String> pl = AnalysisRows.corpus("analysis-stempel.txt");
    List<String> uk = AnalysisRows.corpus("analysis-ukrainian.txt");
    Map<String, Object[]> chains = new LinkedHashMap<>();
    chains.put("morfologik_polish", new Object[] {(Supplier<Analyzer>) MorfologikAnalyzer::new, pl});
    chains.put("ukrainian_analyzer", new Object[] {(Supplier<Analyzer>) UkrainianMorfologikAnalyzer::new, uk});
    chains.put("ukrainian_exclusions", new Object[] {(Supplier<Analyzer>) () -> new UkrainianMorfologikAnalyzer(
        CharArraySet.EMPTY_SET, new CharArraySet(List.of("київ", "школі", "дітей"), false)), uk});
    Dictionary suffix = Dictionary.read(dicts.resolve("suffix_cfsa2.dict"));
    chains.put("filter_own_dict", new Object[] {(Supplier<Analyzer>) () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new MorfologikFilter(new SetKeywordMarkerFilter(t, new CharArraySet(List.of("koty"), false)), suffix)),
        AnalysisRows.corpus("analysis-morfologik.txt")});
    for (Map.Entry<String, Object[]> e : chains.entrySet()) {
      @SuppressWarnings("unchecked")
      Supplier<Analyzer> s = (Supplier<Analyzer>) e.getValue()[0];
      @SuppressWarnings("unchecked")
      List<String> lines = (List<String>) e.getValue()[1];
      try (Analyzer a = s.get()) {
        Files.writeString(out.resolve(e.getKey() + ".tsv"), rowsWithTags(a, lines), StandardCharsets.UTF_8);
      }
    }

    String[][] factories = {
      {"factory_default", "tok:standard", "tf:morfologik"},
      {"factory_own", "tok:whitespace", "tf:morfologik", "dictionary=suffix_fsa5.dict"},
      {"factory_missing", "tok:whitespace", "tf:morfologik", "dictionary=missing.dict"},
      {"factory_resource_param", "tok:whitespace", "tf:morfologik", "dictionary-resource=x"},
      {"factory_param", "tok:whitespace", "tf:morfologik", "x=1"},
    };
    for (String[] fields : factories) {
      StringBuilder f = new StringBuilder();
      CustomAnalyzer a;
      try {
        a = GenAnalysisFactories.build(GenAnalysisFactories.parse(fields), dicts);
      } catch (Exception | Error e) {
        f.append("B\t").append(e.getClass().getSimpleName()).append('\n');
        Files.writeString(out.resolve(fields[0] + ".tsv"), f.toString(), StandardCharsets.UTF_8);
        continue;
      }
      try (a) {
        f.append(rowsWithTags(a, AnalysisRows.corpus("analysis-morfologik.txt")));
      }
      Files.writeString(out.resolve(fields[0] + ".tsv"), f.toString(), StandardCharsets.UTF_8);
    }
  }
}
