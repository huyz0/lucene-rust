import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.cn.smart.HMMChineseTokenizer;
import org.apache.lucene.analysis.cn.smart.SmartChineseAnalyzer;
import org.apache.lucene.analysis.cn.smart.hhmm.SmartcnAccess;
import org.apache.lucene.analysis.custom.CustomAnalyzer;

/**
 * M12 T12.2: smartcn ({@code crates/lucene-analysis-smartcn/tests/smartcn_fixtures.rs} compares).
 *
 * <ul>
 *   <li>{@code <chain>.tsv} ({@link AnalysisRows}' rows) over {@code corpus/analysis-chinese.txt}
 *       (written here) and {@code stress.txt}: 200 seeded lines of common Han characters, Latin,
 *       digits, full-width forms, punctuation, spaces and surrogates, a few past the tokenizer's
 *       1,024-unit window -- {@code HMMChineseTokenizer} ({@code tok_hmm}); over the corpus alone ({@code
 *       c_*}), {@code SmartChineseAnalyzer} with the default, no and a custom stop set, and {@code
 *       hmmChinese} through {@code CustomAnalyzer}.
 *   <li>{@code words.tsv}: {@code WordDictionary} lookups ({@code word getFrequency
 *       getPrefixMatch(0) isEqual}) for every substring of up to four units of the corpus and of
 *       2,000 random Han strings; {@code bigrams.tsv}: {@code BigramDictionary.getFrequency} for
 *       the adjacent words of every corpus segmentation and random pairs.
 *   <li>{@code paths.tsv}: {@code HHMMSegmenter.process} per corpus sentence: each token's text,
 *       offsets, type and frequency.
 * </ul>
 */
public class GenAnalysisSmartcn {

  /** Common Han characters for the stress lines. */
  static final String HAN =
      "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可"
          + "她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三"
          + "已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女"
          + "问力机给等几很业最间新什打便位因重被走电四第门相次东政海口使教西再平真听世气信北少关并内加化由却"
          + "代军产入先山五太水万市眼体别处总才场师书比住员九笑性通目华报立马命张活难神数件安表原车白应路期叫"
          + "死常提感金何更反合放做系计或司利受光王果亲界及今京务制解各任至清物台象记边共风战干接它许八特觉望"
          + "直服毛林题建南度统色字请交爱让认算论百吃义科怎元社术结六功指思非流每青管夫连远资队跟带花快条院变";

  static final String[] OTHER = {
    " ", " ", "　", "\t", "，", "。", "！", "？", "、", "；", "：", "“", "”", "《", "》", "（", "）",
    ",", ".", "!", "?", "-", "abc", "Hello", "XML", "2024", "3.14", "ＡＢＣ", "ｘｙｚ", "１２３", "%",
    "😀", "𠀀", "é", "ß", "αβ", "\r\n", "\n"
  };

  static List<String> stress() {
    Random r = new Random(1212);
    List<String> lines = new ArrayList<>();
    for (int i = 0; i < 200; i++) {
      StringBuilder b = new StringBuilder();
      int len = i % 40 == 39 ? 1100 + r.nextInt(900) : 5 + r.nextInt(60);
      while (b.length() < len) {
        if (r.nextInt(5) == 0) {
          b.append(OTHER[r.nextInt(OTHER.length)]);
        } else {
          int n = 1 + r.nextInt(6);
          for (int k = 0; k < n; k++) b.append(HAN.charAt(r.nextInt(HAN.length())));
        }
      }
      // corpus lines are split on '\n' only
      lines.add(b.toString().replace("\n", " "));
    }
    return lines;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_smartcn");
    Files.createDirectories(out);
    List<String> corpus = AnalysisRows.corpus("analysis-chinese.txt");
    List<String> stress = stress();
    StringBuilder s = new StringBuilder();
    for (String l : stress) s.append(AnalysisRows.esc(l)).append('\n');
    Files.writeString(out.resolve("stress.txt"), s.toString(), StandardCharsets.UTF_8);
    List<String> lines = new ArrayList<>(corpus);
    lines.addAll(stress);

    Map<String, Supplier<Analyzer>> tk = new LinkedHashMap<>();
    tk.put("tok_hmm", () -> AnalysisRows.tok(HMMChineseTokenizer::new));
    AnalysisRows.writeChains(out, tk, lines);
    Map<String, Supplier<Analyzer>> c = new LinkedHashMap<>();
    c.put("c_smart_default", SmartChineseAnalyzer::new);
    c.put("c_smart_nostop", () -> new SmartChineseAnalyzer(false));
    c.put("c_smart_custom_stop", () -> new SmartChineseAnalyzer(new CharArraySet(List.of("的", "了", "是", ","), false)));
    c.put("c_factory_hmm_lower", () -> {
      try {
        return CustomAnalyzer.builder().withTokenizer("hmmChinese").addTokenFilter("lowercase").build();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });
    AnalysisRows.writeChains(out, c, corpus);

    // Dictionary lookups.
    Set<String> words = new LinkedHashSet<>();
    for (String l : corpus) {
      for (int i = 0; i < l.length(); i++) {
        for (int n = 1; n <= 4 && i + n <= l.length(); n++) {
          String w = l.substring(i, i + n);
          if (!Character.isSurrogate(w.charAt(0)) && !Character.isSurrogate(w.charAt(w.length() - 1))) words.add(w);
        }
      }
    }
    Random r = new Random(77);
    for (int i = 0; i < 2000; i++) {
      StringBuilder b = new StringBuilder();
      int n = 1 + r.nextInt(5);
      for (int k = 0; k < n; k++) b.append(HAN.charAt(r.nextInt(HAN.length())));
      words.add(b.toString());
    }
    for (String w : new String[] {"未##串", "未##数", "始##始", "末##末", "，", "。", "！", "a"}) words.add(w);
    StringBuilder ws = new StringBuilder();
    for (String w : words) {
      char[] a = w.toCharArray();
      int prefix = SmartcnAccess.prefixMatch(a);
      ws.append(AnalysisRows.esc(w)).append('\t').append(SmartcnAccess.frequency(a))
          .append('\t').append(prefix).append('\t').append(prefix >= 0 && SmartcnAccess.isEqual(a, prefix) ? 1 : 0)
          .append('\n');
    }
    Files.writeString(out.resolve("words.tsv"), ws.toString(), StandardCharsets.UTF_8);

    StringBuilder paths = new StringBuilder();
    Set<String> pairs = new LinkedHashSet<>();
    for (int ln = 0; ln < corpus.size(); ln++) {
      List<String[]> toks = SmartcnAccess.process(corpus.get(ln));
      for (int i = 0; i < toks.size(); i++) {
        String[] t = toks.get(i);
        paths.append(ln);
        for (String f : t) paths.append('\t').append(AnalysisRows.esc(f));
        paths.append('\n');
        if (i > 0) pairs.add(toks.get(i - 1)[0] + "@" + t[0]);
      }
    }
    Files.writeString(out.resolve("paths.tsv"), paths.toString(), StandardCharsets.UTF_8);
    List<String> list = new ArrayList<>(words);
    for (int i = 0; i < 3000; i++) pairs.add(list.get(r.nextInt(list.size())) + "@" + list.get(r.nextInt(list.size())));
    StringBuilder bs = new StringBuilder();
    for (String p : pairs) bs.append(AnalysisRows.esc(p)).append('\t').append(SmartcnAccess.bigramFrequency(p.toCharArray())).append('\n');
    Files.writeString(out.resolve("bigrams.tsv"), bs.toString(), StandardCharsets.UTF_8);
  }
}
