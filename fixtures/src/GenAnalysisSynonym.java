import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.ParseException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.core.FlattenGraphFilter;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.synonym.SolrSynonymParser;
import org.apache.lucene.analysis.synonym.SynonymFilter;
import org.apache.lucene.analysis.synonym.SynonymGraphFilter;
import org.apache.lucene.analysis.synonym.SynonymMap;
import org.apache.lucene.analysis.synonym.WordnetSynonymParser;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.CharsRefBuilder;
import org.apache.lucene.util.fst.IntsRefFSTEnum;

/**
 * M11 T11.5: synonyms. Parses {@code fixtures/corpus/synonyms-solr.txt} and {@code
 * synonyms-wordnet.txt} (rules written for this project) into {@link SynonymMap}s, dumps each map
 * ({@code maps.tsv}), records what the parsers make of small rule texts, bad ones included ({@code
 * parse.tsv}), and runs {@code SynonymGraphFilter}/{@code SynonymFilter} chains over {@code
 * fixtures/corpus/analysis-synonym.txt} ({@code <chain>.tsv}, {@link AnalysisRows}' rows). {@code
 * crates/lucene-analysis/tests/analysis_synonym_fixtures.rs} compares.
 */
public class GenAnalysisSynonym {

  static Analyzer wsLower() {
    return AnalysisRows.chain(WhitespaceTokenizer::new, LowerCaseFilter::new);
  }

  static Analyzer wsPlain() {
    return AnalysisRows.tok(WhitespaceTokenizer::new);
  }

  static Analyzer stdLower() {
    return AnalysisRows.chain(StandardTokenizer::new, LowerCaseFilter::new);
  }

  static SynonymMap solr(String rules, boolean dedup, boolean expand, Analyzer a) throws Exception {
    SolrSynonymParser p = new SolrSynonymParser(dedup, expand, a);
    p.parse(new StringReader(rules));
    return p.build();
  }

  static SynonymMap wordnet(String rules, boolean dedup, boolean expand, Analyzer a) throws Exception {
    WordnetSynonymParser p = new WordnetSynonymParser(dedup, expand, a);
    p.parse(new StringReader(rules));
    return p.build();
  }

  /** "key\tkeepOrig\tout|out..." per FST entry in order, after a "#\tmaxHorizontalContext\twords" row. */
  static String dump(SynonymMap m) throws Exception {
    StringBuilder b = new StringBuilder();
    b.append("#\t").append(m.maxHorizontalContext).append('\t').append(m.words.size()).append('\n');
    IntsRefFSTEnum<BytesRef> e = new IntsRefFSTEnum<>(m.fst);
    IntsRefFSTEnum.InputOutput<BytesRef> io;
    BytesRef scratch = new BytesRef();
    CharsRefBuilder chars = new CharsRefBuilder();
    while ((io = e.next()) != null) {
      StringBuilder key = new StringBuilder();
      for (int i = 0; i < io.input.length; i++) key.appendCodePoint(io.input.ints[io.input.offset + i]);
      org.apache.lucene.store.ByteArrayDataInput in =
          new org.apache.lucene.store.ByteArrayDataInput(io.output.bytes, io.output.offset, io.output.length);
      int code = in.readVInt();
      b.append(AnalysisRows.esc(key.toString())).append('\t').append((code & 1) == 0 ? 1 : 0);
      for (int i = 0; i < code >>> 1; i++) {
        m.words.get(in.readVInt(), scratch);
        chars.copyUTF8Bytes(scratch);
        b.append(i == 0 ? '\t' : '|').append(AnalysisRows.esc(chars.toString()));
      }
      b.append('\n');
    }
    return b.toString();
  }

  /** Small rule texts, bad ones included: "format\tdedup\texpand\trules(escaped)" -> result rows. */
  static final String[][] PARSE_CASES = {
    {"solr", "a, b, c"},
    {"solr", "a => b\na => c\na => b"},
    {"solr", "a => b => c"},
    {"solr", "\n# only a comment\n\n"},
    {"solr", "a, \\\\"},
    {"solr", "a b, c\r\nd => e f\rg, h"},
    {"solr", "the, a"},
    {"solr", "x,\n,y\n , "},
    {"solr", "trailing\\"},
    {"solr", "a\\=\\>b => c"},
    {"wordnet", "s(100000001,1,'a',n,1,0).\ns(100000001,2,'b c',n,1,0)."},
    {"wordnet", "s(1"},
    {"wordnet", "s(100000001,1,a,n,1,0)."},
    {"wordnet", "s(100000001,1,'a',n,1,0).\ns(100000001,2,'the',n,1,0)."},
    {"wordnet", "s(100000001,1,'it''s',n,1,0).\ns(100000001,2,'x',n,1,0).\ns(100000002,1,'y',n,1,0)."},
    {"wordnet", ""},
    {"wordnet", "s(100000001,1,'a',n,1,0).\n\ns(100000001,2,'b',n,1,0)."},
  };

  static String parseCases() throws Exception {
    StringBuilder b = new StringBuilder();
    // The rules analyzer drops "the" (a stop word), so a phrase can vanish or hold a hole.
    Supplier<Analyzer> analyzer =
        () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new StopFilter(new LowerCaseFilter(t), AnalysisRows.set(false, "the")));
    for (String[] c : PARSE_CASES) {
      for (boolean dedup : new boolean[] {true, false}) {
        for (boolean expand : new boolean[] {true, false}) {
          b.append("C\t").append(c[0]).append('\t').append(dedup ? 1 : 0).append('\t').append(expand ? 1 : 0)
              .append('\t').append(AnalysisRows.esc(c[1])).append('\n');
          try (Analyzer a = analyzer.get()) {
            SynonymMap m = c[0].equals("solr") ? solr(c[1], dedup, expand, a) : wordnet(c[1], dedup, expand, a);
            b.append(dump(m));
          } catch (ParseException e) {
            b.append("X\tParseException\t").append(AnalysisRows.esc(e.getMessage())).append('\n');
          } catch (Exception e) {
            b.append("X\t").append(e.getClass().getSimpleName()).append('\n');
          }
        }
      }
    }
    return b.toString();
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_synonym");
    Files.createDirectories(out);
    String dir = System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus");
    String solrRules = Files.readString(Path.of(dir, "synonyms-solr.txt"), StandardCharsets.UTF_8);
    String wnRules = Files.readString(Path.of(dir, "synonyms-wordnet.txt"), StandardCharsets.UTF_8);

    SynonymMap solrExpand = solr(solrRules, true, true, wsLower());
    SynonymMap solrCollapse = solr(solrRules, true, false, wsLower());
    SynonymMap solrNoDedup = solr(solrRules, false, true, wsLower());
    SynonymMap solrCased = solr(solrRules, true, true, wsPlain());
    SynonymMap wnExpand = wordnet(wnRules, true, true, stdLower());
    SynonymMap wnCollapse = wordnet(wnRules, true, false, stdLower());

    StringBuilder maps = new StringBuilder();
    for (Map.Entry<String, SynonymMap> e :
        Map.of(
                "solr_expand", solrExpand, "solr_collapse", solrCollapse, "solr_nodedup", solrNoDedup,
                "solr_cased", solrCased, "wordnet_expand", wnExpand, "wordnet_collapse", wnCollapse)
            .entrySet().stream().sorted(Map.Entry.comparingByKey()).toList()) {
      maps.append("M\t").append(e.getKey()).append('\n').append(dump(e.getValue()));
    }
    Files.writeString(out.resolve("maps.tsv"), maps.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("parse.tsv"), parseCases(), StandardCharsets.UTF_8);

    Map<String, Supplier<Analyzer>> c = new LinkedHashMap<>();
    c.put("ws_lower_graph_expand", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new SynonymGraphFilter(new LowerCaseFilter(t), solrExpand, true)));
    c.put("ws_lower_graph_collapse", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new SynonymGraphFilter(new LowerCaseFilter(t), solrCollapse, true)));
    c.put("ws_lower_graph_nodedup", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new SynonymGraphFilter(new LowerCaseFilter(t), solrNoDedup, false)));
    c.put("ws_lower_graph_expand_flatten", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new FlattenGraphFilter(new SynonymGraphFilter(new LowerCaseFilter(t), solrExpand, true))));
    c.put("ws_lower_graph_collapse_flatten", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new FlattenGraphFilter(new SynonymGraphFilter(new LowerCaseFilter(t), solrCollapse, false))));
    c.put("ws_graph_cased", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new SynonymGraphFilter(t, solrCased, false)));
    c.put("ws_graph_ignore_case", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new SynonymGraphFilter(t, solrExpand, true)));
    c.put("std_stop_graph_expand", () -> AnalysisRows.chain(StandardTokenizer::new,
        t -> new SynonymGraphFilter(new StopFilter(new LowerCaseFilter(t), EnglishAnalyzer.ENGLISH_STOP_WORDS_SET), wnExpand, false)));
    c.put("std_lower_graph_wordnet", () -> AnalysisRows.chain(StandardTokenizer::new,
        t -> new SynonymGraphFilter(new LowerCaseFilter(t), wnExpand, false)));
    c.put("std_lower_graph_wordnet_collapse_flatten", () -> AnalysisRows.chain(StandardTokenizer::new,
        t -> new FlattenGraphFilter(new SynonymGraphFilter(new LowerCaseFilter(t), wnCollapse, false))));
    c.put("ws_lower_legacy_expand", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new SynonymFilter(new LowerCaseFilter(t), solrExpand, true)));
    c.put("ws_lower_legacy_collapse", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new SynonymFilter(new LowerCaseFilter(t), solrCollapse, false)));
    c.put("ws_legacy_ignore_case", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new SynonymFilter(t, solrExpand, true)));
    c.put("std_stop_legacy_wordnet", () -> AnalysisRows.chain(StandardTokenizer::new,
        t -> new SynonymFilter(new StopFilter(new LowerCaseFilter(t), EnglishAnalyzer.ENGLISH_STOP_WORDS_SET), wnExpand, false)));
    c.put("std_lower_legacy_wordnet_collapse", () -> AnalysisRows.chain(StandardTokenizer::new,
        t -> new SynonymFilter(new LowerCaseFilter(t), wnCollapse, false)));
    // Synonyms over synonyms: the second filter sees the first's graph (undefined in Lucene, but
    // deterministic).
    c.put("ws_lower_graph_twice", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new SynonymGraphFilter(new SynonymGraphFilter(new LowerCaseFilter(t), solrCollapse, false), wnExpand, false)));
    AnalysisRows.writeChains(out, c, AnalysisRows.corpus("analysis-synonym.txt"));
  }
}
