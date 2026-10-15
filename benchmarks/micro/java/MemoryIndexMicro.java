import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.memory.MemoryIndex;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.Query;

/**
 * M13 T13.5's pairs (the Rust twin is {@code benchmarks/rust-runner/src/micro_memory_index.rs}):
 * {@code MemoryIndex} over {@code fixtures/corpus/memory-index-docs.txt} with offsets, the queries of
 * {@code memory-index-queries.txt} parsed beforehand. {@code mi_build}: every document indexed and
 * frozen (units: documents); {@code mi_build_search}: every document indexed, then every query run
 * against it, the monitor's shape (units: document x query); {@code mi_search_frozen}: every query
 * against every document indexed and frozen beforehand (units: queries). Output is {@code
 * name\tns_per_unit\tunits}.
 */
public class MemoryIndexMicro {
  static List<String[]> fields(String line) {
    List<String[]> out = new ArrayList<>();
    for (String pair : line.split("\t")) {
      int eq = pair.indexOf('=');
      out.add(new String[] {pair.substring(0, eq), pair.substring(eq + 1)});
    }
    return out;
  }

  static MemoryIndex build(List<String[]> doc, Analyzer a) {
    MemoryIndex mi = new MemoryIndex(true);
    for (String[] f : doc) {
      mi.addField(f[0], f[1], a);
    }
    mi.freeze();
    return mi;
  }

  public static void main(String[] args) throws Exception {
    Analyzer a = new StandardAnalyzer();
    List<List<String[]>> docs = new ArrayList<>();
    for (String line : Files.readAllLines(Path.of("fixtures/corpus/memory-index-docs.txt"), StandardCharsets.UTF_8)) {
      docs.add(fields(line));
    }
    QueryParser parser = new QueryParser("body", a);
    List<Query> queries = new ArrayList<>();
    for (String q : Files.readAllLines(Path.of("fixtures/corpus/memory-index-queries.txt"), StandardCharsets.UTF_8)) {
      queries.add(parser.parse(q));
    }
    SweepMicro.measure("mi_build", () -> {
      for (List<String[]> d : docs) {
        SweepMicro.sink += build(d, a).hashCode();
      }
      return docs.size();
    });
    SweepMicro.measure("mi_build_search", () -> {
      for (List<String[]> d : docs) {
        MemoryIndex mi = build(d, a);
        for (Query q : queries) {
          SweepMicro.sink += Float.floatToIntBits(mi.search(q));
        }
      }
      return (long) docs.size() * queries.size();
    });
    List<MemoryIndex> frozen = new ArrayList<>();
    for (List<String[]> d : docs) {
      frozen.add(build(d, a));
    }
    SweepMicro.measure("mi_search_frozen", () -> {
      for (MemoryIndex mi : frozen) {
        for (Query q : queries) {
          SweepMicro.sink += Float.floatToIntBits(mi.search(q));
        }
      }
      return (long) frozen.size() * queries.size();
    });
  }
}
