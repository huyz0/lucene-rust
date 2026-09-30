import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedSetSortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * {@code TopDocs.merge} (by score and by a {@code Sort}, with {@code start}, shard indices set or
 * unset, and custom tie-breakers) and {@code TopDocs.rrf}, recorded from Lucene for
 * crates/lucene-search/tests/top_docs_fixtures.rs.
 *
 * <p>No index: every input is a synthetic {@code TopDocs} drawn from a seeded {@link Random},
 * with scores from a small set so ties across and within shards are common. Each case records its
 * inputs and the merged output (documents, score bits, shard indices, the total and its
 * relation), or {@code error} when Lucene throws.
 *
 * <p>Format ({@code top_docs/cases.txt}), one key per line: {@code case.N.kind} is
 * {@code score}, {@code field} or {@code rrf}; {@code case.N.shard.S} is
 * {@code total:rel;hit;hit...} with a hit {@code doc:scoreBits:shardIndex[:v0:v1...]} (a sort value
 * is a long, a double's bits, a float's bits, an int, or {@code -} / hex bytes for a keyword);
 * {@code case.N.out} is {@code total:rel;hit;...} or {@code error}.
 */
public class GenTopDocs {
  static final float[] SCORES = {0.5f, 1f, 1.5f, 2f, 2.25f, Float.MIN_VALUE, 3f};

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("top_docs");
    Files.createDirectories(out);
    StringBuilder m = new StringBuilder();
    Random r = new Random(20260930L);
    int n = 0;

    // Merge by score.
    for (int i = 0; i < 60; i++) {
      boolean shardSet = i % 3 != 0;
      int shards = 1 + r.nextInt(5);
      TopDocs[] in = new TopDocs[shards];
      for (int s = 0; s < shards; s++) {
        in[s] = scoreShard(r, s, shardSet, i % 7 == 0);
      }
      int start = i % 4 == 0 ? r.nextInt(6) : 0;
      int topN = r.nextInt(12);
      String tie = new String[] {"default", "doc", "shard"}[i % 3];
      Comparator<ScoreDoc> cmp = tieBreaker(tie);
      String k = "case." + n++;
      m.append(k).append(".kind=score\n");
      m.append(k).append(".start=").append(start).append('\n');
      m.append(k).append(".top_n=").append(topN).append('\n');
      m.append(k).append(".tie=").append(tie).append('\n');
      m.append(k).append(".shards=").append(shards).append('\n');
      for (int s = 0; s < shards; s++) {
        m.append(k).append(".shard.").append(s).append('=').append(render(in[s], null)).append('\n');
      }
      String result;
      try {
        result = render(TopDocs.merge(start, topN, in, cmp), null);
      } catch (IllegalArgumentException e) {
        result = "error";
      }
      m.append(k).append(".out=").append(result).append('\n');
    }

    // One inconsistent case: shard indices set on one shard and unset on another.
    {
      TopDocs a =
          new TopDocs(
              new TotalHits(1, TotalHits.Relation.EQUAL_TO), new ScoreDoc[] {new ScoreDoc(1, 2f, 0)});
      TopDocs b =
          new TopDocs(
              new TotalHits(1, TotalHits.Relation.EQUAL_TO), new ScoreDoc[] {new ScoreDoc(2, 1f)});
      String k = "case." + n++;
      m.append(k).append(".kind=score\n");
      m.append(k).append(".start=0\n").append(k).append(".top_n=5\n");
      m.append(k).append(".tie=default\n").append(k).append(".shards=2\n");
      m.append(k).append(".shard.0=").append(render(a, null)).append('\n');
      m.append(k).append(".shard.1=").append(render(b, null)).append('\n');
      String result;
      try {
        result = render(TopDocs.merge(0, 5, new TopDocs[] {a, b}), null);
      } catch (IllegalArgumentException e) {
        result = "error";
      }
      m.append(k).append(".out=").append(result).append('\n');
    }

    // Merge by a sort.
    String[][] sorts = {
      {"long:false"},
      {"long:true", "doc:false"},
      {"score:false"},
      {"string_first:false", "long:false"},
      {"string_last:true"},
      {"double:false", "score:false"},
      {"float:true"},
      {"int:false", "string_last:false"},
    };
    for (int i = 0; i < 64; i++) {
      String[] spec = sorts[i % sorts.length];
      Sort sort = sort(spec);
      boolean shardSet = i % 5 != 0;
      int shards = 1 + r.nextInt(4);
      TopFieldDocs[] in = new TopFieldDocs[shards];
      for (int s = 0; s < shards; s++) {
        in[s] = fieldShard(r, s, shardSet, spec, sort);
      }
      int start = i % 3 == 0 ? r.nextInt(5) : 0;
      int topN = r.nextInt(10);
      String tie = i % 4 == 3 ? "doc" : "default";
      String k = "case." + n++;
      m.append(k).append(".kind=field\n");
      m.append(k).append(".sort=").append(String.join(",", spec)).append('\n');
      m.append(k).append(".start=").append(start).append('\n');
      m.append(k).append(".top_n=").append(topN).append('\n');
      m.append(k).append(".tie=").append(tie).append('\n');
      m.append(k).append(".shards=").append(shards).append('\n');
      for (int s = 0; s < shards; s++) {
        m.append(k).append(".shard.").append(s).append('=').append(render(in[s], spec)).append('\n');
      }
      String result;
      try {
        result = render(TopDocs.merge(sort, start, topN, in, tieBreaker(tie)), spec);
      } catch (IllegalArgumentException e) {
        result = "error";
      }
      m.append(k).append(".out=").append(result).append('\n');
    }

    // Reciprocal rank fusion.
    for (int i = 0; i < 40; i++) {
      boolean shardSet = i % 4 == 1;
      int lists = 1 + r.nextInt(4);
      TopDocs[] in = new TopDocs[lists];
      for (int s = 0; s < lists; s++) {
        in[s] = rrfList(r, shardSet);
      }
      int topN = i == 7 ? 0 : 1 + r.nextInt(15);
      int k = i == 9 ? 0 : new int[] {1, 2, 10, 60}[i % 4];
      String key = "case." + n++;
      m.append(key).append(".kind=rrf\n");
      m.append(key).append(".top_n=").append(topN).append('\n');
      m.append(key).append(".k=").append(k).append('\n');
      m.append(key).append(".shards=").append(lists).append('\n');
      for (int s = 0; s < lists; s++) {
        m.append(key).append(".shard.").append(s).append('=').append(render(in[s], null)).append('\n');
      }
      String result;
      try {
        result = render(TopDocs.rrf(topN, k, in), null);
      } catch (IllegalArgumentException e) {
        result = "error";
      }
      m.append(key).append(".out=").append(result).append('\n');
    }
    // A mix of set and unset shard indices across lists.
    {
      TopDocs a =
          new TopDocs(
              new TotalHits(1, TotalHits.Relation.EQUAL_TO), new ScoreDoc[] {new ScoreDoc(1, 2f, 0)});
      TopDocs b =
          new TopDocs(
              new TotalHits(1, TotalHits.Relation.EQUAL_TO), new ScoreDoc[] {new ScoreDoc(2, 1f)});
      String key = "case." + n++;
      m.append(key).append(".kind=rrf\n").append(key).append(".top_n=5\n");
      m.append(key).append(".k=60\n").append(key).append(".shards=2\n");
      m.append(key).append(".shard.0=").append(render(a, null)).append('\n');
      m.append(key).append(".shard.1=").append(render(b, null)).append('\n');
      String result;
      try {
        result = render(TopDocs.rrf(5, 60, new TopDocs[] {a, b}), null);
      } catch (IllegalArgumentException e) {
        result = "error";
      }
      m.append(key).append(".out=").append(result).append('\n');
    }

    m.append("case_count=").append(n).append('\n');
    Files.writeString(out.resolve("cases.txt"), m.toString(), StandardCharsets.UTF_8);
  }

  static Comparator<ScoreDoc> tieBreaker(String name) {
    switch (name) {
      case "doc":
        return Comparator.comparingInt(d -> d.doc);
      case "shard":
        return Comparator.comparingInt(d -> d.shardIndex);
      default:
        return Comparator.<ScoreDoc>comparingInt(d -> d.shardIndex)
            .thenComparingInt(d -> d.doc);
    }
  }

  static TopDocs scoreShard(Random r, int shard, boolean shardSet, boolean empty) {
    int count = empty ? 0 : r.nextInt(8);
    List<ScoreDoc> hits = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      float score = SCORES[r.nextInt(SCORES.length)];
      int doc = r.nextInt(30);
      hits.add(shardSet ? new ScoreDoc(doc, score, shard) : new ScoreDoc(doc, score));
    }
    // A shard's hits are sorted: by score descending, then doc ascending.
    hits.sort((a, b) -> a.score != b.score ? Float.compare(b.score, a.score) : Integer.compare(a.doc, b.doc));
    long total = count + r.nextInt(3);
    TotalHits.Relation rel =
        r.nextInt(4) == 0 ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
    return new TopDocs(new TotalHits(total, rel), hits.toArray(new ScoreDoc[0]));
  }

  static TopDocs rrfList(Random r, boolean shardSet) {
    int count = r.nextInt(10);
    List<ScoreDoc> hits = new ArrayList<>();
    java.util.Set<Long> seen = new java.util.HashSet<>();
    while (hits.size() < count) {
      int doc = r.nextInt(25);
      int shard = shardSet ? r.nextInt(3) : -1;
      if (!seen.add(((long) shard << 32) | doc)) {
        continue;
      }
      hits.add(new ScoreDoc(doc, 10f - hits.size(), shard));
    }
    long total = count + r.nextInt(20);
    return new TopDocs(
        new TotalHits(total, TotalHits.Relation.EQUAL_TO), hits.toArray(new ScoreDoc[0]));
  }

  static Sort sort(String[] spec) {
    SortField[] fields = new SortField[spec.length];
    for (int i = 0; i < spec.length; i++) {
      String[] p = spec[i].split(":");
      boolean reverse = Boolean.parseBoolean(p[1]);
      switch (p[0]) {
        case "long":
          fields[i] = new SortField("f" + i, SortField.Type.LONG, reverse);
          break;
        case "int":
          fields[i] = new SortField("f" + i, SortField.Type.INT, reverse);
          break;
        case "double":
          fields[i] = new SortField("f" + i, SortField.Type.DOUBLE, reverse);
          break;
        case "float":
          fields[i] = new SortField("f" + i, SortField.Type.FLOAT, reverse);
          break;
        case "score":
          fields[i] = new SortField(null, SortField.Type.SCORE, reverse);
          break;
        case "doc":
          fields[i] = new SortField(null, SortField.Type.DOC, reverse);
          break;
        case "string_first":
          fields[i] = new SortedSetSortField("f" + i, reverse);
          fields[i].setMissingValue(SortField.STRING_FIRST);
          break;
        case "string_last":
          fields[i] = new SortedSetSortField("f" + i, reverse);
          fields[i].setMissingValue(SortField.STRING_LAST);
          break;
        default:
          throw new AssertionError(p[0]);
      }
    }
    return new Sort(fields);
  }

  static Object value(Random r, String type, int doc, float score) {
    switch (type) {
      case "long":
        return (long) (r.nextInt(7) - 3) * (r.nextBoolean() ? 1 : 1_000_000_000_000L);
      case "int":
        return r.nextInt(5) - 2;
      case "double":
        return new double[] {-0.0, 0.0, 1.5, -2.5, Double.NaN, 1e300, 1.5}[r.nextInt(7)];
      case "float":
        return new float[] {-0.0f, 0.0f, 1.5f, -2.5f, Float.NaN, 3f}[r.nextInt(6)];
      case "score":
        return score;
      case "doc":
        return doc;
      default:
        if (r.nextInt(4) == 0) {
          return null;
        }
        return new BytesRef(new String[] {"a", "ab", "b", "é", "ba"}[r.nextInt(5)]);
    }
  }

  static TopFieldDocs fieldShard(Random r, int shard, boolean shardSet, String[] spec, Sort sort) {
    int count = r.nextInt(7);
    List<FieldDoc> hits = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      int doc = r.nextInt(40);
      float score = SCORES[r.nextInt(SCORES.length)];
      Object[] fields = new Object[spec.length];
      for (int f = 0; f < spec.length; f++) {
        fields[f] = value(r, spec[f].split(":")[0], doc, score);
      }
      hits.add(shardSet ? new FieldDoc(doc, score, fields, shard) : new FieldDoc(doc, score, fields));
    }
    // A shard's hits are sorted by the sort, then by doc.
    SortField[] sf = sort.getSort();
    hits.sort(
        (a, b) -> {
          for (int f = 0; f < sf.length; f++) {
            @SuppressWarnings({"rawtypes", "unchecked"})
            org.apache.lucene.search.FieldComparator<Object> c =
                (org.apache.lucene.search.FieldComparator<Object>)
                    sf[f].getComparator(1, org.apache.lucene.search.Pruning.NONE);
            int cmp = c.compareValues(a.fields[f], b.fields[f]);
            if (sf[f].getReverse()) {
              cmp = -cmp;
            }
            if (cmp != 0) {
              return cmp;
            }
          }
          return Integer.compare(a.doc, b.doc);
        });
    long total = count + r.nextInt(3);
    TotalHits.Relation rel =
        r.nextInt(4) == 0 ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
    return new TopFieldDocs(new TotalHits(total, rel), hits.toArray(new FieldDoc[0]), sf);
  }

  static String render(TopDocs td, String[] spec) {
    StringBuilder b = new StringBuilder();
    b.append(td.totalHits.value())
        .append(':')
        .append(td.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "eq" : "gte");
    for (ScoreDoc sd : td.scoreDocs) {
      b.append(';')
          .append(sd.doc)
          .append(':')
          .append(Float.floatToRawIntBits(sd.score))
          .append(':')
          .append(sd.shardIndex);
      if (spec != null) {
        FieldDoc fd = (FieldDoc) sd;
        for (int f = 0; f < spec.length; f++) {
          b.append(':').append(renderValue(spec[f].split(":")[0], fd.fields[f]));
        }
      }
    }
    return b.toString();
  }

  static String renderValue(String type, Object v) {
    switch (type) {
      case "long":
        return Long.toString((Long) v);
      case "int":
        return Integer.toString((Integer) v);
      case "double":
        return Long.toString(Double.doubleToRawLongBits((Double) v));
      case "float":
        return Integer.toString(Float.floatToRawIntBits((Float) v));
      case "score":
        return Integer.toString(Float.floatToRawIntBits((Float) v));
      case "doc":
        return Integer.toString((Integer) v);
      default:
        if (v == null) {
          return "-";
        }
        BytesRef br = (BytesRef) v;
        StringBuilder h = new StringBuilder("x");
        for (int i = 0; i < br.length; i++) {
          h.append(String.format("%02x", br.bytes[br.offset + i] & 0xff));
        }
        return h.toString();
    }
  }
}
