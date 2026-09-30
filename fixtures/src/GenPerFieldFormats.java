import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsFormat;
import org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.TreeMap;

/**
 * A segment whose fields are split across <b>two postings formats and two
 * doc-values formats</b> (per-field routing through {@code
 * PerFieldPostingsFormat}/{@code PerFieldDocValuesFormat}), using formats in
 * lucene-core only: {@code Lucene104PostingsFormat()} and {@code
 * Lucene104PostingsFormat(10, 20)} (same name, different block sizes, so
 * suffixes {@code Lucene104_0}/{@code Lucene104_1}), {@code
 * Lucene90DocValuesFormat()} and {@code Lucene90DocValuesFormat(1024)} ({@code
 * Lucene90_0}/{@code Lucene90_1}).
 *
 * <p>Fields starting with {@code b} go to the second postings format;
 * doc-values fields starting with {@code dvb} to the second doc-values format.
 * The manifest records every term's documents for {@code a_id}, {@code
 * a_text} (the first ten terms) and {@code b_id}/{@code b_tag}, every
 * document's doc values, and each field's per-field attributes, for {@code
 * crates/lucene-search/tests/per_field_formats_fixtures.rs}. The second
 * postings format's own files (all {@code DOCS}, no norms) are the expected
 * bytes of {@code crates/lucene-codecs/tests/per_field_postings_fixture.rs}.
 */
public class GenPerFieldFormats {

  static final String[] WORDS = {
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango"
  };

  static long seed = 20260930L;

  static int next(int bound) {
    seed = seed * 6364136223846793005L + 1442695040888963407L;
    return (int) Long.remainderUnsigned(seed >>> 33, bound);
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("per_field_formats_index");
    if (Files.exists(out)) {
      try (var walk = Files.walk(out)) {
        for (Path p : (Iterable<Path>) walk.sorted(Comparator.reverseOrder())::iterator) {
          Files.delete(p);
        }
      }
    }
    Files.createDirectories(out);

    PostingsFormat postingsA = new Lucene104PostingsFormat();
    PostingsFormat postingsB = new Lucene104PostingsFormat(10, 20);
    DocValuesFormat dvA = new Lucene90DocValuesFormat();
    DocValuesFormat dvB = new Lucene90DocValuesFormat(1024);
    Lucene104Codec codec =
        new Lucene104Codec() {
          @Override
          public PostingsFormat getPostingsFormatForField(String field) {
            return field.startsWith("b") ? postingsB : postingsA;
          }

          @Override
          public DocValuesFormat getDocValuesFormatForField(String field) {
            return field.startsWith("dvb") ? dvB : dvA;
          }
        };

    int numDocs = 400;
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setCodec(codec);
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < numDocs; i++) {
          Document d = new Document();
          d.add(new StringField("a_id", "a" + i, Field.Store.NO));
          StringBuilder text = new StringBuilder();
          for (int k = 0, n = 3 + next(12); k < n; k++) {
            text.append(WORDS[next(WORDS.length)]).append(' ');
          }
          d.add(new TextField("a_text", text.toString(), Field.Store.NO));
          d.add(new StringField("b_id", String.format("b%05d", i * 3), Field.Store.NO));
          d.add(new StringField("b_tag", "tag" + next(60), Field.Store.NO));
          d.add(new NumericDocValuesField("dva_num", next(1000)));
          d.add(new NumericDocValuesField("dvb_num", -next(100000)));
          d.add(new SortedDocValuesField("dvb_sorted", new BytesRef(WORDS[next(WORDS.length)])));
          w.addDocument(d);
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      SegmentCommitInfo sci = sis.info(0);
      StringBuilder m = new StringBuilder();
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        LeafReader leaf = r.leaves().get(0).reader();
        for (FieldInfo fi : leaf.getFieldInfos()) {
          m.append("field.").append(fi.name).append(".number=").append(fi.number).append('\n');
          TreeMap<String, String> attrs = new TreeMap<>(fi.attributes());
          StringBuilder a = new StringBuilder();
          for (var e : attrs.entrySet()) {
            if (a.length() > 0) a.append(',');
            a.append(e.getKey()).append(':').append(e.getValue());
          }
          m.append("field.").append(fi.name).append(".attributes=").append(a).append('\n');
        }
        for (String field : new String[] {"a_id", "a_text", "b_id", "b_tag"}) {
          Terms terms = leaf.terms(field);
          TermsEnum te = terms.iterator();
          int count = 0;
          StringBuilder termLines = new StringBuilder();
          for (BytesRef t = te.next(); t != null; t = te.next()) {
            count++;
            if (field.equals("a_text") && count > 10) {
              continue;
            }
            PostingsEnum pe = te.postings(null, PostingsEnum.FREQS);
            StringBuilder docs = new StringBuilder();
            for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
              if (docs.length() > 0) docs.append(',');
              docs.append(doc).append(':').append(pe.freq());
            }
            termLines.append(t.utf8ToString()).append('=').append(docs).append(';');
          }
          m.append("terms.").append(field).append('=').append(termLines).append('\n');
          m.append("num_terms.").append(field).append('=').append(count).append('\n');
        }
        for (String field : new String[] {"dva_num", "dvb_num"}) {
          NumericDocValues v = leaf.getNumericDocValues(field);
          StringBuilder vals = new StringBuilder();
          for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
            if (vals.length() > 0) vals.append(',');
            vals.append(v.longValue());
          }
          m.append("dv.").append(field).append('=').append(vals).append('\n');
        }
        SortedDocValues sorted = leaf.getSortedDocValues("dvb_sorted");
        StringBuilder vals = new StringBuilder();
        for (int doc = sorted.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = sorted.nextDoc()) {
          if (vals.length() > 0) vals.append(',');
          vals.append(sorted.lookupOrd(sorted.ordValue()).utf8ToString());
        }
        m.append("dv.dvb_sorted=").append(vals).append('\n');
      }
      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }
    System.out.println("wrote per_field_formats_index/");
  }

  static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder();
    for (byte v : b) sb.append(String.format("%02x", v));
    return sb.toString();
  }
}
