import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Generates real `.tvd`/`.tvx`/`.tvm` (Lucene90TermVectorsFormat) fixtures.
 * Uses a hand-built {@link TokenStream} (not a real analyzer) so the exact
 * term, position, offset, and payload of every token is known up front and
 * can be cross-checked in the manifest -- three documents, with terms
 * repeating (to exercise same-term multi-occurrence delta chains), varying
 * field counts per doc (to exercise the chunk's per-doc field-count array),
 * and one doc with no term-vector field at all (numFields==0 path).
 */
public class GenTermVectors {

  /** One token: term text, position increment, start/end offset, payload (or null). */
  record Tok(String term, int posInc, int startOffset, int endOffset, byte[] payload) {}

  static final class CannedTokenStream extends TokenStream {
    private final List<Tok> tokens;
    private int index = 0;
    private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
    private final PositionIncrementAttribute posIncAtt = addAttribute(PositionIncrementAttribute.class);
    private final OffsetAttribute offsetAtt = addAttribute(OffsetAttribute.class);
    private final PayloadAttribute payloadAtt = addAttribute(PayloadAttribute.class);

    CannedTokenStream(List<Tok> tokens) {
      this.tokens = tokens;
    }

    @Override
    public boolean incrementToken() {
      if (index >= tokens.size()) {
        return false;
      }
      clearAttributes();
      Tok t = tokens.get(index++);
      termAtt.append(t.term());
      posIncAtt.setPositionIncrement(t.posInc());
      offsetAtt.setOffset(t.startOffset(), t.endOffset());
      payloadAtt.setPayload(t.payload() == null ? null : new BytesRef(t.payload()));
      return true;
    }

    @Override
    public void reset() throws IOException {
      super.reset();
      index = 0;
    }
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("term_vectors_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    FieldType tvType = new FieldType();
    tvType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    tvType.setStoreTermVectors(true);
    tvType.setStoreTermVectorPositions(true);
    tvType.setStoreTermVectorOffsets(true);
    tvType.setStoreTermVectorPayloads(true);
    tvType.setTokenized(true);
    tvType.freeze();

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        // doc 0: field "text" -- "cat" (twice) and "car" (once), with payloads.
        Document doc0 = new Document();
        doc0.add(new StringField("id", "0", Field.Store.NO));
        doc0.add(
            new Field(
                "text",
                new CannedTokenStream(
                    List.of(
                        new Tok("cat", 1, 0, 3, new byte[] {(byte) 0xAA}),
                        new Tok("car", 1, 4, 7, new byte[] {(byte) 0xBB, (byte) 0xCC}),
                        new Tok("cat", 1, 8, 11, null))),
                tvType));
        w.addDocument(doc0);

        // doc 1: two fields "text" and "title", to exercise multi-field docs
        // and the distinct-field-numbers array.
        Document doc1 = new Document();
        doc1.add(new StringField("id", "1", Field.Store.NO));
        doc1.add(
            new Field(
                "text",
                new CannedTokenStream(
                    List.of(new Tok("dog", 1, 0, 3, null), new Tok("run", 1, 4, 7, null))),
                tvType));
        doc1.add(
            new Field(
                "title",
                new CannedTokenStream(List.of(new Tok("hello", 1, 0, 5, null))),
                tvType));
        w.addDocument(doc1);

        // doc 2: no term-vector field at all (numFields==0 path).
        Document doc2 = new Document();
        doc2.add(new StringField("id", "2", Field.Store.NO));
        w.addDocument(doc2);

        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);

      String tvdFileName = null;
      String tvxFileName = null;
      String tvmFileName = null;
      for (String f : sci.info.files()) {
        if (f.endsWith(".tvd")) tvdFileName = f;
        if (f.endsWith(".tvx")) tvxFileName = f;
        if (f.endsWith(".tvm")) tvmFileName = f;
      }
      if (tvdFileName == null || tvxFileName == null || tvmFileName == null) {
        throw new AssertionError("expected .tvd/.tvx/.tvm files, files=" + sci.info.files());
      }

      dump(dir, tvdFileName, out);
      dump(dir, tvxFileName, out);
      dump(dir, tvmFileName, out);

      StringBuilder m = new StringBuilder();
      m.append("tvd_file_name=").append(tvdFileName).append('\n');
      m.append("tvx_file_name=").append(tvxFileName).append('\n');
      m.append("tvm_file_name=").append(tvmFileName).append('\n');
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');

      org.apache.lucene.codecs.TermVectorsReader tvReader =
          sci.info
              .getCodec()
              .termVectorsFormat()
              .vectorsReader(
                  dir,
                  sci.info,
                  sci.info.getCodec().fieldInfosFormat().read(dir, sci.info, "", IOContext.READONCE),
                  IOContext.READONCE);

      for (int doc = 0; doc < sci.info.maxDoc(); doc++) {
        org.apache.lucene.index.Fields fields = tvReader.get(doc);
        if (fields == null) {
          m.append("doc.").append(doc).append(".fields=NONE\n");
          continue;
        }
        StringBuilder fieldNames = new StringBuilder();
        for (String fieldName : fields) {
          if (fieldNames.length() > 0) fieldNames.append(',');
          fieldNames.append(fieldName);

          Terms terms = fields.terms(fieldName);
          TermsEnum te = terms.iterator();
          StringBuilder termsOut = new StringBuilder();
          BytesRef term;
          while ((term = te.next()) != null) {
            if (termsOut.length() > 0) termsOut.append(';');
            termsOut.append(term.utf8ToString());
            PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
            pe.nextDoc();
            int freq = pe.freq();
            termsOut.append(':').append(freq);
            for (int k = 0; k < freq; k++) {
              int pos = pe.nextPosition();
              termsOut
                  .append(':')
                  .append(pos)
                  .append(',')
                  .append(pe.startOffset())
                  .append(',')
                  .append(pe.endOffset())
                  .append(',');
              BytesRef payload = pe.getPayload();
              termsOut.append(payload == null ? "NONE" : hex(payload.bytes, payload.offset, payload.length));
            }
          }
          m.append("doc.")
              .append(doc)
              .append(".field.")
              .append(fieldName)
              .append(".terms=")
              .append(termsOut)
              .append('\n');
        }
        m.append("doc.").append(doc).append(".fields=").append(fieldNames).append('\n');
      }
      tvReader.close();

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote term_vectors_index/ fixture directory");
  }

  static void dump(Directory dir, String fileName, Path out) throws IOException {
    try (IndexInput in = dir.openInput(fileName, IOContext.READONCE)) {
      byte[] bytes = new byte[(int) in.length()];
      in.readBytes(bytes, 0, bytes.length);
      Files.write(out.resolve(fileName + ".raw"), bytes);
    }
  }

  static void deleteRecursive(Path p) throws IOException {
    if (Files.isDirectory(p)) {
      try (var entries = Files.list(p)) {
        for (Path child : (Iterable<Path>) entries::iterator) {
          deleteRecursive(child);
        }
      }
    }
    Files.deleteIfExists(p);
  }

  static String hex(byte[] b) {
    return hex(b, 0, b.length);
  }

  static String hex(byte[] b, int offset, int length) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < length; i++) sb.append(String.format("%02x", b[offset + i]));
    return sb.toString();
  }
}
