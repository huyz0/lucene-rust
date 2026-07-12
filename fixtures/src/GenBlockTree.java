import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
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
 * Generates real `.tim`/`.tip`/`.tmd` (Lucene103BlockTreeTermsReader/Writer, via
 * Lucene104PostingsFormat) fixtures, plus the `.fnm`/`.si` this port's own readers
 * need to open them.
 *
 * <p>Deliberately small (well under the default minItemsInBlock=25/maxItemsInBlock=48
 * thresholds) so every field's term dictionary is exactly one non-floor, leaf `.tim`
 * block -- see `crates/lucene-codecs/src/blocktree.rs`'s module doc for why that's this
 * slice's scope. Uses a hand-built {@link TokenStream} (not a real analyzer) so every
 * term's exact per-doc frequency is known up front, the same technique
 * {@link GenTermVectors} uses.
 *
 * <p>Two fields exercise both `sumTotalTermFreq`/`sumDocFreq` wire shapes:
 * "body" (`IndexOptions.DOCS_AND_FREQS`, repeated terms with varying frequencies) and
 * "id" (`IndexOptions.DOCS`, one distinct token per doc, freq always 1 -- exercises the
 * DOCS-only aliasing path where only one vlong is written for sumDocFreq/sumTotalTermFreq
 * and per-term stats never write a separate totalTermFreq).
 */
public class GenBlockTree {

  static final class CannedTokenStream extends TokenStream {
    private final List<String> tokens;
    private int index = 0;
    private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);

    CannedTokenStream(List<String> tokens) {
      this.tokens = tokens;
    }

    @Override
    public boolean incrementToken() {
      if (index >= tokens.size()) {
        return false;
      }
      clearAttributes();
      termAtt.append(tokens.get(index++));
      return true;
    }

    @Override
    public void reset() throws IOException {
      super.reset();
      index = 0;
    }
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("blocktree_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    FieldType bodyType = new FieldType();
    bodyType.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    bodyType.setTokenized(true);
    bodyType.freeze();

    FieldType idType = new FieldType();
    idType.setIndexOptions(IndexOptions.DOCS);
    idType.setTokenized(true);
    idType.freeze();

    // doc0: body "cat cat dog" -> cat freq=2, dog freq=1
    // doc1: body "dog bird"    -> dog freq=1, bird freq=1
    // doc2: body "cat"         -> cat freq=1
    // doc3: no body field at all (docCount < maxDoc path)
    // doc4: body "bird bird bird" -> bird freq=3
    //
    // body totals: cat docFreq=2 totalTermFreq=3; dog docFreq=2 totalTermFreq=2;
    //              bird docFreq=2 totalTermFreq=4
    String[][] bodies = {
      {"cat", "cat", "dog"},
      {"dog", "bird"},
      {"cat"},
      null,
      {"bird", "bird", "bird"}
    };

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < bodies.length; i++) {
          Document doc = new Document();
          doc.add(new Field("id", new CannedTokenStream(List.of("id" + i)), idType));
          if (bodies[i] != null) {
            doc.add(new Field("body", new CannedTokenStream(List.of(bodies[i])), bodyType));
          }
          w.addDocument(doc);
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);

      String timFileName = null, tipFileName = null, tmdFileName = null;
      String fnmFileName = null, siFileName = null;
      for (String f : sci.info.files()) {
        if (f.endsWith(".tim")) timFileName = f;
        if (f.endsWith(".tip")) tipFileName = f;
        if (f.endsWith(".tmd")) tmdFileName = f;
        if (f.endsWith(".fnm")) fnmFileName = f;
        if (f.endsWith(".si")) siFileName = f;
      }
      if (timFileName == null || tipFileName == null || tmdFileName == null) {
        throw new AssertionError("expected .tim/.tip/.tmd files, files=" + sci.info.files());
      }
      if (fnmFileName == null || siFileName == null) {
        throw new AssertionError("expected .fnm/.si files, files=" + sci.info.files());
      }

      dump(dir, timFileName, out);
      dump(dir, tipFileName, out);
      dump(dir, tmdFileName, out);
      dump(dir, fnmFileName, out);
      dump(dir, siFileName, out);

      StringBuilder m = new StringBuilder();
      m.append("tim_file_name=").append(timFileName).append('\n');
      m.append("tip_file_name=").append(tipFileName).append('\n');
      m.append("tmd_file_name=").append(tmdFileName).append('\n');
      m.append("fnm_file_name=").append(fnmFileName).append('\n');
      m.append("si_file_name=").append(siFileName).append('\n');
      // PerFieldPostingsFormat assigns a per-format segment suffix (e.g.
      // "Lucene104_0"), embedded in the .tim/.tip/.tmd file names as
      // "<segmentName>_<suffix>.<ext>" -- recover it from the .tim name
      // since that's the suffix Lucene103BlockTreeTermsReader itself checks.
      String prefix = sci.info.name + "_";
      String segmentSuffix = timFileName.substring(prefix.length(), timFileName.length() - 4);
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("segment_suffix=").append(segmentSuffix).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();

        appendFieldManifest(
            m,
            leaf,
            "body",
            new String[] {"cat", "dog", "bird", "zzz-missing", "", "ca"});
        appendFieldManifest(
            m,
            leaf,
            "id",
            new String[] {"id0", "id1", "id2", "id3", "id4", "id5-missing"});
      }

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote blocktree_index/ fixture directory");
  }

  static void appendFieldManifest(StringBuilder m, LeafReader leaf, String field, String[] lookups)
      throws IOException {
    Terms terms = leaf.terms(field);
    if (terms == null) {
      throw new AssertionError("expected terms for field " + field);
    }
    m.append("field.").append(field).append(".numTerms=").append(terms.size()).append('\n');
    m.append("field.")
        .append(field)
        .append(".sumDocFreq=")
        .append(terms.getSumDocFreq())
        .append('\n');
    m.append("field.")
        .append(field)
        .append(".sumTotalTermFreq=")
        .append(terms.getSumTotalTermFreq())
        .append('\n');
    m.append("field.").append(field).append(".docCount=").append(terms.getDocCount()).append('\n');
    m.append("field.").append(field).append(".minTerm=").append(terms.getMin().utf8ToString()).append('\n');
    m.append("field.").append(field).append(".maxTerm=").append(terms.getMax().utf8ToString()).append('\n');

    TermsEnum te = terms.iterator();
    for (String lookup : lookups) {
      boolean found = te.seekExact(new BytesRef(lookup));
      String key = "field." + field + ".term." + (lookup.isEmpty() ? "EMPTY" : lookup);
      if (found) {
        m.append(key).append(".found=true\n");
        m.append(key).append(".docFreq=").append(te.docFreq()).append('\n');
        m.append(key).append(".totalTermFreq=").append(te.totalTermFreq()).append('\n');
      } else {
        m.append(key).append(".found=false\n");
      }
    }
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
    StringBuilder sb = new StringBuilder();
    for (byte value : b) sb.append(String.format("%02x", value));
    return sb.toString();
  }
}
