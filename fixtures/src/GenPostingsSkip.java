import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.PostingsEnum;
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
import java.util.ArrayList;
import java.util.List;

/**
 * Real `.doc`/`.pos`/`.pay` bytes for a positions-indexing term long enough to
 * carry **skip data**, which no other fixture in this tree does.
 *
 * <p>Why it exists. `Lucene104PostingsWriter` writes, into every `.doc`
 * level-0 block header and level-1 span entry of a field that indexes
 * positions, the `.pos`/`.pay` file pointer and in-block offset that record's
 * documents' occurrences begin at (`flushDocBlock`'s
 * `posOut.getFilePointer() - level0LastPosFP` / `posBufferUpto` pair, and the
 * same again in `writeLevel1SkipData`). `Lucene104PostingsReader.seekPosData`
 * uses them so `advance(doc)` can jump `.pos` instead of summing every
 * preceding document's frequency.
 *
 * <p>`blocktree_index`'s own positions field ("pos") has `docFreq = 3` and
 * `totalTermFreq = 4`: everything lives in the vint tail, no `.doc` full block
 * exists, and therefore not one byte of that skip data is present. This
 * fixture makes it present: one term in {@value #DOC_FREQ} documents, past
 * `LEVEL1_NUM_DOCS` (32 * 256 = 8192), so real Lucene emits a level-1 entry,
 * 32 level-0 block headers under it, a trailing run of level-0 blocks, and a
 * group-varint tail -- each level-0 header and the level-1 entry carrying the
 * pos/pay sub-fields. Per-document frequencies cycle 1..5, a period **coprime
 * with 256**, so no `.pos` block boundary lines up with a `.doc` block
 * boundary and `posBufferUpto` is non-zero in nearly every record -- including
 * the level-1 one, which is 253. (With the period-4 cycle this generator
 * started with, the 8192-document level-1 boundary landed exactly on a `.pos`
 * block boundary and *every* level-1 `posBufferUpto` was `0`, so a reader
 * that ignored the field entirely would have passed.) Offsets and payloads are
 * both indexed so the `.pay` half of the skip data exists too, and payload
 * lengths vary (including zero-length) so a block's payload byte run is not
 * uniform.
 *
 * <p>A second, sparser term ({@link #SPARSE_TERM}) shares the field: see its
 * doc for why one term in every document is not enough.
 *
 * <p>Ground truth is taken through Java's own `PostingsEnum.advance(doc)` +
 * `nextPosition()`/`startOffset()`/`endOffset()`/`getPayload()` -- the exact
 * API `crates/lucene-codecs/src/postings.rs`'s
 * `read_occurrences_for_doc` ports -- for a sample of documents chosen to
 * bracket every structural boundary (first, last, either side of the level-1
 * span end, either side of several level-0 block ends, and the first document
 * of the vint tail).
 *
 * <p>Kept deliberately to one field and two terms: this fixture exists for the
 * skip data, and 8000+ documents is already the slow part of generating it.
 */
public class GenPostingsSkip {

  /** Past LEVEL1_NUM_DOCS (8192) so exactly one level-1 entry is written. */
  static final int DOC_FREQ = 8500;

  static final String TERM = "skipterm";

  /**
   * A second term, present in only about 40% of the documents, so the `.doc`
   * blocks it produces are *not* all-consecutive.
   *
   * <p>{@link #TERM} is in every document, which makes every one of its
   * level-0 blocks take Lucene's `docRange == BLOCK_SIZE` degenerate encoding
   * (the `bitsPerValue == 0` marker, no body bytes at all). That is the one
   * doc-delta shape that carries no information, so without this second term
   * the cross-engine ground truth would never cover a skip-driven
   * `advance(doc)` into a packed-FOR or unary-bit-set block, nor an
   * `advance(doc)` whose target the term does not contain.
   */
  static final String SPARSE_TERM = "gapterm";

  /** True when {@link #SPARSE_TERM} occurs in document {@code d}. */
  static boolean hasSparse(int d) {
    return (d % 5) < 2;
  }

  /** One token with an explicit position increment, offsets and payload. */
  record PosTok(String term, int posInc, int startOffset, int endOffset, byte[] payload) {}

  static final class CannedPosTokenStream extends TokenStream {
    private final List<PosTok> tokens;
    private int index = 0;
    private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
    private final PositionIncrementAttribute posIncAtt =
        addAttribute(PositionIncrementAttribute.class);
    private final OffsetAttribute offsetAtt = addAttribute(OffsetAttribute.class);
    private final PayloadAttribute payloadAtt = addAttribute(PayloadAttribute.class);

    CannedPosTokenStream(List<PosTok> tokens) {
      this.tokens = tokens;
    }

    @Override
    public boolean incrementToken() {
      if (index >= tokens.size()) {
        return false;
      }
      clearAttributes();
      PosTok t = tokens.get(index++);
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

  /** {@link #TERM}'s frequency in document {@code d}. */
  static int freqFor(int d) {
    // 1..5, cycling on a length **coprime with 256**. That is load-bearing:
    // with a period of 4 the occurrence count at every 256-document boundary
    // -- and, worse, at the 8192-document level-1 boundary -- lands exactly on
    // a `.pos` block boundary, so every `posBufferUpto` in the skip data would
    // be `0` and a reader that ignored the field entirely would still pass.
    // With a period of 5 the level-1 entry's `posBufferUpto` is 253.
    return 1 + (d % 5);
  }

  /**
   * Document {@code d}'s tokens: {@link #freqFor} occurrences of {@link
   * #TERM}, at positions spaced by {@code 1 + (d % 3)}, with offsets derived
   * from the position and a payload whose length cycles 0..2, followed --
   * in the documents {@link #hasSparse} names -- by one {@link #SPARSE_TERM}.
   */
  static List<PosTok> tokensFor(int d) {
    int freq = freqFor(d);
    int step = 1 + (d % 3);
    List<PosTok> toks = new ArrayList<>(freq + 1);
    int lastPos = -1;
    for (int i = 0; i < freq; i++) {
      int pos = (d % 7) + i * step;
      int posInc = (lastPos < 0) ? pos + 1 : pos - lastPos;
      lastPos = pos;
      int start = pos * 3;
      int end = start + 2 + (d % 5);
      int payloadLength = (pos + d) % 3;
      byte[] payload = null;
      if (payloadLength > 0) {
        payload = new byte[payloadLength];
        java.util.Arrays.fill(payload, (byte) (d % 251));
      }
      toks.add(new PosTok(TERM, posInc, start, end, payload));
    }
    if (hasSparse(d)) {
      // One occurrence, after all of TERM's, with its own offsets and a
      // payload on every other document.
      int pos = lastPos + 2;
      byte[] payload = (d % 2 == 0) ? new byte[] {(byte) (d % 251), 0x5A} : null;
      toks.add(new PosTok(SPARSE_TERM, 2, pos * 3, pos * 3 + 4, payload));
    }
    return toks;
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("postings_skip_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    FieldType posType = new FieldType();
    posType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    posType.setTokenized(true);
    posType.freeze();

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int d = 0; d < DOC_FREQ; d++) {
          Document doc = new Document();
          doc.add(new Field("pskip", new CannedPosTokenStream(tokensFor(d)), posType));
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
      String fnmFileName = null, siFileName = null, docFileName = null;
      String posFileName = null, payFileName = null;
      for (String f : sci.info.files()) {
        if (f.endsWith(".tim")) timFileName = f;
        if (f.endsWith(".tip")) tipFileName = f;
        if (f.endsWith(".tmd")) tmdFileName = f;
        if (f.endsWith(".fnm")) fnmFileName = f;
        if (f.endsWith(".si")) siFileName = f;
        if (f.endsWith(".doc")) docFileName = f;
        if (f.endsWith(".pos")) posFileName = f;
        if (f.endsWith(".pay")) payFileName = f;
      }
      if (timFileName == null
          || tipFileName == null
          || tmdFileName == null
          || fnmFileName == null
          || siFileName == null
          || docFileName == null
          || posFileName == null
          || payFileName == null) {
        throw new AssertionError("missing a codec file, files=" + sci.info.files());
      }

      dump(dir, timFileName, out);
      dump(dir, tipFileName, out);
      dump(dir, tmdFileName, out);
      dump(dir, fnmFileName, out);
      dump(dir, siFileName, out);
      dump(dir, docFileName, out);
      dump(dir, posFileName, out);
      dump(dir, payFileName, out);

      StringBuilder m = new StringBuilder();
      m.append("tim_file_name=").append(timFileName).append('\n');
      m.append("tip_file_name=").append(tipFileName).append('\n');
      m.append("tmd_file_name=").append(tmdFileName).append('\n');
      m.append("fnm_file_name=").append(fnmFileName).append('\n');
      m.append("si_file_name=").append(siFileName).append('\n');
      m.append("doc_file_name=").append(docFileName).append('\n');
      m.append("pos_file_name=").append(posFileName).append('\n');
      m.append("pay_file_name=").append(payFileName).append('\n');
      String prefix = sci.info.name + "_";
      String segmentSuffix = timFileName.substring(prefix.length(), timFileName.length() - 4);
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("segment_suffix=").append(segmentSuffix).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');
      m.append("term=").append(TERM).append('\n');
      m.append("sparse_term=").append(SPARSE_TERM).append('\n');

      // The level-1 entry's own `posBufferUpto`, derived the way the writer
      // derives it: the occurrence count at the 8192-document span boundary,
      // modulo the 256-occurrence `.pos` block size. Emitted so the Rust test
      // can *assert the fixture is non-degenerate* -- a zero here would mean
      // a reader that never read the field would still pass every check
      // below, which is exactly the hole this fixture existed to close and
      // did not, in its first revision.
      long level1Occurrences = 0;
      for (int d = 0; d < 32 * 256; d++) {
        level1Occurrences += freqFor(d);
      }
      m.append("level1_occurrences=").append(level1Occurrences).append('\n');
      m.append("level1_pos_buffer_upto=").append(level1Occurrences % 256).append('\n');

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        Terms terms = leaf.terms("pskip");
        TermsEnum te = terms.iterator();
        if (!te.seekExact(new BytesRef(TERM))) {
          throw new AssertionError("the term this fixture exists for is missing");
        }
        m.append("docFreq=").append(te.docFreq()).append('\n');
        m.append("totalTermFreq=").append(te.totalTermFreq()).append('\n');

        TermsEnum ste = terms.iterator();
        if (!ste.seekExact(new BytesRef(SPARSE_TERM))) {
          throw new AssertionError("the sparse term is missing");
        }
        m.append("sparse_docFreq=").append(ste.docFreq()).append('\n');
        m.append("sparse_totalTermFreq=").append(ste.totalTermFreq()).append('\n');

        // Documents chosen to bracket every structural boundary of the .doc
        // stream: the first and last, either side of the single level-1
        // entry's span end (8192), either side of several level-0 block ends
        // (256, 512, ..., which are also where the pos/pay skip records sit),
        // and the first document of the trailing group-varint tail.
        List<Integer> sampled = new ArrayList<>();
        sampled.add(0);
        sampled.add(1);
        for (int b = 256; b < DOC_FREQ; b += 256) {
          sampled.add(b - 1);
          sampled.add(b);
        }
        sampled.add(8191);
        sampled.add(8192);
        sampled.add(8193);
        sampled.add(DOC_FREQ - 1);
        // Plus an irregular stride, so the sample is not only boundaries.
        for (int d = 37; d < DOC_FREQ; d += 613) {
          sampled.add(d);
        }
        sampled = sampled.stream().distinct().sorted().toList();

        StringBuilder sampledCsv = new StringBuilder();
        for (int d : sampled) {
          if (sampledCsv.length() > 0) {
            sampledCsv.append(',');
          }
          sampledCsv.append(d);
        }
        m.append("sampled_docs=").append(sampledCsv).append('\n');

        // One fresh PostingsEnum per sampled document, each driven by a
        // single advance(doc) -- which is exactly the shape that uses the
        // skip data, and exactly the shape this port's
        // `read_occurrences_for_doc` implements. Reusing one enum and
        // advancing forward through the sample would instead exercise the
        // sequential path.
        for (int d : sampled) {
          TermsEnum te2 = terms.iterator();
          if (!te2.seekExact(new BytesRef(TERM))) {
            throw new AssertionError("term vanished");
          }
          PostingsEnum postings = te2.postings(null, PostingsEnum.ALL);
          if (postings.advance(d) != d) {
            throw new AssertionError("every document contains the term; d=" + d);
          }
          int freq = postings.freq();
          StringBuilder occurrences = new StringBuilder();
          for (int k = 0; k < freq; k++) {
            int pos = postings.nextPosition();
            if (occurrences.length() > 0) {
              occurrences.append(';');
            }
            BytesRef payload = postings.getPayload();
            occurrences
                .append(pos)
                .append(',')
                .append(postings.startOffset())
                .append(',')
                .append(postings.endOffset())
                .append(',')
                .append(
                    payload == null
                        ? "NONE"
                        : hex(payload.bytes, payload.offset, payload.length));
          }
          m.append("doc.").append(d).append(".freq=").append(freq).append('\n');
          m.append("doc.").append(d).append(".occurrences=").append(occurrences).append('\n');
        }

        // The sparse term, over the same sample: `advance(d)` for a document
        // the term is *not* in must land past it, and its `.doc` blocks are
        // packed-FOR or bit-set rather than the all-consecutive marker.
        StringBuilder sparseSampled = new StringBuilder();
        for (int d : sampled) {
          if (sparseSampled.length() > 0) {
            sparseSampled.append(',');
          }
          sparseSampled.append(d);
          TermsEnum te2 = terms.iterator();
          if (!te2.seekExact(new BytesRef(SPARSE_TERM))) {
            throw new AssertionError("sparse term vanished");
          }
          PostingsEnum postings = te2.postings(null, PostingsEnum.ALL);
          int landed = postings.advance(d);
          m.append("sparse.").append(d).append(".advance=").append(landed).append('\n');
          if (landed == PostingsEnum.NO_MORE_DOCS) {
            continue;
          }
          int freq = postings.freq();
          StringBuilder occurrences = new StringBuilder();
          for (int k = 0; k < freq; k++) {
            int pos = postings.nextPosition();
            if (occurrences.length() > 0) {
              occurrences.append(';');
            }
            BytesRef payload = postings.getPayload();
            occurrences
                .append(pos)
                .append(',')
                .append(postings.startOffset())
                .append(',')
                .append(postings.endOffset())
                .append(',')
                .append(
                    payload == null
                        ? "NONE"
                        : hex(payload.bytes, payload.offset, payload.length));
          }
          m.append("sparse.").append(d).append(".freq=").append(freq).append('\n');
          m.append("sparse.").append(d).append(".occurrences=").append(occurrences).append('\n');
        }
      }

      Files.writeString(out.resolve("manifest.properties"), m.toString());
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
    return hex(b, 0, b.length);
  }

  static String hex(byte[] b, int offset, int length) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < length; i++) sb.append(String.format("%02x", b[offset + i]));
    return sb.toString();
  }
}
