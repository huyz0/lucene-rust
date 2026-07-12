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

  /** One token with an explicit position increment, offsets, and optional payload. */
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

    FieldType bigType = new FieldType();
    bigType.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    bigType.setTokenized(true);
    bigType.freeze();

    FieldType posType = new FieldType();
    posType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    posType.setTokenized(true);
    posType.freeze();

    // "big": a single term ("everywhere") appearing in BIG_DOC_FREQ (300)
    // separate documents with a varying (1..4) per-doc frequency -- forces
    // Lucene104PostingsWriter past ForUtil.BLOCK_SIZE (256), producing one
    // full PFOR-encoded block plus a group-varint tail block on the wire
    // (see crates/lucene-codecs/src/postings.rs's module doc). These
    // documents deliberately carry no "id"/"body" field, so the existing
    // single-leaf-block assumption for those two fields' term dictionaries
    // (see blocktree.rs's module doc) is untouched.
    final int bigDocFreq = 300;

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
        for (int i = 0; i < bigDocFreq; i++) {
          Document doc = new Document();
          int freq = 1 + (i % 4);
          String[] toks = new String[freq];
          java.util.Arrays.fill(toks, "everywhere");
          doc.add(new Field("big", new CannedTokenStream(List.of(toks)), bigType));
          w.addDocument(doc);
        }

        // "pos": positions/offsets/payloads on real postings (not term
        // vectors -- exercises Lucene104PostingsWriter's .pos/.pay path,
        // reusing GenTermVectors.java's hand-built-TokenStream technique so
        // every occurrence's exact position/offset/payload is known.
        // doc5: "alpha" (pos 0, offset [0,5), payload 0xAA), "beta" (pos 1,
        //       offset [6,10), no payload) -- alpha docFreq contribution 1,
        //       beta docFreq contribution 1.
        // doc6: "alpha" (pos 0, offset [0,5), no payload), "alpha" (pos 1,
        //       offset [6,11), payload 0xBB,0xCC) -- same term repeated in
        //       one doc with a payload on only the second occurrence,
        //       exercising the vint tail's payload-length-change bit.
        // "alpha": docFreq=2, totalTermFreq=3 (1 + 2). "beta": docFreq=1,
        // totalTermFreq=1.
        Document doc5 = new Document();
        doc5.add(
            new Field(
                "pos",
                new CannedPosTokenStream(
                    List.of(
                        new PosTok("alpha", 1, 0, 5, new byte[] {(byte) 0xAA}),
                        new PosTok("beta", 1, 6, 10, null))),
                posType));
        w.addDocument(doc5);

        Document doc6 = new Document();
        doc6.add(
            new Field(
                "pos",
                new CannedPosTokenStream(
                    List.of(
                        new PosTok("alpha", 1, 0, 5, null),
                        new PosTok("alpha", 1, 6, 11, new byte[] {(byte) 0xBB, (byte) 0xCC}))),
                posType));
        w.addDocument(doc6);

        // "many": 400 distinct terms ("term0000".."term0399"), one per doc,
        // deliberately past the default minItemsInBlock=25/maxItemsInBlock=48
        // thresholds -- forces Lucene103BlockTreeTermsWriter to both split
        // the field's term dictionary across multiple `.tim` blocks (a
        // multi-child `.tip` trie, since a 4-digit zero-padded numeric suffix
        // spreads across many distinct leading bytes) *and* to floor-split
        // at least one of those prefix groups (some leading-byte groups have
        // 40 terms sharing that prefix, past maxItemsInBlock=48 once you
        // include the shared "term0" prefix's own bookkeeping) -- exercising
        // both new decode paths blocktree.rs added this slice, against real
        // Lucene bytes rather than only this port's own hand-built encoders.
        int manyCount = 400;
        String[] manyLookups = new String[manyCount];
        for (int i = 0; i < manyCount; i++) {
            manyLookups[i] = String.format("term%04d", i);
        }
        for (String term : manyLookups) {
            Document doc = new Document();
            doc.add(new Field("many", new CannedTokenStream(List.of(term)), idType));
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
      if (timFileName == null || tipFileName == null || tmdFileName == null) {
        throw new AssertionError("expected .tim/.tip/.tmd files, files=" + sci.info.files());
      }
      if (fnmFileName == null || siFileName == null) {
        throw new AssertionError("expected .fnm/.si files, files=" + sci.info.files());
      }
      if (docFileName == null) {
        throw new AssertionError("expected .doc file, files=" + sci.info.files());
      }
      if (posFileName == null || payFileName == null) {
        throw new AssertionError("expected .pos/.pay files, files=" + sci.info.files());
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
        appendFieldManifest(
            m,
            leaf,
            "big",
            new String[] {"everywhere", "zzz-missing"});
        appendPositionFieldManifest(
            m,
            leaf,
            "pos",
            new String[] {"alpha", "beta"});

        // Sample across the "many" field's range: first/last, several mid
        // values spanning different leading digits/bytes (so different trie
        // branches and, where the writer floor-split a prefix, different
        // floor sub-blocks get exercised), plus two absent lookups (one
        // between real terms, one past the end).
        appendFieldManifest(
            m,
            leaf,
            "many",
            new String[] {
              "term0000", "term0001", "term0037", "term0038", "term0099",
              "term0100", "term0150", "term0199", "term0200", "term0250",
              "term0299", "term0300", "term0350", "term0398", "term0399",
              "term0400-missing", "term9999-missing"
            });

        // Ordered enumeration ground truth: walk the whole "many" field via
        // real TermsEnum.next() (not seekExact) and dump the full sequence
        // of (term, docFreq, totalTermFreq) -- the hardest target for this
        // because it's multi-block/floor-split, so a correct dump proves
        // enumeration walks block/floor boundaries in the right order, not
        // just within one block. Also seekCeil ground truth: an exact match,
        // a between-terms ceiling match, before-the-first-term, and
        // after-the-last-term (END).
        appendEnumerationManifest(m, leaf, "many", out);
        appendSeekCeilManifest(m, leaf, "many", "term0037", "exact");
        appendSeekCeilManifest(m, leaf, "many", "term0037a", "ceiling");
        appendSeekCeilManifest(m, leaf, "many", "", "beforeFirst");
        appendSeekCeilManifest(m, leaf, "many", "zzzz", "afterLast");

        // Real PostingsEnum.advance(target) ground truth: "big"/"everywhere"
        // (docFreq=300, multi-block .doc) exercises advancing across a full
        // 256-doc block boundary into the group-varint tail; "body"/"cat"
        // (docFreq=2, single tail block) exercises the same targets on a
        // small single-block postings list.
        appendAdvanceManifest(m, leaf, "big", "everywhere");
        appendAdvanceManifest(m, leaf, "body", "cat");
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

        // Real PostingsEnum.nextDoc()/freq() ground truth for the postings
        // decode this fixture also verifies (DOCS_AND_FREQS mode) -- not just
        // the aggregate docFreq/totalTermFreq stats above.
        PostingsEnum postings = te.postings(null, PostingsEnum.FREQS);
        StringBuilder docs = new StringBuilder();
        StringBuilder freqs = new StringBuilder();
        int doc;
        while ((doc = postings.nextDoc()) != PostingsEnum.NO_MORE_DOCS) {
          if (docs.length() > 0) {
            docs.append(',');
            freqs.append(',');
          }
          docs.append(doc);
          freqs.append(postings.freq());
        }
        m.append(key).append(".postingsDocs=").append(docs).append('\n');
        m.append(key).append(".postingsFreqs=").append(freqs).append('\n');
      } else {
        m.append(key).append(".found=false\n");
      }
    }
  }

  /**
   * Like {@link #appendFieldManifest} but for a field with positions/offsets/
   * payloads: dumps real {@code PostingsEnum.nextPosition()}/{@code
   * startOffset()}/{@code endOffset()}/{@code getPayload()} ground truth for
   * every occurrence, in doc order, alongside the same docFreq/totalTermFreq/
   * postingsDocs/postingsFreqs manifest keys {@link #appendFieldManifest}
   * writes.
   */
  static void appendPositionFieldManifest(
      StringBuilder m, LeafReader leaf, String field, String[] lookups) throws IOException {
    Terms terms = leaf.terms(field);
    if (terms == null) {
      throw new AssertionError("expected terms for field " + field);
    }
    m.append("field.").append(field).append(".numTerms=").append(terms.size()).append('\n');

    TermsEnum te = terms.iterator();
    for (String lookup : lookups) {
      boolean found = te.seekExact(new BytesRef(lookup));
      String key = "field." + field + ".term." + lookup;
      if (!found) {
        m.append(key).append(".found=false\n");
        continue;
      }
      m.append(key).append(".found=true\n");
      m.append(key).append(".docFreq=").append(te.docFreq()).append('\n');
      m.append(key).append(".totalTermFreq=").append(te.totalTermFreq()).append('\n');

      PostingsEnum postings = te.postings(null, PostingsEnum.ALL);
      StringBuilder docs = new StringBuilder();
      StringBuilder freqs = new StringBuilder();
      StringBuilder occurrences = new StringBuilder();
      int doc;
      while ((doc = postings.nextDoc()) != PostingsEnum.NO_MORE_DOCS) {
        if (docs.length() > 0) {
          docs.append(',');
          freqs.append(',');
        }
        docs.append(doc);
        int freq = postings.freq();
        freqs.append(freq);
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
              .append(payload == null ? "NONE" : hex(payload.bytes, payload.offset, payload.length));
        }
      }
      m.append(key).append(".postingsDocs=").append(docs).append('\n');
      m.append(key).append(".postingsFreqs=").append(freqs).append('\n');
      m.append(key).append(".occurrences=").append(occurrences).append('\n');
    }
  }

  /**
   * Walks {@code field}'s entire term dictionary via real {@code
   * TermsEnum.next()} (never {@code seekExact}) and writes the ordered
   * sequence of {@code term\tdocFreq\ttotalTermFreq} lines to a sibling file
   * ({@code <field>.enumeration.tsv}) rather than inline as a manifest
   * property -- a 400-term field's dump doesn't fit comfortably as a single
   * properties-file value (embedded newlines break line-based parsing).
   */
  static void appendEnumerationManifest(StringBuilder m, LeafReader leaf, String field, Path out)
      throws IOException {
    Terms terms = leaf.terms(field);
    if (terms == null) {
      throw new AssertionError("expected terms for field " + field);
    }
    TermsEnum te = terms.iterator();
    StringBuilder enumeration = new StringBuilder();
    BytesRef term;
    int count = 0;
    while ((term = te.next()) != null) {
      enumeration
          .append(term.utf8ToString())
          .append('\t')
          .append(te.docFreq())
          .append('\t')
          .append(te.totalTermFreq())
          .append('\n');
      count++;
    }
    String fileName = field + ".enumeration.tsv";
    Files.writeString(out.resolve(fileName), enumeration.toString());
    m.append("field.").append(field).append(".enumeration.count=").append(count).append('\n');
    m.append("field.").append(field).append(".enumeration.file=").append(fileName).append('\n');
  }

  /**
   * Dumps real {@code TermsEnum.seekCeil()} ground truth for one target:
   * whether it found/status ({@code FOUND}/{@code NOT_FOUND}/{@code END}),
   * and if not {@code END}, the term/docFreq/totalTermFreq it landed on.
   */
  static void appendSeekCeilManifest(
      StringBuilder m, LeafReader leaf, String field, String target, String label)
      throws IOException {
    Terms terms = leaf.terms(field);
    if (terms == null) {
      throw new AssertionError("expected terms for field " + field);
    }
    TermsEnum te = terms.iterator();
    TermsEnum.SeekStatus status = te.seekCeil(new BytesRef(target));
    String key = "field." + field + ".seekCeil." + label;
    m.append(key).append(".status=").append(status).append('\n');
    if (status != TermsEnum.SeekStatus.END) {
      m.append(key).append(".term=").append(te.term().utf8ToString()).append('\n');
      m.append(key).append(".docFreq=").append(te.docFreq()).append('\n');
      m.append(key).append(".totalTermFreq=").append(te.totalTermFreq()).append('\n');
    }
  }

  /**
   * Dumps real {@code PostingsEnum.advance(target)} ground truth for a
   * variety of targets against one term: before the first doc, an exact
   * match, a target strictly between two docs, a target exactly on the
   * doc right after a match, the last doc, and a target past the last doc
   * (must return {@code NO_MORE_DOCS}). Each target gets a *fresh*
   * {@code PostingsEnum} (advance() only moves forward, so reusing one
   * enum across targets would make later targets depend on earlier ones).
   */
  static void appendAdvanceManifest(StringBuilder m, LeafReader leaf, String field, String term)
      throws IOException {
    Terms terms = leaf.terms(field);
    if (terms == null) {
      throw new AssertionError("expected terms for field " + field);
    }
    TermsEnum probe = terms.iterator();
    if (!probe.seekExact(new BytesRef(term))) {
      throw new AssertionError("expected term " + term + " in field " + field);
    }
    List<Integer> allDocs = new java.util.ArrayList<>();
    PostingsEnum p0 = probe.postings(null, PostingsEnum.FREQS);
    int d;
    while ((d = p0.nextDoc()) != PostingsEnum.NO_MORE_DOCS) {
      allDocs.add(d);
    }

    int first = allDocs.get(0);
    int mid = allDocs.get(allDocs.size() / 2);
    int last = allDocs.get(allDocs.size() - 1);
    int[] targets = {0, first, first + 1, mid, mid + 1, last, last + 1, last + 100000};

    StringBuilder sb = new StringBuilder();
    for (int target : targets) {
      TermsEnum te = terms.iterator();
      te.seekExact(new BytesRef(term));
      PostingsEnum p = te.postings(null, PostingsEnum.FREQS);
      int result = p.advance(target);
      if (sb.length() > 0) {
        sb.append(';');
      }
      if (result == PostingsEnum.NO_MORE_DOCS) {
        sb.append(target).append(":NO_MORE_DOCS");
      } else {
        sb.append(target).append(':').append(result).append(',').append(p.freq());
      }
    }
    String key = "field." + field + ".term." + term + ".advance";
    m.append(key).append(".results=").append(sb).append('\n');
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
