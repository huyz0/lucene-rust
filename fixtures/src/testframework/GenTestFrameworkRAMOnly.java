import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HexFormat;
import java.util.Map;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.analysis.CannedTokenStream;
import org.apache.lucene.tests.analysis.Token;
import org.apache.lucene.tests.codecs.ramonly.RAMOnlyPostingsFormat;
import org.apache.lucene.util.*;

/** RAM-only state must be observed while its actual Java format instance is alive. */
public class GenTestFrameworkRAMOnly {
  static String hex(BytesRef b) {
    return b == null ? "-" : HexFormat.of().formatHex(b.bytes, b.offset, b.offset + b.length);
  }

  static String postings(Fields fields, int flags) throws Exception {
    StringBuilder out = new StringBuilder();
    TermsEnum terms = fields.terms("body").iterator();
    while (terms.next() != null) {
      out.append(hex(terms.term())).append('\t').append(terms.docFreq())
          .append('\t').append(terms.totalTermFreq());
      PostingsEnum p = terms.postings(null, flags);
      while (p.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
        out.append('\t').append(p.docID()).append(':').append(p.freq());
        for (int i = 0; i < p.freq(); i++) {
          out.append(':').append(p.nextPosition()).append('/').append(hex(p.getPayload()));
        }
      }
      out.append('\n');
    }
    return out.toString();
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("ramonly");
    generate(root, true);
    generate(root.resolve("no-payloads"), false);
  }

  static void generate(Path root, boolean payloads) throws Exception {
    Files.createDirectories(root);
    try (Directory source = new ByteBuffersDirectory(); Directory target = new ByteBuffersDirectory()) {
      FieldType type = new FieldType(TextField.TYPE_NOT_STORED);
      type.setOmitNorms(true);
      type.freeze();
      try (IndexWriter writer = new IndexWriter(source, new IndexWriterConfig().setUseCompoundFile(false))) {
        for (int i = 0; i < 6; i++) {
          Token first = new Token("common", 1, 0, 6);
          first.setPayload(payloads && i % 2 == 0 ? new BytesRef(new byte[] {(byte) i, (byte) 255}) : null);
          Token second = new Token("common", 0, 7, 13);
          second.setPayload(payloads ? new BytesRef(new byte[0]) : null);
          Token unicode = new Token(i % 2 == 0 ? "\uE000" : "\uD800\uDC00", 2, 14, 16);
          Token last = new Token(i % 2 == 0 ? "a" : "z", 1, 17, 18);
          Document doc = new Document();
          doc.add(new Field("body", new CannedTokenStream(first, second, unicode, last), type));
          writer.addDocument(doc);
        }
        writer.addDocument(new Document());
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(source)) {
        SegmentReader leaf = (SegmentReader) reader.leaves().get(0).reader();
        Files.writeString(root.resolve("input.tsv"), postings(leaf.getPostingsReader(), PostingsEnum.ALL));
        // Record exactly what Lucene104 exposes for RAMOnly's requested flags.
        Files.writeString(root.resolve("selected.tsv"), postings(leaf.getPostingsReader(),
            payloads ? PostingsEnum.PAYLOADS : PostingsEnum.NONE));
        byte[] id = new byte[16];
        java.util.Arrays.fill(id, (byte) 7);
        SegmentInfo si = new SegmentInfo(target, Version.LATEST, Version.LATEST, "_0",
            leaf.maxDoc(), false, false, new Lucene104Codec(), Map.of(), id, Map.of(), null);
        RAMOnlyPostingsFormat format = new RAMOnlyPostingsFormat();
        for (int slot = 0; slot < 2; slot++) {
          String suffix = "RAMOnly_" + slot;
          SegmentWriteState write = new SegmentWriteState(InfoStream.NO_OUTPUT, target, si,
              leaf.getFieldInfos(), null, IOContext.DEFAULT, suffix);
          try (FieldsConsumer consumer = format.fieldsConsumer(write)) {
            consumer.write(leaf.getPostingsReader(), null);
          }
          SegmentReadState read = new SegmentReadState(target, si, leaf.getFieldInfos(), IOContext.DEFAULT, suffix);
          try (FieldsProducer producer = format.fieldsProducer(read)) {
            producer.checkIntegrity();
            Files.writeString(root.resolve("output" + slot + ".tsv"), postings(producer, PostingsEnum.ALL));
            Terms terms = producer.terms("body");
            Files.writeString(root.resolve("stats.tsv"), terms.size() + "\t" + terms.getSumDocFreq()
                + "\t" + terms.getSumTotalTermFreq() + "\t" + terms.getDocCount() + "\n");
            StringBuilder probes = new StringBuilder();
            for (String value : new String[] {"", "a", "b", "common", "z", "zzzz", "\uE000", "\uD800\uDC00", "\uFFFF"}) {
              TermsEnum e = terms.iterator();
              BytesRef probe = new BytesRef(value);
              probes.append(hex(probe)).append('\t').append(e.seekCeil(probe))
                  .append('\t').append(hex(e.term())).append('\t').append(hex(e.next())).append('\n');
            }
            Files.writeString(root.resolve("seeks.tsv"), probes);
          }
          if (new RAMOnlyPostingsFormat().fieldsProducer(read) != null) {
            throw new AssertionError("state must belong to this format instance");
          }
        }
        si.getCodec().fieldInfosFormat().write(target, si, "", leaf.getFieldInfos(), IOContext.DEFAULT);
        for (String file : target.listAll()) {
          try (IndexInput input = target.openInput(file, IOContext.DEFAULT)) {
            byte[] bytes = new byte[Math.toIntExact(input.length())];
            input.readBytes(bytes, 0, bytes.length);
            Files.write(root.resolve(file), bytes);
          }
        }
      }
    }
  }
}
