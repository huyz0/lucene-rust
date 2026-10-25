import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.uniformsplit.Rot13CypherTestUtil;
import org.apache.lucene.tests.codecs.uniformsplit.UniformSplitRot13PostingsFormat;
import org.apache.lucene.tests.codecs.uniformsplit.sharedterms.STUniformSplitRot13PostingsFormat;
import org.apache.lucene.util.*;

/** Actual test formats, including Java's shared static encoding flags. */
public class GenTestFrameworkRot13 {
  static String flags() {
    return UniformSplitRot13PostingsFormat.encoderCalled + "\t"
        + UniformSplitRot13PostingsFormat.decoderCalled + "\t"
        + UniformSplitRot13PostingsFormat.blocksEncoded + "\t"
        + UniformSplitRot13PostingsFormat.fieldsMetadataEncoded + "\t"
        + UniformSplitRot13PostingsFormat.dictionaryEncoded;
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("rot13");
    Files.createDirectories(root);
    StringBuilder cypher = new StringBuilder();
    for (int length : new int[] {0, 1, 6, 7, 255, 256, 257, 4096}) {
      byte[] plain = new byte[length];
      for (int i = 0; i < length; i++) plain[i] = (byte) (i * 37 + 123);
      byte[] encoded = Rot13CypherTestUtil.encode(new ByteArrayDataInput(plain), length);
      byte[] decoded = Rot13CypherTestUtil.decode(new ByteArrayDataInput(encoded), encoded.length);
      if (!Arrays.equals(plain, decoded)) throw new AssertionError("cypher roundtrip");
      ByteBuffersDataOutput output = new ByteBuffersDataOutput();
      var writable = Rot13CypherTestUtil.getBlockEncoder().encode(new ByteArrayDataInput(plain), length);
      writable.writeTo(output);
      if (writable.size() != encoded.length || !Arrays.equals(encoded, output.toArrayCopy()))
        throw new AssertionError("encoder wrapper");
      if (!new BytesRef(plain).equals(Rot13CypherTestUtil.getBlockDecoder()
          .decode(new ByteArrayDataInput(encoded), encoded.length))) throw new AssertionError("decoder wrapper");
      cypher.append(HexFormat.of().formatHex(plain)).append('\t')
          .append(HexFormat.of().formatHex(encoded)).append('\n');
    }
    Files.writeString(root.resolve("cypher.tsv"), cypher);
    for (int count : new int[] {0, 1, 500}) generate(root, count);
  }

  static void generate(Path root, int count) throws Exception {
    try (Directory source = new ByteBuffersDirectory()) {
      FieldType type = new FieldType(TextField.TYPE_NOT_STORED);
      type.setOmitNorms(true);
      type.freeze();
      try (IndexWriter writer = new IndexWriter(source,
          new IndexWriterConfig().setUseCompoundFile(false))) {
        for (int i = 0; i < Math.max(count, 1); i++) {
          Document doc = new Document();
          doc.add(new Field("body", "common common t" + (i % 90) + " t" + i, type));
          doc.add(new Field("title", "common x x t" + (i % 30), type));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(source)) {
        SegmentReader leaf = (SegmentReader) reader.leaves().get(0).reader();
        for (UniformSplitRot13PostingsFormat format : new UniformSplitRot13PostingsFormat[] {
            new UniformSplitRot13PostingsFormat(), new STUniformSplitRot13PostingsFormat()}) {
          Path out = root.resolve(format.getName()).resolve(Integer.toString(count));
          Files.createDirectories(out);
          try (Directory target = new ByteBuffersDirectory()) {
            byte[] id = new byte[16];
            Arrays.fill(id, (byte) 7);
            SegmentInfo si = new SegmentInfo(target, Version.LATEST, Version.LATEST,
                "_0", count, false, false, new Lucene104Codec(), Map.of(), id, Map.of(), null);
            String suffix = format.getName() + "_0";
            SegmentWriteState write = new SegmentWriteState(InfoStream.NO_OUTPUT, target, si,
                leaf.getFieldInfos(), null, IOContext.DEFAULT, suffix);
            UniformSplitRot13PostingsFormat.resetEncodingFlags();
            try (FieldsConsumer consumer = format.fieldsConsumer(write)) {
              Fields fields = count == 0 ? new Fields() {
                public Iterator<String> iterator() { return Collections.emptyIterator(); }
                public Terms terms(String name) { return null; }
                public int size() { return 0; }
              } : leaf.getPostingsReader();
              consumer.write(fields, null);
            }
            StringBuilder expectedFlags = new StringBuilder("write\t" + flags() + "\n");
            si.getCodec().fieldInfosFormat().write(target, si, "", leaf.getFieldInfos(), IOContext.DEFAULT);
            SegmentReadState read = new SegmentReadState(target, si, leaf.getFieldInfos(), IOContext.DEFAULT, suffix);
            StringBuilder postings = new StringBuilder();
            StringBuilder stats = new StringBuilder();
            try (FieldsProducer producer = format.fieldsProducer(read)) {
              expectedFlags.append("open\t").append(flags()).append('\n');
              producer.checkIntegrity();
              for (String name : producer) {
                Terms terms = producer.terms(name);
                stats.append(name).append('\t').append(terms.size()).append('\t')
                    .append(terms.getSumDocFreq()).append('\t').append(terms.getSumTotalTermFreq())
                    .append('\t').append(terms.getDocCount()).append('\n');
                TermsEnum walk = terms.iterator();
                while (walk.next() != null) {
                  BytesRef term = walk.term();
                  postings.append(name).append('\t')
                      .append(HexFormat.of().formatHex(term.bytes, term.offset, term.offset + term.length))
                      .append('\t').append(walk.docFreq()).append('\t').append(walk.totalTermFreq());
                  PostingsEnum p = walk.postings(null, PostingsEnum.POSITIONS);
                  while (p.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                    postings.append('\t').append(p.docID()).append(':').append(p.freq());
                    for (int i = 0; i < p.freq(); i++) postings.append(':').append(p.nextPosition());
                  }
                  postings.append('\n');
                }
              }
            }
            expectedFlags.append("walk\t").append(flags()).append('\n');
            Files.writeString(out.resolve("flags.tsv"), expectedFlags);
            Files.writeString(out.resolve("postings.tsv"), postings);
            Files.writeString(out.resolve("stats.tsv"), stats);
            for (String file : target.listAll()) {
              try (IndexInput input = target.openInput(file, IOContext.DEFAULT)) {
                byte[] bytes = new byte[Math.toIntExact(input.length())];
                input.readBytes(bytes, 0, bytes.length);
                Files.write(out.resolve(file), bytes);
              }
            }
          }
        }
      }
    }
  }
}
