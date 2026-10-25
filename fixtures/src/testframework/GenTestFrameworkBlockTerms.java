import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HexFormat;
import java.util.Map;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.blockterms.*;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;

/** Actual test-framework formats, with fixed segment ids so --check is byte-exact. */
public class GenTestFrameworkBlockTerms {
  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("blockterms");
    PostingsFormat[] formats = {
      new LuceneFixedGap(), new LuceneVarGapFixedInterval(),
      new LuceneVarGapDocFreqInterval(6, 24),
      new LuceneFixedGap(7), new LuceneVarGapFixedInterval(11),
      new LuceneVarGapDocFreqInterval()
    };
    String[] cases = {"LuceneFixedGap", "LuceneVarGapFixedInterval",
      "LuceneVarGapDocFreqInterval", "CustomFixedGap",
      "CustomVarGapFixedInterval", "DefaultVarGapDocFreqInterval"};
    generate(root, formats, cases, 500);
  }

  /** Shared direct-format fixture path; source postings still come from IndexWriter. */
  static void generate(Path root, PostingsFormat[] formats, String[] cases, int docCount)
      throws Exception {
    try (Directory source = new ByteBuffersDirectory()) {
      FieldType type = new FieldType(TextField.TYPE_NOT_STORED);
      type.setOmitNorms(true);
      type.freeze();
      try (IndexWriter writer = new IndexWriter(source,
          new IndexWriterConfig().setUseCompoundFile(false))) {
        for (int i = 0; i < docCount; i++) {
          Document doc = new Document();
          doc.add(new Field("body", "common common t" + (i % 90) + " t" + i, type));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(source)) {
        SegmentReader leaf = (SegmentReader) reader.leaves().get(0).reader();
        for (int caseIndex = 0; caseIndex < formats.length; caseIndex++) {
          PostingsFormat format = formats[caseIndex];
          Path out = root.resolve(cases[caseIndex]);
          Files.createDirectories(out);
          try (Directory target = new ByteBuffersDirectory()) {
            byte[] id = new byte[16];
            java.util.Arrays.fill(id, (byte) 7);
            SegmentInfo si = new SegmentInfo(target, Version.LATEST, Version.LATEST,
                "_0", leaf.maxDoc(), false, false, new Lucene104Codec(),
                Map.of(), id, Map.of(), null);
            String suffix = format.getName() + "_0";
            SegmentWriteState write = new SegmentWriteState(InfoStream.NO_OUTPUT,
                target, si, leaf.getFieldInfos(), null, IOContext.DEFAULT, suffix);
            try (FieldsConsumer consumer = format.fieldsConsumer(write)) {
              consumer.write(leaf.getPostingsReader(), null);
            }
            si.getCodec().fieldInfosFormat().write(target, si, "",
                leaf.getFieldInfos(), IOContext.DEFAULT);
            SegmentReadState read = new SegmentReadState(target, si,
                leaf.getFieldInfos(), IOContext.DEFAULT, suffix);
            StringBuilder expected = new StringBuilder();
            try (FieldsProducer producer = format.fieldsProducer(read)) {
              producer.checkIntegrity();
              TermsEnum terms = producer.terms("body").iterator();
              while (terms.next() != null) {
                expected.append(HexFormat.of().formatHex(terms.term().bytes,
                    terms.term().offset, terms.term().offset + terms.term().length));
                expected.append('\t').append(terms.docFreq());
                PostingsEnum postings = terms.postings(null, PostingsEnum.POSITIONS);
                while (postings.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                  expected.append('\t').append(postings.docID()).append(':')
                      .append(postings.freq());
                  for (int p = 0; p < postings.freq(); p++) {
                    expected.append(':').append(postings.nextPosition());
                  }
                }
                expected.append('\n');
              }
            }
            Files.writeString(out.resolve("postings.tsv"), expected);
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
