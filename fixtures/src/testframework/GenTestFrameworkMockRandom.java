import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.mockrandom.MockRandomPostingsFormat;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.*;

/** Every actual MockRandom dictionary/selector branch, with stable segment ids. */
public class GenTestFrameworkMockRandom {
  static String selection(long seed) {
    Random r = new Random(seed);
    r.nextInt(); // writer's unused buffer-size draw
    return switch (r.nextInt(4)) {
      case 0 -> "fst";
      case 1 -> "blocktree:" + blockSizes(r);
      case 3 -> "ords:" + blockSizes(r);
      case 2 -> {
        if (r.nextBoolean()) yield "fixed:" + TestUtil.nextInt(r, 1, 100);
        yield switch (r.nextInt(3)) {
          case 0 -> "every-n:" + TestUtil.nextInt(r, 1, 100);
          case 1 -> "doc-freq:" + TestUtil.nextInt(r, 2, 100) + ":" + TestUtil.nextInt(r, 1, 100);
          default -> "random-gap:" + r.nextLong() + ":" + TestUtil.nextInt(r, 2, 40);
        };
      }
      default -> throw new AssertionError();
    };
  }

  static String blockSizes(Random r) {
    int min = TestUtil.nextInt(r, 2, 100);
    int max = Math.max(2, (min - 1) * 2 + r.nextInt(100));
    return min + ":" + max;
  }

  static long firstSeed(long callerSeed) {
    Random r = new Random(new Random(callerSeed).nextLong());
    TestUtil.nextInt(r, 2, 10);
    return r.nextLong();
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("mockrandom");
    Map<String, Long> seeds = new TreeMap<>();
    for (long seed = 0; seeds.size() < 7; seed++) {
      seeds.putIfAbsent(selection(firstSeed(seed)).split(":")[0], seed);
    }
    try (Directory source = new ByteBuffersDirectory()) {
      FieldType type = new FieldType(TextField.TYPE_NOT_STORED);
      type.setOmitNorms(true);
      type.freeze();
      try (IndexWriter writer = new IndexWriter(source,
          new IndexWriterConfig().setUseCompoundFile(false))) {
        for (int i = 0; i < 500; i++) {
          Document doc = new Document();
          doc.add(new Field("body", "common common t" + (i % 90) + " t" + i, type));
          doc.add(new Field("title", "x x t" + (i % 30), type));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(source)) {
        SegmentReader leaf = (SegmentReader) reader.leaves().get(0).reader();
        for (var entry : seeds.entrySet()) {
          Random caller = new Random(entry.getValue());
          MockRandomPostingsFormat format = new MockRandomPostingsFormat(caller);
          Random replay = new Random(new Random(entry.getValue()).nextLong());
          long callerNext = caller.nextLong();
          for (int slot = 0; slot < 2; slot++) {
            Path out = root.resolve(entry.getKey()).resolve(Integer.toString(slot));
            Files.createDirectories(out);
            int skip = TestUtil.nextInt(replay, 2, 10);
            long seed = replay.nextLong();
            String selected = selection(seed);
            String suffix = "MockRandom_" + slot;
            try (Directory target = new ByteBuffersDirectory()) {
              byte[] id = new byte[16];
              Arrays.fill(id, (byte) 7);
              SegmentInfo si = new SegmentInfo(target, Version.LATEST, Version.LATEST,
                  "_0", leaf.maxDoc(), false, false, new Lucene104Codec(), Map.of(), id, Map.of(), null);
              SegmentWriteState write = new SegmentWriteState(InfoStream.NO_OUTPUT, target, si,
                  leaf.getFieldInfos(), null, IOContext.DEFAULT, suffix);
              try (FieldsConsumer consumer = format.fieldsConsumer(write)) {
                String expectedClass = switch (selected.split(":")[0]) {
                  case "fst" -> "FSTTermsWriter";
                  case "blocktree" -> "Lucene103BlockTreeTermsWriter";
                  case "ords" -> "OrdsBlockTreeTermsWriter";
                  default -> "BlockTermsWriter";
                };
                if (!consumer.getClass().getSimpleName().equals(expectedClass)) {
                  throw new AssertionError(consumer.getClass() + " vs " + expectedClass);
                }
                consumer.write(leaf.getPostingsReader(), null);
              }
              try (ChecksumIndexInput input = target.openChecksumInput("_0_" + suffix + ".sd")) {
                CodecUtil.checkIndexHeader(input, "MockRandomSeed", 0, 0, id, suffix);
                if (input.readLong() != seed) throw new AssertionError("seed replay diverged");
                CodecUtil.checkFooter(input);
              }
              Files.writeString(out.resolve("selection.tsv"), entry.getValue() + "\t" + callerNext
                  + "\t" + skip + "\t" + seed + "\t" + selected + "\n");
              si.getCodec().fieldInfosFormat().write(target, si, "", leaf.getFieldInfos(), IOContext.DEFAULT);
              SegmentReadState read = new SegmentReadState(target, si, leaf.getFieldInfos(), IOContext.DEFAULT, suffix);
              StringBuilder postings = new StringBuilder();
              StringBuilder stats = new StringBuilder();
              try (FieldsProducer producer = new MockRandomPostingsFormat().fieldsProducer(read)) {
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
}
