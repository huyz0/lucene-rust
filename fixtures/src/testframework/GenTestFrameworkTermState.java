import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.TermState;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * Differential fixture for {@code TermsEnum.termState()} and {@code seekExact(term, state)} on
 * the block-tree terms dictionary (M14.h, {@code RandomPostingsTester}'s {@code TERM_STATE}): a
 * one-segment index of {@value #DOCS} standard-analyzed documents drawn from {@code new
 * Random(42)}, every term's state saved by one enum, then {@value #SEEKS} seeks of fresh enums
 * through a saved state drawn from {@code new Random(7)}.
 *
 * <p>Writes {@code term_state.txt}: per seek, {@code <term> <docFreq> <totalTermFreq> |} and each
 * document as {@code doc:freq:pos,pos...}, then {@code next=<term>} (or {@code next=null}) when
 * the draw calls {@code next()} after the seek.
 *
 * <p>Usage: java GenTestFrameworkTermState <outdir>
 */
public class GenTestFrameworkTermState {
  static final int DOCS = 300;
  static final int SEEKS = 400;

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    StringBuilder sb = new StringBuilder();
    try (ByteBuffersDirectory dir = new ByteBuffersDirectory()) {
      IndexWriterConfig c = new IndexWriterConfig(new StandardAnalyzer());
      c.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, c)) {
        Random r = new Random(42);
        for (int i = 0; i < DOCS; i++) {
          StringBuilder body = new StringBuilder();
          int words = 1 + r.nextInt(12);
          for (int j = 0; j < words; j++) {
            body.append(" w").append(r.nextInt(1 + r.nextInt(300)));
          }
          Document d = new Document();
          d.add(new TextField("body", body.toString(), Field.Store.NO));
          w.addDocument(d);
        }
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        Terms terms = leaf.terms("body");
        List<BytesRef> saved = new ArrayList<>();
        List<TermState> states = new ArrayList<>();
        TermsEnum all = terms.iterator();
        for (BytesRef t = all.next(); t != null; t = all.next()) {
          saved.add(BytesRef.deepCopyOf(t));
          states.add(all.termState());
        }
        Random r = new Random(7);
        for (int i = 0; i < SEEKS; i++) {
          int idx = r.nextInt(saved.size());
          TermsEnum te = terms.iterator();
          te.seekExact(saved.get(idx), states.get(idx));
          sb.append(te.term().utf8ToString())
              .append(' ')
              .append(te.docFreq())
              .append(' ')
              .append(te.totalTermFreq())
              .append(" |");
          PostingsEnum pe = te.postings(null, PostingsEnum.POSITIONS);
          for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
            sb.append(' ').append(doc).append(':').append(pe.freq()).append(':');
            for (int p = 0; p < pe.freq(); p++) {
              sb.append(p == 0 ? "" : ",").append(pe.nextPosition());
            }
          }
          if (r.nextBoolean()) {
            BytesRef next = te.next();
            sb.append(" next=").append(next == null ? "null" : next.utf8ToString());
          }
          sb.append('\n');
        }
      }
    }
    Files.writeString(out.resolve("term_state.txt"), sb.toString(), StandardCharsets.UTF_8);
  }
}
