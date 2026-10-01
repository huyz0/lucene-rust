import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.nio.file.Path;

/**
 * Verifies {@code crates/lucene-index/examples/write_token_attributes_fixture.rs} (the documents,
 * analyzer and similarity of {@code GenTokenAttributes}): in {@code <dir>/flushed} and {@code
 * <dir>/merged} (one segment each) field {@code pay} stores payloads and every occurrence carries
 * the one {@code GenTokenAttributes.Tag} gives its term -- {@code [length, first char]} before
 * {@code m}, none otherwise -- field {@code tf} has no payloads and every document's frequency is
 * a multiple of its term's length, and {@code CheckIndex} is clean.
 */
public class VerifyTokenAttributes {
  public static void main(String[] args) throws Exception {
    for (String sub : new String[] {"flushed", "merged"}) {
      Path path = Path.of(args[0]).resolve(sub);
      try (Directory dir = FSDirectory.open(path)) {
        check(dir, sub);
        try (CheckIndex check = new CheckIndex(dir)) {
          if (!check.checkIndex().clean) {
            throw new AssertionError(sub + ": CheckIndex failed");
          }
        }
      }
    }
    System.out.println("VerifyTokenAttributes: ok");
  }

  static void check(Directory dir, String sub) throws Exception {
    try (DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 1) {
        throw new AssertionError(sub + ": one segment expected");
      }
      LeafReader leaf = reader.leaves().get(0).reader();
      FieldInfo pay = leaf.getFieldInfos().fieldInfo("pay");
      FieldInfo tf = leaf.getFieldInfos().fieldInfo("tf");
      if (!pay.hasPayloads() || tf.hasPayloads()) {
        throw new AssertionError(sub + ": storePayloads pay=" + pay.hasPayloads() + " tf=" + tf.hasPayloads());
      }
      int withPayload = 0;
      TermsEnum te = leaf.terms("pay").iterator();
      BytesRef term;
      while ((term = te.next()) != null) {
        String t = term.utf8ToString();
        char c = t.charAt(0);
        PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
        while (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
          for (int i = 0; i < pe.freq(); i++) {
            pe.nextPosition();
            BytesRef p = pe.getPayload();
            if (c < 'm') {
              if (p == null || p.length != 2 || p.bytes[p.offset] != t.length() || p.bytes[p.offset + 1] != c) {
                throw new AssertionError(sub + ": " + t + " payload " + p);
              }
              withPayload++;
            } else if (p != null && p.length > 0) {
              throw new AssertionError(sub + ": " + t + " has payload " + p);
            }
          }
        }
      }
      if (withPayload == 0) {
        throw new AssertionError(sub + ": no payload read");
      }
      Terms tfTerms = leaf.terms("tf");
      te = tfTerms.iterator();
      int custom = 0;
      while ((term = te.next()) != null) {
        int len = term.utf8ToString().length();
        PostingsEnum pe = te.postings(null, PostingsEnum.FREQS);
        while (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
          if (pe.freq() % len != 0) {
            throw new AssertionError(sub + ": tf " + term.utf8ToString() + " freq " + pe.freq());
          }
          custom++;
        }
      }
      if (custom == 0) {
        throw new AssertionError(sub + ": no tf postings");
      }
      System.out.println(sub + ": " + withPayload + " payloads, " + custom + " custom frequencies");
    }
  }
}
