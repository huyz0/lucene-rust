import java.io.IOException;
import java.nio.file.Paths;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.store.MMapDirectory;

/**
 * Java side of the stored-fields microbenchmark -- {@code StoredFields.document(docId)}, which
 * every real search does once per returned hit and which this project had never compared.
 *
 * <p>Same fixed odd stride as the Rust side, so consecutive fetches land in different compressed
 * blocks: that is what a top-k result set looks like, whereas a sequential sweep would measure the
 * block cache.
 *
 * <p>Emits TSV {@code document<TAB>ns_per_doc<TAB>docs} on stdout.
 */
public final class StoredFieldsMicro {

    private static final int STRIDE = 4099;

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("usage: StoredFieldsMicro <index-dir>");
            System.exit(2);
        }
        long warmupMs = Long.getLong("warmupMs", 1500);
        long measureMs = Long.getLong("measureMs", 2000);

        try (MMapDirectory dir = new MMapDirectory(Paths.get(args[0]));
             DirectoryReader reader = DirectoryReader.open(dir)) {
            StoredFields sf = reader.storedFields();
            int maxDoc = reader.maxDoc();
            if (sf.document(0).getFields().isEmpty()) {
                System.err.println(
                        "micro: this index stores no fields, so stored-field retrieval cannot be "
                                + "measured against it -- regenerate the corpus with a stored field");
                return;
            }
            walk(sf, maxDoc, warmupMs);
            long docs = walk(sf, maxDoc, measureMs);
            System.out.printf("document\t%.3f\t%d%n", (double) lastNanos / docs, docs);
        }
    }

    private static long lastNanos;
    private static int sink;

    private static long walk(StoredFields sf, int maxDoc, long budgetMs) throws IOException {
        long budgetNs = budgetMs * 1_000_000L;
        long docs = 0;
        int doc = 0;
        long start = System.nanoTime();
        long elapsed;
        do {
            for (int i = 0; i < 64; i++) {
                doc = (doc + STRIDE) % maxDoc;
                sink += sf.document(doc).getFields().size();
            }
            docs += 64;
            elapsed = System.nanoTime() - start;
        } while (elapsed < budgetNs);
        lastNanos = elapsed;
        if (sink == 0xDEADBEEF) {
            System.err.print("");
        }
        return docs;
    }
}
