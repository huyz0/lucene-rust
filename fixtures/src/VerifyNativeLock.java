import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.Lock;
import org.apache.lucene.store.LockObtainFailedException;
import org.apache.lucene.store.NativeFSLockFactory;

/**
 * The Java half of the cross-engine write-lock test
 * (crates/lucene-store/tests/native_lock_interop.rs): proves that Lucene's
 * NativeFSLockFactory and the Rust port's NativeFsLockFactory exclude each
 * other on one directory.
 *
 * <pre>
 *   VerifyNativeLock try  DIR   prints HELD or OBTAINED for NativeFSLockFactory,
 *                               then the same for a real IndexWriter
 *   VerifyNativeLock hold DIR   takes the lock, prints LOCKED, holds it until a
 *                               line arrives on stdin, then releases it and
 *                               prints RELEASED
 * </pre>
 *
 * Run with the source launcher: {@code java -cp lucene-core.jar VerifyNativeLock.java ...}.
 */
public class VerifyNativeLock {
  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      System.err.println("usage: VerifyNativeLock try|hold DIR");
      System.exit(2);
    }
    try (Directory dir = FSDirectory.open(Paths.get(args[1]), NativeFSLockFactory.INSTANCE)) {
      switch (args[0]) {
        case "try" -> {
          try (Lock lock = dir.obtainLock(IndexWriter.WRITE_LOCK_NAME)) {
            lock.ensureValid();
            System.out.println("lock OBTAINED");
          } catch (LockObtainFailedException e) {
            System.out.println("lock HELD");
          }
          try (IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig())) {
            writer.rollback();
            System.out.println("writer OBTAINED");
          } catch (LockObtainFailedException e) {
            System.out.println("writer HELD");
          }
        }
        case "hold" -> {
          try (Lock lock = dir.obtainLock(IndexWriter.WRITE_LOCK_NAME)) {
            System.out.println("LOCKED");
            System.out.flush();
            new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))
                .readLine();
            lock.ensureValid();
          }
          System.out.println("RELEASED");
        }
        default -> {
          System.err.println("unknown mode " + args[0]);
          System.exit(2);
        }
      }
    }
  }
}
