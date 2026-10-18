import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.lucene.replicator.nrt.CopyJob;
import org.apache.lucene.replicator.nrt.CopyOneFile;
import org.apache.lucene.replicator.nrt.FileMetaData;
import org.apache.lucene.replicator.nrt.Node;
import org.apache.lucene.replicator.nrt.ReplicaNode;
import org.apache.lucene.search.SearcherFactory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

/**
 * M14 T14.3's replicator pairs (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_replicator.rs}), over the 25 files of the NRT point in {@code
 * fixtures/data/replicator/plain} (copied to a scratch directory first):
 *
 * <ul>
 *   <li>{@code repl_file_meta}: {@code Node.readLocalFileMetaData} of every file -- what a primary
 *       does per file of each new point, and a replica per file it might already hold;
 *   <li>{@code repl_copy_file}: {@code CopyOneFile} of every file into a replica's directory,
 *       checksum verified, then the temporary file deleted -- the replica's copy loop.
 * </ul>
 *
 * Units: files. Output is {@code name\tns_per_unit\tunits}.
 */
public class ReplicatorMicro {

  /** A node that only reads metadata. */
  static final class MetaNode extends Node {
    MetaNode(Directory dir) {
      super(0, dir, new SearcherFactory(), null);
      setVerboseFiles(false);
    }

    @Override
    public void commit() {}

    @Override
    public boolean isClosed() {
      return false;
    }

    @Override
    public void close() {}
  }

  /** A replica that only copies files. */
  static final class CopyReplica extends ReplicaNode {
    CopyReplica(Directory dir) throws IOException {
      super(1, dir, new SearcherFactory(), null);
      setVerboseFiles(false);
    }

    @Override
    protected CopyJob newCopyJob(
        String reason,
        Map<String, FileMetaData> files,
        Map<String, FileMetaData> prevFiles,
        boolean highPriority,
        CopyJob.OnceDone onceDone) {
      throw new UnsupportedOperationException();
    }

    @Override
    protected void launch(CopyJob job) {
      throw new UnsupportedOperationException();
    }

    @Override
    protected void sendNewReplica() {}
  }

  public static void main(String[] args) throws Exception {
    // Read from a scratch copy: off the repository's bind mount, a file open costs a round trip
    // to the host and the benchmark would measure the mount.
    Path srcPath = java.nio.file.Files.createTempDirectory("repl-bench-src");
    try (var files = java.nio.file.Files.list(Path.of("fixtures/data/replicator/plain"))) {
      for (Path f : (Iterable<Path>) files::iterator) {
        java.nio.file.Files.copy(f, srcPath.resolve(f.getFileName()));
      }
    }
    Directory src = FSDirectory.open(srcPath);
    List<String> names = new ArrayList<>();
    for (String name : src.listAll()) {
      if (!name.startsWith("segments_") && !name.equals("write.lock")) {
        names.add(name);
      }
    }
    MetaNode node = new MetaNode(src);
    SweepMicro.measure(
        "repl_file_meta",
        () -> {
          for (String name : names) {
            SweepMicro.sink += node.readLocalFileMetaData(name).checksum();
          }
          return names.size();
        });

    List<FileMetaData> metas = new ArrayList<>();
    for (String name : names) {
      metas.add(node.readLocalFileMetaData(name));
    }
    Path tmp = java.nio.file.Files.createTempDirectory("repl-bench");
    Directory dest = FSDirectory.open(tmp);
    CopyReplica replica = new CopyReplica(dest);
    byte[] buffer = new byte[65536];
    SweepMicro.measure(
        "repl_copy_file",
        () -> {
          for (int i = 0; i < names.size(); i++) {
            String name = names.get(i);
            try (IndexInput in = src.openInput(name, IOContext.READONCE)) {
              CopyOneFile copy = new CopyOneFile(in, replica, name, metas.get(i), buffer);
              while (!copy.visit()) {}
              dest.deleteFile(copy.tmpName);
            }
          }
          return names.size();
        });
    // The replica was never started (ReplicaNode.close needs its searcher manager); the write
    // lock it holds goes with the process.
    for (String f : dest.listAll()) {
      if (!f.equals("write.lock")) {
        dest.deleteFile(f);
      }
    }
  }
}
