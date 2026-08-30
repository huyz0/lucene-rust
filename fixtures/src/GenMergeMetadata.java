import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Version;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Source segments for the merged-segment *metadata* differential case: three
 * real-Lucene segments that disagree about both facts `SegmentMerger` folds
 * across its readers -- `minVersion` and `hasBlocks`.
 *
 * <p>Why this cannot be produced by an ordinary IndexWriter run: a single
 * writer stamps `Version.LATEST` as every flushed segment's minVersion, so an
 * index built normally has nothing to take a minimum *of*. The fixture
 * therefore writes the three segments normally -- so their stored fields,
 * `.fnm` and `hasBlocks` flags are genuinely Lucene's -- and then rewrites
 * each `.si` through the codec's own `SegmentInfoFormat.write` with a chosen
 * `minVersion`, exactly the way a real index that had been written across
 * several Lucene releases and then upgraded would look. Everything else about
 * each segment (name, id, files, codec, diagnostics, maxDoc, hasBlocks) is
 * carried across unchanged, so the rewritten `.si` is still a file real
 * Lucene wrote.
 *
 * <p>`hasBlocks` is not synthesised at all: segment `_1` is built with
 * `addDocuments`, which is what makes `DocumentsWriterPerThread` call
 * `segmentInfo.setHasBlocks()`. The other two are single `addDocument` calls
 * and carry `false`.
 *
 * <p>What this port then has to get right, in `write_merged_metadata_fixture`
 * + `VerifyMergedMetadata`: merging these three must produce a `.si` whose
 * `minVersion` is the *oldest* of the three (10.0.0) and whose `hasBlocks` is
 * the disjunction (true). Writing the merging writer's own version -- what
 * this port did before c36 -- passes every checksum, opens cleanly and is
 * wrong.
 */
public class GenMergeMetadata {
  /** One `minVersion` per segment, oldest in the middle so a fold that just takes the first or the last is caught. */
  static final Version[] MIN_VERSIONS = {
    Version.LUCENE_10_2_0, Version.LUCENE_10_0_0, Version.LUCENE_10_1_0
  };

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("merge_metadata");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    List<String> ids = new ArrayList<>();
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      // One flush per commit and no merging, so the three segments survive
      // exactly as written and the `.si` rewrite below has three distinct
      // targets.
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        // _0: two ordinary single-document adds -> hasBlocks false.
        w.addDocument(storedOnly("a0"));
        w.addDocument(storedOnly("a1"));
        ids.add("a0");
        ids.add("a1");
        w.commit();

        // _1: one addDocuments call with three documents -> Lucene sets
        // hasBlocks on this segment and only this one.
        List<Document> block = new ArrayList<>();
        for (String id : new String[] {"b0", "b1", "b2"}) {
          block.add(storedOnly(id));
          ids.add(id);
        }
        w.addDocuments(block);
        w.commit();

        // _2: two more ordinary adds -> hasBlocks false.
        w.addDocument(storedOnly("c0"));
        w.addDocument(storedOnly("c1"));
        ids.add("c0");
        ids.add("c1");
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != MIN_VERSIONS.length) {
        throw new IllegalStateException("expected " + MIN_VERSIONS.length + " segments, got " + sis.size());
      }

      StringBuilder names = new StringBuilder();
      StringBuilder mins = new StringBuilder();
      StringBuilder blocks = new StringBuilder();
      Version oldest = null;
      boolean anyBlocks = false;
      for (int i = 0; i < sis.size(); i++) {
        SegmentCommitInfo sci = sis.info(i);
        SegmentInfo old = sci.info;
        Version min = MIN_VERSIONS[i];
        if (oldest == null || oldest.onOrAfter(min)) {
          oldest = min;
        }
        anyBlocks |= old.getHasBlocks();

        SegmentInfo rewritten =
            new SegmentInfo(
                old.dir,
                old.getVersion(),
                min,
                old.name,
                old.maxDoc(),
                old.getUseCompoundFile(),
                old.getHasBlocks(),
                old.getCodec(),
                old.getDiagnostics(),
                old.getId(),
                old.getAttributes(),
                old.getIndexSort());
        rewritten.setFiles(old.files());
        dir.deleteFile(old.name + ".si");
        Codec codec = old.getCodec();
        codec.segmentInfoFormat().write(dir, rewritten, IOContext.DEFAULT);

        if (i > 0) {
          names.append(',');
          mins.append(',');
          blocks.append(',');
        }
        names.append(old.name);
        mins.append(min.toString());
        blocks.append(old.getHasBlocks());
      }
      dir.sync(List.of());

      // Prove the rewritten `.si` files still form an index real Lucene reads,
      // and that the flags came back the way they were written.
      SegmentInfos reread = SegmentInfos.readLatestCommit(dir);
      for (int i = 0; i < reread.size(); i++) {
        SegmentInfo si = reread.info(i).info;
        if (!MIN_VERSIONS[i].equals(si.getMinVersion())) {
          throw new IllegalStateException(
              "minVersion rewrite did not stick for " + si.name + ": " + si.getMinVersion());
        }
      }

      StringBuilder m = new StringBuilder();
      m.append("segment_names=").append(names).append('\n');
      m.append("segment_min_versions=").append(mins).append('\n');
      m.append("segment_has_blocks=").append(blocks).append('\n');
      m.append("expected_merged_min_version=").append(oldest).append('\n');
      m.append("expected_merged_has_blocks=").append(anyBlocks).append('\n');
      m.append("doc_ids=").append(String.join(",", ids)).append('\n');
      m.append("field_name=id\n");
      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote merge_metadata/ fixture directory");
  }

  static Document storedOnly(String id) {
    Document doc = new Document();
    // Stored-only: no postings, no doc values, no norms -- the merged segment
    // is then about nothing but the two metadata fields under test.
    doc.add(new StoredField("id", id));
    return doc;
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
}
