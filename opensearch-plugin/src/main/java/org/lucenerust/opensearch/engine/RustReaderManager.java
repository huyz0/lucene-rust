/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SoftDeletesDirectoryReaderWrapper;
import org.apache.lucene.index.StandardDirectoryReader;
import org.apache.lucene.search.ReferenceManager;
import org.apache.lucene.store.Directory;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.common.lucene.index.OpenSearchDirectoryReader;
import org.opensearch.core.index.shard.ShardId;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * The engine's internal reader manager: every reader is a Java {@link StandardDirectoryReader} on
 * one of the Rust writer's commits, with soft deletes applied ({@link
 * SoftDeletesDirectoryReaderWrapper}, as OpenSearch's segment-replication replicas read) and wrapped
 * for OpenSearch. A refresh asks the writer to commit whatever is buffered and opens the newest
 * commit, reusing the segment readers of every segment that did not change.
 *
 * <p>Each reader pins its commit's files on the Rust side for as long as it is open -- what {@code
 * IndexWriter}'s NRT readers do through {@code IndexFileDeleter} -- so a merge or a dropped commit
 * never deletes a file a search, or a replica copying segments, still reads.
 */
final class RustReaderManager extends ReferenceManager<OpenSearchDirectoryReader> {
    private final RustIndexWriter writer;
    private final Directory directory;
    private final ShardId shardId;
    private final Comparator<LeafReader> leafSorter;

    RustReaderManager(RustIndexWriter writer, Directory directory, ShardId shardId, Comparator<LeafReader> leafSorter)
        throws IOException {
        this.writer = writer;
        this.directory = directory;
        this.shardId = shardId;
        this.leafSorter = leafSorter;
        this.current = open(null);
    }

    /** The generation of the commit {@code reader} was opened on. */
    static long generation(OpenSearchDirectoryReader reader) {
        return standard(reader).getSegmentInfos().getGeneration();
    }

    static StandardDirectoryReader standard(DirectoryReader reader) {
        return (StandardDirectoryReader) FilterDirectoryReader.unwrap(reader);
    }

    private OpenSearchDirectoryReader open(OpenSearchDirectoryReader previous) throws IOException {
        long[] latest = writer.acquireLatest();
        long generation = latest[0];
        long hold = latest[1];
        boolean success = false;
        try {
            if (previous != null && generation(previous) == generation) {
                writer.releaseHold(hold);
                success = true;
                return null;
            }
            SegmentInfos infos = SegmentInfos.readCommit(
                directory,
                IndexFileNames.fileNameFromGeneration(IndexFileNames.SEGMENTS, "", generation)
            );
            List<LeafReader> old = null;
            if (previous != null) {
                old = new ArrayList<>();
                for (LeafReaderContext ctx : standard(previous).leaves()) {
                    old.add(ctx.reader());
                }
            }
            DirectoryReader std = StandardDirectoryReader.open(directory, infos, old, leafSorter, null);
            std.getReaderCacheHelper().addClosedListener(key -> writer.releaseHold(hold));
            success = true;
            return OpenSearchDirectoryReader.wrap(new SoftDeletesDirectoryReaderWrapper(std, Lucene.SOFT_DELETES_FIELD), shardId);
        } finally {
            if (success == false) {
                writer.releaseHold(hold);
            }
        }
    }

    @Override
    protected void decRef(OpenSearchDirectoryReader reference) throws IOException {
        reference.decRef();
    }

    @Override
    protected OpenSearchDirectoryReader refreshIfNeeded(OpenSearchDirectoryReader referenceToRefresh) throws IOException {
        return open(referenceToRefresh);
    }

    @Override
    protected boolean tryIncRef(OpenSearchDirectoryReader reference) {
        return reference.tryIncRef();
    }

    @Override
    protected int getRefCount(OpenSearchDirectoryReader reference) {
        return reference.getRefCount();
    }
}
