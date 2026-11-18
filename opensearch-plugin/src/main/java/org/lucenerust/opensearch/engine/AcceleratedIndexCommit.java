/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.index.IndexCommit;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;

/** A commit point as {@code IndexDeletionPolicy} sees it; its {@code segments_N} is read lazily. */
final class AcceleratedIndexCommit extends IndexCommit {
    private final Directory directory;
    private final long generation;
    private final String segmentsFileName;
    private SegmentInfos infos;
    private boolean deleted;

    AcceleratedIndexCommit(Directory directory, long generation) {
        this.directory = directory;
        this.generation = generation;
        this.segmentsFileName = org.apache.lucene.index.IndexFileNames.fileNameFromGeneration(
            org.apache.lucene.index.IndexFileNames.SEGMENTS,
            "",
            generation
        );
    }

    synchronized SegmentInfos segmentInfos() throws IOException {
        if (infos == null) {
            infos = SegmentInfos.readCommit(directory, segmentsFileName);
        }
        return infos;
    }

    @Override
    public String getSegmentsFileName() {
        return segmentsFileName;
    }

    @Override
    public Collection<String> getFileNames() throws IOException {
        return Collections.unmodifiableCollection(segmentInfos().files(true));
    }

    @Override
    public Directory getDirectory() {
        return directory;
    }

    @Override
    public void delete() {
        deleted = true;
    }

    @Override
    public boolean isDeleted() {
        return deleted;
    }

    @Override
    public int getSegmentCount() {
        try {
            return segmentInfos().size();
        } catch (IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    @Override
    public long getGeneration() {
        return generation;
    }

    @Override
    public Map<String, String> getUserData() throws IOException {
        return segmentInfos().getUserData();
    }

    @Override
    public String toString() {
        return "AcceleratedIndexCommit(" + segmentsFileName + ")";
    }
}
