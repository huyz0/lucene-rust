/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

/**
 * The k-NN plugin's {@code KNN80CompoundFormat}: a native engine's graph files ({@code .faiss},
 * {@code .hnsw}) stay outside the compound file, copied to the same name plus {@code c}, and the
 * compound reader opens those copies from the segment's directory. This plugin writes no such
 * files; the codecs carry the format so that, on a node where the stock k-NN plugin is installed
 * too and these codec names resolve here, its faiss and nmslib segments still open.
 */
final class KnnCompoundFormat extends CompoundFormat {
    private static final String[] ENGINE_EXTENSIONS = { ".faiss", ".hnsw" };

    private final CompoundFormat delegate;

    KnnCompoundFormat(CompoundFormat delegate) {
        this.delegate = delegate;
    }

    private static boolean engineCompound(String name) {
        for (String ext : ENGINE_EXTENSIONS) {
            if (name.endsWith(ext + "c")) {
                return true;
            }
        }
        return false;
    }

    @Override
    public CompoundDirectory getCompoundReader(Directory dir, SegmentInfo si) throws IOException {
        CompoundDirectory compound = delegate.getCompoundReader(dir, si);
        return new CompoundDirectory() {
            @Override
            public void checkIntegrity() throws IOException {
                compound.checkIntegrity();
            }

            @Override
            public String[] listAll() throws IOException {
                return compound.listAll();
            }

            @Override
            public long fileLength(String name) throws IOException {
                return compound.fileLength(name);
            }

            @Override
            public IndexInput openInput(String name, IOContext context) throws IOException {
                return engineCompound(name) ? dir.openInput(name, context) : compound.openInput(name, context);
            }

            @Override
            public void close() throws IOException {
                compound.close();
            }

            @Override
            public Set<String> getPendingDeletions() throws IOException {
                return compound.getPendingDeletions();
            }
        };
    }

    @Override
    public void write(Directory dir, SegmentInfo si, IOContext context) throws IOException {
        Set<String> files = new HashSet<>(si.files());
        for (String file : si.files()) {
            for (String ext : ENGINE_EXTENSIONS) {
                if (file.endsWith(ext)) {
                    dir.copyFrom(dir, file, file + "c", context);
                    files.remove(file);
                }
            }
        }
        si.setFiles(files);
        delegate.write(dir, si, context);
    }
}
