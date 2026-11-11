/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KeepOnlyLastCommitDeletionPolicy;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.FSDirectory;
import org.lucenerust.opensearch.NativeBridge;
import org.lucenerust.opensearch.substring.RustWildcardFieldMapper;
import org.opensearch.index.mapper.ParseContext;

/** An index of {@code rust_wildcard} values written by the Rust engine's writer, for the self test. */
public final class SubstringTestIndex {
    private SubstringTestIndex() {}

    /**
     * Writes {@code values} (null: no value) as documents {@code id = i} with the value in {@code
     * field}, committing every {@code perCommit} documents; then closes the writer and has a new one,
     * which no document registers the field with, merge the segments down to {@code mergeTo}.
     */
    public static void write(Path root, String field, List<String> values, int perCommit, int mergeTo) throws IOException {
        try (FSDirectory dir = FSDirectory.open(root)) {
            try (IndexWriter empty = new IndexWriter(dir, new IndexWriterConfig())) {
                empty.commit();
            }
            RustIndexWriter w = open(dir, root);
            for (int i = 0; i < values.size(); i++) {
                ParseContext.Document d = new ParseContext.Document();
                d.add(new StringField("id", Integer.toString(i), Field.Store.YES));
                if (values.get(i) != null) {
                    d.add(RustWildcardFieldMapper.valueField(field, values.get(i)));
                }
                w.addDocument(d, null);
                if ((i + 1) % perCommit == 0) {
                    w.commit();
                }
            }
            w.commit();
            w.close();
            RustIndexWriter merger = open(dir, root);
            merger.forceMerge(mergeTo, true);
            merger.commit();
            merger.close();
        }
    }

    private static RustIndexWriter open(FSDirectory dir, Path root) throws IOException {
        RustIndexWriter w = RustIndexWriter.open(
            dir,
            root,
            16,
            false,
            false,
            false,
            false,
            false,
            false,
            null,
            false,
            null,
            null,
            IndexWriter.MAX_DOCS,
            NativeBridge.STORED_FIELDS_BEST_SPEED,
            3,
            new StandardAnalyzer(),
            new BM25Similarity(),
            10,
            "__soft_deletes",
            null,
            new KeepOnlyLastCommitDeletionPolicy(),
            null,
            null,
            null
        );
        w.setLiveCommitData(Map.<String, String>of().entrySet());
        return w;
    }
}
