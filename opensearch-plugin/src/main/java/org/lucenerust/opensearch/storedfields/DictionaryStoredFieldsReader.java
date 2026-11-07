/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.storedfields;

import java.io.IOException;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.util.IOUtils;

/**
 * A stored-fields reader that owns its segment's {@link ZstdDictionary}: closing it releases the
 * dictionary. Its clones and merge instance share the dictionary and own nothing, as a Lucene
 * reader's clones never outlive it.
 */
final class DictionaryStoredFieldsReader extends StoredFieldsReader {
    private final StoredFieldsReader in;
    /** Null for a clone or merge instance. */
    private final ZstdDictionary dictionary;

    DictionaryStoredFieldsReader(StoredFieldsReader in, ZstdDictionary dictionary) {
        this.in = in;
        this.dictionary = dictionary;
    }

    @Override
    public void prefetch(int docID) throws IOException {
        in.prefetch(docID);
    }

    @Override
    public void document(int docID, StoredFieldVisitor visitor) throws IOException {
        in.document(docID, visitor);
    }

    @Override
    public StoredFieldsReader clone() {
        return new DictionaryStoredFieldsReader(in.clone(), null);
    }

    @Override
    public StoredFieldsReader getMergeInstance() {
        return new DictionaryStoredFieldsReader(in.getMergeInstance(), null);
    }

    @Override
    public void checkIntegrity() throws IOException {
        in.checkIntegrity();
    }

    @Override
    public void close() throws IOException {
        if (dictionary == null) {
            in.close();
        } else {
            IOUtils.close(in, dictionary);
        }
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(" + in + ")";
    }
}
