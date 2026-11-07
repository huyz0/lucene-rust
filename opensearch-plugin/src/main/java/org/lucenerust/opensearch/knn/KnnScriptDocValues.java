/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.opensearch.ExceptionsHelper;
import org.opensearch.index.fielddata.ScriptDocValues;

import java.io.IOException;
import java.util.Locale;

/**
 * {@code doc['field']} of a {@code knn_vector} field in a script, k-NN's {@code
 * KNNVectorScriptDocValues}: at most one value, the document's vector ({@code float[]} for float and
 * half-float fields, {@code byte[]} for byte and binary ones), read from the vector files or else
 * the binary doc values.
 */
public final class KnnScriptDocValues extends ScriptDocValues<Object> {
    /** A segment's vectors, by document. */
    interface Source {
        /** The first document at or after {@code doc} with a vector. */
        int advance(int doc) throws IOException;

        /** The vector of the document {@link #advance} stopped at. */
        Object vector() throws IOException;
    }

    private final Source source;
    private final String field;
    private final String dataType;
    private boolean exists;
    private int lastDoc = -1;
    private int current = -1;

    KnnScriptDocValues(Source source, String field, String dataType) {
        this.source = source;
        this.field = field;
        this.dataType = dataType;
    }

    /** The field's {@code data_type}: {@code float}, {@code half_float}, {@code byte} or {@code binary}. */
    public String dataType() {
        return dataType;
    }

    /** A float or half-float field: its vectors are {@code float[]}. */
    public boolean floats() {
        return dataType.equals("float") || dataType.equals("half_float");
    }

    @Override
    public void setNextDocId(int docId) throws IOException {
        if (docId < lastDoc) {
            throw new IllegalArgumentException("docs were sent out-of-order: lastDocID=" + lastDoc + " vs docID=" + docId);
        }
        lastDoc = docId;
        if (docId > current) {
            current = source == null ? Integer.MAX_VALUE : source.advance(docId);
        }
        exists = docId == current;
    }

    /** The document's vector. */
    public Object getValue() {
        if (exists == false) {
            throw new IllegalStateException(
                String.format(
                    Locale.ROOT,
                    "One of the document doesn't have a value for field '%s'. "
                        + "This can be avoided by checking if a document has a value for the field or not "
                        + "by doc['%s'].size() == 0 ? 0 : {your script}",
                    field,
                    field
                )
            );
        }
        try {
            return source.vector();
        } catch (IOException e) {
            throw ExceptionsHelper.convertToOpenSearchException(e);
        }
    }

    @Override
    public int size() {
        return exists ? 1 : 0;
    }

    @Override
    public Object get(int i) {
        throw new UnsupportedOperationException("knn vector does not support this operation");
    }
}
