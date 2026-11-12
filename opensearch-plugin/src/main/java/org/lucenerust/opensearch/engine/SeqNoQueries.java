/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexReaderContext;
import org.apache.lucene.index.FilterCodecReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Query;
import org.opensearch.index.mapper.SeqNoFieldMapper;

/**
 * Range queries on {@code _seq_no} that match the same documents whichever way it is indexed: on
 * its points (stock), or on its doc values and their skip index ({@link SeqNoSettings#DOC_VALUES_ONLY}).
 */
public final class SeqNoQueries {
    private SeqNoQueries() {}

    /**
     * Whether some segment of {@code top} has {@code _seq_no} without points of its own (shown or
     * not as {@link SeqNoPointsReader}'s synthetic ones).
     */
    public static boolean docValuesOnly(IndexReaderContext top) {
        for (LeafReaderContext leaf : top.leaves()) {
            if (docValuesOnly(leaf.reader())) {
                return true;
            }
        }
        return false;
    }

    private static boolean docValuesOnly(LeafReader reader) {
        while (true) {
            if (SeqNoPointsReader.synthesizes(reader)) {
                return true;
            }
            if (reader instanceof FilterLeafReader f) {
                reader = f.getDelegate();
            } else if (reader instanceof FilterCodecReader f) {
                reader = f.getDelegate();
            } else {
                FieldInfo info = reader.getFieldInfos().fieldInfo(SeqNoFieldMapper.NAME);
                return info != null && info.getPointDimensionCount() == 0;
            }
        }
    }

    /** {@code _seq_no} in {@code [from, to]} over {@code reader}. */
    static Query range(IndexReader reader, long from, long to) {
        return docValuesOnly(reader.getContext())
            ? NumericDocValuesField.newSlowRangeQuery(SeqNoFieldMapper.NAME, from, to)
            : LongPoint.newRangeQuery(SeqNoFieldMapper.NAME, from, to);
    }
}
