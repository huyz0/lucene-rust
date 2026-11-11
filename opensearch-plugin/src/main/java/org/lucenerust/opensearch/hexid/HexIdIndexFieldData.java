/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.hexid;

import org.apache.lucene.search.SortField;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.util.BigArrays;
import org.opensearch.core.indices.breaker.CircuitBreakerService;
import org.opensearch.index.fielddata.FieldData;
import org.opensearch.index.fielddata.IndexFieldData.XFieldComparatorSource.Nested;
import org.opensearch.index.fielddata.IndexFieldDataCache;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.index.fielddata.plain.SortedSetOrdinalsIndexFieldData;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.MultiValueMode;
import org.opensearch.search.aggregations.support.CoreValuesSourceType;
import org.opensearch.search.sort.BucketedSort;
import org.opensearch.search.sort.SortOrder;

/**
 * A {@code hex_id} field's ordinals: a {@code keyword}'s, with scripts reading the strings and a
 * sort's {@code missing} value encoded as the field's terms are (OpenSearch hands a string
 * {@code missing} to the comparator as its UTF-8 bytes).
 */
final class HexIdIndexFieldData extends SortedSetOrdinalsIndexFieldData {
    HexIdIndexFieldData(IndexFieldDataCache cache, String name, CircuitBreakerService breakerService) {
        super(cache, name, CoreValuesSourceType.BYTES, breakerService, dv -> new ScriptDocValues.Strings(FieldData.toString(dv)) {
            @Override
            protected String bytesToString(BytesRef bytes) {
                return HexIdCodec.decode(bytes);
            }
        });
    }

    /** {@code missing} as a term: {@code _first}, {@code _last} and null as they are. */
    static Object encodeMissing(Object missing) {
        if (missing == null || "_first".equals(missing) || "_last".equals(missing) || missing instanceof BytesRef) {
            return missing;
        }
        return HexIdCodec.encode(missing.toString());
    }

    @Override
    public SortField sortField(Object missingValue, MultiValueMode sortMode, Nested nested, boolean reverse) {
        return super.sortField(encodeMissing(missingValue), sortMode, nested, reverse);
    }

    @Override
    public BucketedSort newBucketedSort(
        BigArrays bigArrays,
        Object missingValue,
        MultiValueMode sortMode,
        Nested nested,
        SortOrder sortOrder,
        DocValueFormat format,
        int bucketSize,
        BucketedSort.ExtraData extra
    ) {
        return super.newBucketedSort(bigArrays, encodeMissing(missingValue), sortMode, nested, sortOrder, format, bucketSize, extra);
    }
}
