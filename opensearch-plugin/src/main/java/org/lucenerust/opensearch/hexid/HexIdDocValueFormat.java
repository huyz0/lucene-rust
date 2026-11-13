/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.hexid;

import org.apache.lucene.util.BytesRef;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.search.DocValueFormat;

/**
 * A {@code hex_id} field's doc values as their strings: terms aggregation and composite keys, sort
 * values, {@code docvalue_fields}, and back for {@code search_after}, {@code after}, {@code missing}
 * and {@code include}/{@code exclude} values.
 */
public final class HexIdDocValueFormat implements DocValueFormat {
    public static final String NAME = "hex_id";
    public static final HexIdDocValueFormat INSTANCE = new HexIdDocValueFormat();

    private HexIdDocValueFormat() {}

    /** The {@code NamedWriteableRegistry} reader: the format has no state. */
    public static HexIdDocValueFormat fromStream(StreamInput in) {
        return INSTANCE;
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    public void writeTo(StreamOutput out) {}

    @Override
    public Object format(BytesRef value) {
        return HexIdCodec.decode(value);
    }

    @Override
    public BytesRef parseBytesRef(String value) {
        return HexIdCodec.encode(value);
    }

    @Override
    public String toString() {
        return NAME;
    }
}
