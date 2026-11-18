/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.index.IndexableField;
import org.apache.lucene.util.BytesRef;

import java.nio.charset.StandardCharsets;
import java.util.List;

/** The segment partition a block of documents belongs to ({@code index.segment_partition.field}). */
final class PartitionKey {
    private PartitionKey() {}

    /**
     * The partition of a block of documents: the value of the partition field on its root document,
     * which OpenSearch puts last (nested documents before their parent). A keyword's value is its
     * doc-values bytes (UTF-8, or {@code 0x} and hex for bytes that are not, such as an ip's); a
     * numeric's or date's is its long; with several values, the smallest. Null -- the shard's default
     * partition -- without a partition field or a value. A value the field holds only as analyzed
     * text is refused: a partition is one exact value.
     */
    static String of(String partitionField, List<? extends Iterable<? extends IndexableField>> docs) {
        if (partitionField == null || docs.isEmpty()) {
            return null;
        }
        BytesRef bytes = null;
        boolean numeric = false;
        long number = Long.MAX_VALUE;
        BytesRef indexed = null;
        boolean text = false;
        for (IndexableField f : docs.get(docs.size() - 1)) {
            if (f.name().equals(partitionField) == false) {
                continue;
            }
            switch (f.fieldType().docValuesType()) {
                case SORTED, SORTED_SET -> {
                    BytesRef v = f.binaryValue();
                    if (bytes == null || v.compareTo(bytes) < 0) {
                        bytes = v;
                    }
                }
                case NUMERIC, SORTED_NUMERIC -> {
                    numeric = true;
                    number = Math.min(number, f.numericValue().longValue());
                }
                default -> {
                    if (f.fieldType().indexOptions() == org.apache.lucene.index.IndexOptions.NONE) {
                        continue;
                    }
                    if (f.fieldType().tokenized()) {
                        text = true;
                    } else if (f.binaryValue() != null && (indexed == null || f.binaryValue().compareTo(indexed) < 0)) {
                        // A keyword without doc values: its indexed term.
                        indexed = f.binaryValue();
                    }
                }
            }
        }
        if (bytes != null) {
            return partitionString(bytes);
        }
        if (numeric) {
            return Long.toString(number);
        }
        if (indexed != null) {
            return partitionString(indexed);
        }
        if (text) {
            throw new IllegalArgumentException(
                "the partition field [" + partitionField + "] must be a keyword, numeric, date or ip field, not text"
            );
        }
        return null;
    }

    /** A partition value's bytes as the string the segment records: UTF-8, or {@code 0x} and hex. */
    private static String partitionString(BytesRef v) {
        try {
            return StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
                .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)
                .decode(java.nio.ByteBuffer.wrap(v.bytes, v.offset, v.length))
                .toString();
        } catch (java.nio.charset.CharacterCodingException e) {
            return "0x" + java.util.HexFormat.of().formatHex(v.bytes, v.offset, v.offset + v.length);
        }
    }
}
