/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.opensearch.common.document.DocumentField;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.DocValueFetcher;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.fetch.subphase.FieldAndFormat;

/**
 * The {@code docvalue_fields} of a direct search's hits and the {@code stored_fields} shapes it answers: formatted
 * doc values read through the field data the fetch phase uses, hit documents visited in document order (doc values
 * only move forward), each hit's {@code fields} member written by {@link DocumentField}.
 */
final class DirectFields {
    private record Field(String name, IndexFieldData<?> data, DocValueFormat format) {}

    private final List<Field> fields;

    private DirectFields(List<Field> fields) {
        this.fields = fields;
    }

    /** The named stored fields ({@code null} names, as {@code _none_} has them, are none). */
    static List<String> names(SearchSourceBuilder source) {
        List<String> names = source.storedFields() == null ? null : source.storedFields().fieldNames();
        return names == null ? List.of() : names;
    }

    /** {@code "stored_fields": "_none_"}: hits carry neither {@code _id}, {@code _source} nor metadata. */
    static boolean none(SearchSourceBuilder source) {
        return source.storedFields() != null && source.storedFields().fetchFields() == false;
    }

    /** Whether the request's {@code stored_fields} and {@code docvalue_fields} have a shape this path writes. */
    static boolean shape(SearchSourceBuilder source) {
        if (source.storedFields() != null) {
            for (String name : names(source)) {
                if (name.indexOf('*') >= 0) {
                    return false;
                }
            }
        }
        if (source.docValueFields() != null) {
            for (FieldAndFormat f : source.docValueFields()) {
                if (f.field.indexOf('*') >= 0 || "use_field_mapping".equals(f.format)) {
                    return false;
                }
            }
        }
        return true;
    }

    /** Whether the named stored fields all hold nothing to write: {@code _source} or a field that is not stored. */
    static boolean supported(SearchSourceBuilder source, MapperService mappers) {
        if (source.storedFields() == null) {
            return true;
        }
        for (String name : names(source)) {
            MappedFieldType type = name.equals("_source") ? null : mappers.fieldType(name);
            if (type != null && (type.isStored() || name.startsWith("_"))) {
                return false;
            }
        }
        return true;
    }

    /** The fields to read, or null when one cannot be read as the fetch phase reads it. */
    static DirectFields of(List<FieldAndFormat> requested, MapperService mappers, QueryShardContext context) {
        List<Field> out = new ArrayList<>();
        for (FieldAndFormat f : requested) {
            MappedFieldType type = mappers.fieldType(f.field);
            if (type == null) {
                continue;
            }
            if (type.hasDocValues() == false) {
                return null;
            }
            out.add(new Field(f.field, context.getForField(type), type.docValueFormat(f.format, null)));
        }
        return new DirectFields(out);
    }

    /** Each hit's {@code {...}} member for hits {@code from} to {@code to} of {@code docs}, null where it has no values. */
    String[] render(IndexReader reader, int[] docs, int from, int to) throws IOException {
        String[] out = new String[Math.max(to - from, 0)];
        Integer[] order = new Integer[out.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = from + i;
        }
        Arrays.sort(order, Comparator.comparingInt(i -> docs[i]));
        List<LeafReaderContext> leaves = reader.leaves();
        DocValueFetcher.Leaf[] values = new DocValueFetcher.Leaf[fields.size()];
        int leaf = -1;
        for (int i : order) {
            int doc = docs[i];
            while (leaf + 1 < leaves.size() && leaves.get(leaf + 1).docBase <= doc) {
                leaf++;
                for (int f = 0; f < values.length; f++) {
                    values[f] = fields.get(f).data().load(leaves.get(leaf)).getLeafValueFetcher(fields.get(f).format());
                }
            }
            int local = doc - leaves.get(leaf).docBase;
            // A field named twice gets both entries' values, in request order.
            Map<String, List<Object>> merged = new java.util.LinkedHashMap<>();
            for (int f = 0; f < values.length; f++) {
                if (values[f].advanceExact(local) == false) {
                    continue;
                }
                int count = values[f].docValueCount();
                for (int k = 0; k < count; k++) {
                    merged.computeIfAbsent(fields.get(f).name(), n -> new ArrayList<>()).add(values[f].nextValue());
                }
            }
            XContentBuilder b = null;
            for (Map.Entry<String, List<Object>> e : merged.entrySet()) {
                if (b == null) {
                    b = JsonXContent.contentBuilder();
                    b.startObject();
                }
                new DocumentField(e.getKey(), e.getValue()).toXContent(b, ToXContent.EMPTY_PARAMS);
            }
            if (b != null) {
                b.endObject();
                out[i - from] = b.toString();
            }
        }
        return out;
    }
}
