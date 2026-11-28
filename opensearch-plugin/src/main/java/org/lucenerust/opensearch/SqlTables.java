/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.index.mapper.TextFieldMapper;

/**
 * An index's mapping as the SQL table the native side reads ({@code decode_table} in {@code jvm_sql.rs}): the
 * fields a statement can name, each by how its values are read.
 *
 * <ul>
 *   <li>{@code keyword} with doc values and an index: a keyword column ({@code GROUP BY}, equality);
 *   <li>{@code long}, {@code integer}, {@code short}, {@code byte} and {@code date} (as epoch milliseconds) with doc
 *       values: a long column;
 *   <li>{@code text}: a column {@code match_text(column, '...')} searches, analysed with the field's search analyzer
 *       when the native side can run it.
 * </ul>
 *
 * Any other field (floating point, boolean, ip, geo, vectors, objects, multi-fields) is not a column.
 */
final class SqlTables {
    private SqlTables() {}

    /** The layout version {@code decode_table} expects. */
    static final int VERSION = 1;

    private static final int KEY = 0;
    private static final int LONG = 1;
    private static final int KEYWORD = 2;
    private static final int TEXT = 3;

    /** A column the table has: its name and the kind it is read as. */
    record Column(String name, String type) {}

    /** The table of {@code table} over {@code mapper}: its blob, and its columns for a client that asks. */
    record Table(byte[] blob, List<Column> columns) {}

    static Table of(String table, MapperService mapper) {
        List<Column> columns = new ArrayList<>();
        ByteArrayOutputStream body = new ByteArrayOutputStream();
        int count = 0;
        for (MappedFieldType type : mapper.fieldTypes()) {
            String name = type.name();
            if (name.startsWith("_") || mapper.documentMapper().mappers().isMultiField(name)) {
                continue;
            }
            int kind = kindOf(type);
            if (kind < 0) {
                continue;
            }
            byte[] analyzer = new byte[0];
            if (kind == TEXT) {
                AnalyzerSpec.Spec spec = AnalyzerSpec.cached(type.getTextSearchInfo().getSearchAnalyzer(), name);
                if (spec.ok() == false) {
                    continue;
                }
                analyzer = spec.blob();
            }
            body.write(kind);
            putBytes(body, name.getBytes(StandardCharsets.UTF_8));
            putInt(body, 0);
            putBytes(body, analyzer);
            columns.add(new Column(name, type.typeName()));
            count++;
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        putInt(out, VERSION);
        putBytes(out, table.getBytes(StandardCharsets.UTF_8));
        putInt(out, count);
        out.writeBytes(body.toByteArray());
        return new Table(out.toByteArray(), columns);
    }

    private static int kindOf(MappedFieldType type) {
        if (type.hasDocValues() == false && type instanceof TextFieldMapper.TextFieldType == false) {
            return -1;
        }
        if (type.isSearchable() == false) {
            return -1;
        }
        return switch (type.typeName()) {
            case "keyword" -> KEYWORD;
            case "text" -> type instanceof TextFieldMapper.TextFieldType ? TEXT : -1;
            case "long", "integer", "short", "byte", "date" -> type instanceof NumberFieldMapper.NumberFieldType
                || type.typeName().equals("date") ? LONG : -1;
            default -> -1;
        };
    }

    private static void putInt(ByteArrayOutputStream out, int v) {
        out.writeBytes(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array());
    }

    private static void putBytes(ByteArrayOutputStream out, byte[] b) {
        putInt(out, b.length);
        out.writeBytes(b);
    }
}
