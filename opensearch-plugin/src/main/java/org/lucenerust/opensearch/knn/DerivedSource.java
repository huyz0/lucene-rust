/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.IndexableField;
import org.lucenerust.opensearch.engine.RustEngineSupport;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.regex.Regex;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.mapper.DocumentMapper;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.SourceFieldMapper;
import org.opensearch.index.shard.IndexingOperationListener;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;

/**
 * k-NN's derived source on this node's writes: in an index that turns it on, each root vector
 * field's values are masked in the stored {@code _source} ({@link #MASK}) and read back from the
 * vector files. A field the Rust writer must list in the segment's {@code derived_vector_fields}
 * carries {@link #FIELD_ATTRIBUTE}, and {@link Listener} masks exactly those fields, so every
 * reader -- this node's native fetch and Lucene codec, and stock k-NN's -- puts the vectors back.
 *
 * <p>Only the Rust engine lists the fields, so only its indexes are masked; and only with realtime
 * search off, whose buffered documents are read without the vector files.
 */
public final class DerivedSource {
    /** The field attribute the Rust writer lists a field for ({@code engine_writer.rs}). */
    public static final String FIELD_ATTRIBUTE = "lucene_rust.derived_source";
    /** {@code KNN10010DerivedSourceStoredFieldsWriter.MASK}. */
    static final Byte MASK = 0x1;

    private DerivedSource() {}

    /**
     * {@code IndexUtil.isDerivedEnabledForIndex}, for an index this node masks: k-NN's setting on,
     * core's own derived source off, no node-to-node segment replication, served by the Rust engine
     * without realtime search.
     */
    public static boolean forIndex(Settings settings) {
        return KnnSettings.KNN_INDEX.get(settings)
            && KnnSettings.DERIVED_SOURCE.get(settings)
            && IndexSettings.INDEX_DERIVED_SOURCE_SETTING.get(settings) == false
            && segRepLocal(settings) == false
            && RustEngineSupport.servedByRust(settings)
            && RustEngineSupport.REALTIME_SEARCH.get(settings) == false;
    }

    private static boolean segRepLocal(Settings settings) {
        return "SEGMENT".equalsIgnoreCase(settings.get(org.opensearch.cluster.metadata.IndexMetadata.SETTING_REPLICATION_TYPE))
            && settings.getAsBoolean(org.opensearch.cluster.metadata.IndexMetadata.SETTING_REMOTE_STORE_ENABLED, false) == false;
    }

    /**
     * {@code IndexUtil.isDerivedEnabledForField}: {@code _source} on and holding the field, and no
     * {@code copy_to}, which would read the masked value.
     */
    static boolean forField(String name, DocumentMapper mapper) {
        SourceFieldMapper source = mapper.metadataMapper(SourceFieldMapper.class);
        if (source == null || source.enabled() == false || excluded(name, source)) {
            return false;
        }
        if (mapper.mappers().getMapper(name) instanceof FieldMapper f) {
            return f.copyTo() == null || f.copyTo().copyToFields() == null || f.copyTo().copyToFields().isEmpty();
        }
        return true;
    }

    private static boolean excluded(String name, SourceFieldMapper source) {
        Collection<String> includes = source.getIncludes();
        if (includes != null && includes.isEmpty() == false && includes.stream().noneMatch(p -> Regex.simpleMatch(p, name))) {
            return true;
        }
        Collection<String> excludes = source.getExcludes();
        return excludes != null && excludes.stream().anyMatch(p -> Regex.simpleMatch(p, name));
    }

    /** {@code type} with {@link #FIELD_ATTRIBUTE}. */
    static FieldType marked(FieldType type) {
        FieldType ft = new FieldType(type);
        ft.putAttribute(FIELD_ATTRIBUTE, "true");
        ft.freeze();
        return ft;
    }

    private static boolean isMarked(IndexableField f) {
        Map<String, String> attributes = f.fieldType().getAttributes();
        return attributes != null && "true".equals(attributes.get(FIELD_ATTRIBUTE));
    }

    /**
     * {@code DerivedSourceIndexOperationListener}: before an index operation, the translog's source
     * takes each marked field's vector as indexed, and the stored {@code _source} field the mask.
     */
    public static final class Listener implements IndexingOperationListener {
        @Override
        public Engine.Index preIndex(ShardId shardId, Engine.Index operation) {
            if (operation.parsedDoc().rootDoc().getField(SourceFieldMapper.RECOVERY_SOURCE_NAME) != null) {
                return operation;
            }
            Map<String, Object> vectors = new HashMap<>();
            for (IndexableField f : operation.parsedDoc().rootDoc()) {
                if (isMarked(f) == false) {
                    continue;
                }
                if (f instanceof KnnFloatVectorField v) {
                    vectors.put(f.name(), v.vectorValue());
                } else if (f instanceof KnnByteVectorField v) {
                    byte[] b = v.vectorValue();
                    int[] ints = new int[b.length];
                    for (int i = 0; i < b.length; i++) {
                        ints[i] = b[i];
                    }
                    vectors.put(f.name(), ints);
                }
            }
            if (vectors.isEmpty()) {
                return operation;
            }
            Map<String, Function<Object, Object>> inject = new HashMap<>();
            Map<String, Function<Object, Object>> mask = new HashMap<>();
            for (Map.Entry<String, Object> e : vectors.entrySet()) {
                Object vector = e.getValue();
                inject.put(e.getKey(), o -> o == null ? null : vector);
                mask.put(e.getKey(), o -> o == null ? null : MASK);
            }
            Tuple<? extends MediaType, Map<String, Object>> source = XContentHelper.convertToMap(
                operation.parsedDoc().source(),
                true,
                operation.parsedDoc().getMediaType()
            );
            MediaType type = source.v1();
            operation.parsedDoc()
                .setSource(
                    write(type, XContentMapValues.transform(inject, true).apply(source.v2())),
                    XContentType.valueOf(type.subtype().toUpperCase(Locale.ROOT))
                );
            IndexableField stored = operation.parsedDoc().rootDoc().getField(SourceFieldMapper.NAME);
            if (stored instanceof StoredField f && f.storedValue() != null) {
                f.setBytesValue(write(type, XContentMapValues.transform(mask, true).apply(source.v2())).toBytesRef());
            }
            return operation;
        }

        private static BytesReference write(MediaType type, Map<String, Object> map) {
            try (BytesStreamOutput out = new BytesStreamOutput()) {
                XContentBuilder b = MediaTypeRegistry.contentBuilder(type, out).map(map);
                b.close();
                return out.bytes();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }
}
