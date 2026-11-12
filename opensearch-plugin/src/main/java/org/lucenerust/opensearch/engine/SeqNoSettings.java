/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Leaner sequence-number fields, opt-in per index (Elasticsearch's {@code index.seq_no.index_options}
 * and {@code index.disable_sequence_numbers}).
 *
 * <ul>
 *   <li>{@code index.lucene_rust.seq_no.index_options: doc_values_only} indexes {@code _seq_no} as
 *       doc values with a skip index and no points. Every query on it (recovery, history
 *       snapshots, searches on {@code _seq_no}) runs on the doc values, with the same results.
 *   <li>{@code index.lucene_rust.disable_sequence_numbers: true} (needs {@code doc_values_only})
 *       also makes merges write {@code UNASSIGNED_SEQ_NO} and {@code UNASSIGNED_PRIMARY_TERM} for
 *       every document whose operation history is no longer retained. Replication and recovery
 *       still work, but get and search report no {@code _seq_no} or {@code _primary_term} for
 *       such documents, so an {@code if_seq_no} write on one conflicts and an update of one runs
 *       without a sequence-number check.
 * </ul>
 *
 * Both are fixed at index creation. The segments stay readable by stock Lucene.
 */
public final class SeqNoSettings {
    private SeqNoSettings() {}

    /** Indexes {@code _seq_no} as points and doc values, as stock OpenSearch does. */
    public static final String POINTS = "points";
    /** Indexes {@code _seq_no} as doc values with a skip index only. */
    public static final String DOC_VALUES_ONLY = "doc_values_only";

    public static final Setting<String> INDEX_OPTIONS = Setting.simpleString(
        "index.lucene_rust.seq_no.index_options",
        POINTS,
        value -> {
            if (POINTS.equals(value) == false && DOC_VALUES_ONLY.equals(value) == false) {
                throw new IllegalArgumentException(
                    "[index.lucene_rust.seq_no.index_options] must be [" + POINTS + "] or [" + DOC_VALUES_ONLY + "], got [" + value + "]"
                );
            }
        },
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    public static final Setting<Boolean> DISABLE_SEQUENCE_NUMBERS = Setting.boolSetting(
        "index.lucene_rust.disable_sequence_numbers",
        false,
        new Setting.Validator<>() {
            @Override
            public void validate(Boolean value) {}

            @Override
            public void validate(Boolean value, Map<Setting<?>, Object> settings) {
                if (value && DOC_VALUES_ONLY.equals(settings.get(INDEX_OPTIONS)) == false) {
                    throw new IllegalArgumentException(
                        "[index.lucene_rust.disable_sequence_numbers] requires [index.lucene_rust.seq_no.index_options] to be ["
                            + DOC_VALUES_ONLY
                            + "]"
                    );
                }
            }

            @Override
            public Iterator<Setting<?>> settings() {
                List<Setting<?>> dependencies = List.of(INDEX_OPTIONS);
                return dependencies.iterator();
            }
        },
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    public static List<Setting<?>> settings() {
        return List.of(INDEX_OPTIONS, DISABLE_SEQUENCE_NUMBERS);
    }

    /** Whether {@code settings} index {@code _seq_no} without points. */
    static boolean docValuesOnly(Settings settings) {
        return DOC_VALUES_ONLY.equals(INDEX_OPTIONS.get(settings));
    }

    /** Whether merges write sentinels for sequence numbers no longer retained. */
    static boolean sentinels(Settings settings) {
        return DISABLE_SEQUENCE_NUMBERS.get(settings);
    }
}
