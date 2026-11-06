/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;

import java.util.List;
import java.util.function.Function;

/**
 * {@code index.lucene_rust.bp_reorder.*}: Lucene's {@code BPReorderingMergePolicy} around the
 * shard's merge policy. Merges big enough, and every forced merge, renumber their documents with
 * {@code BPIndexReorderer} (recursive graph bisection), so documents that share terms sit together:
 * postings get smaller and queries faster. The merged segments are ordinary Lucene segments that any
 * Lucene reads. Off by default; a sorted index keeps its sort and is never reordered.
 */
public final class BpReorderSettings {
    private BpReorderSettings() {}

    private static final String PREFIX = "index.lucene_rust.bp_reorder.";

    /** Whether merges reorder documents. */
    public static final Setting<Boolean> ENABLED = Setting.boolSetting(
        PREFIX + "enabled",
        false,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /** The fields whose terms decide the order; every indexed field when empty. */
    public static final Setting<List<String>> FIELDS = Setting.listSetting(
        PREFIX + "fields",
        List.of(),
        Function.identity(),
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /** {@code BPIndexReorderer.setMinDocFreq}: rarer terms are ignored. */
    public static final Setting<Integer> MIN_DOC_FREQ = Setting.intSetting(
        PREFIX + "min_doc_freq",
        4096,
        1,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /** {@code BPIndexReorderer.setMaxDocFreq}: terms in more than this share of documents are ignored. */
    public static final Setting<Double> MAX_DOC_FREQ = Setting.doubleSetting(
        PREFIX + "max_doc_freq",
        1.0,
        Double.MIN_VALUE,
        1.0,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /** {@code BPIndexReorderer.setMinPartitionSize}: partitions this small are not split further. */
    public static final Setting<Integer> MIN_PARTITION_SIZE = Setting.intSetting(
        PREFIX + "min_partition_size",
        32,
        1,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /** {@code BPIndexReorderer.setMaxIters}: swap rounds per partition. */
    public static final Setting<Integer> MAX_ITERS = Setting.intSetting(
        PREFIX + "max_iters",
        20,
        1,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /** {@code BPIndexReorderer.setRAMBudgetMB}: a merge that needs more is left in its order. */
    public static final Setting<ByteSizeValue> RAM_BUDGET = Setting.byteSizeSetting(
        PREFIX + "ram_budget",
        new ByteSizeValue(256, ByteSizeUnit.MB),
        new ByteSizeValue(1, ByteSizeUnit.MB),
        new ByteSizeValue(Long.MAX_VALUE, ByteSizeUnit.BYTES),
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /** {@code setMinNaturalMergeNumDocs}: smaller natural merges keep their order. */
    public static final Setting<Integer> MIN_MERGE_DOCS = Setting.intSetting(
        PREFIX + "min_merge_docs",
        65536,
        1,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /**
     * {@code setMinNaturalMergeRatioFromBiggestSegment}: natural merges smaller than this share of
     * the biggest segment keep their order.
     */
    public static final Setting<Double> MIN_MERGE_RATIO = Setting.doubleSetting(
        PREFIX + "min_merge_ratio",
        0.0,
        0.0,
        Math.nextDown(1.0),
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /** Every setting, for the plugin to register. */
    public static List<Setting<?>> settings() {
        return List.of(ENABLED, FIELDS, MIN_DOC_FREQ, MAX_DOC_FREQ, MIN_PARTITION_SIZE, MAX_ITERS, RAM_BUDGET, MIN_MERGE_DOCS, MIN_MERGE_RATIO);
    }

    /** The values {@link RustIndexWriter#setBpReorder} sends, as one comparable record. */
    record Values(
        boolean enabled,
        List<String> fields,
        int minDocFreq,
        float maxDocFreq,
        int minPartitionSize,
        int maxIters,
        double ramBudgetMb,
        int minMergeDocs,
        float minMergeRatio
    ) {
        static Values of(Settings settings) {
            return new Values(
                ENABLED.get(settings),
                List.copyOf(FIELDS.get(settings)),
                MIN_DOC_FREQ.get(settings),
                MAX_DOC_FREQ.get(settings).floatValue(),
                MIN_PARTITION_SIZE.get(settings),
                MAX_ITERS.get(settings),
                RAM_BUDGET.get(settings).getBytes() / (1024.0 * 1024.0),
                MIN_MERGE_DOCS.get(settings),
                MIN_MERGE_RATIO.get(settings).floatValue()
            );
        }
    }
}
