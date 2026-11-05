/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.sparse;

import org.opensearch.common.settings.Setting;

/** neural-search's {@code SparseSettings} that this plugin serves. */
public final class SparseSettings {
    /**
     * {@code index.sparse}: the Rust writer records {@link SparseCodec#NAME} for the index's segments
     * and clusters their {@code sparse_vector} fields' postings, as neural-search's codec does.
     */
    public static final Setting<Boolean> SPARSE_INDEX = Setting.boolSetting(
        "index.sparse",
        false,
        Setting.Property.IndexScope,
        Setting.Property.Final,
        Setting.Property.UnmodifiableOnRestore
    );

    private SparseSettings() {}
}
