package org.lucenerust.opensearch.highlight;

import org.opensearch.common.settings.Setting;
import org.opensearch.index.IndexSettings;
import org.opensearch.search.fetch.subphase.highlight.FieldHighlightContext;

/**
 * Whether the highlighters also take the cases OpenSearch's highlighters used to keep: a hit on a
 * realtime view, a derived field, a text with an unpaired surrogate, a Thai locale, a request or
 * hit OpenSearch refuses (its exceptions are then thrown from the native side), a
 * {@code more_like_this} or a {@code bool} whose clauses are not in the extractor's order. Off,
 * those run on OpenSearch's highlighters, as they did before they were ported.
 */
public final class ExtendedHighlighting {
    public static final Setting<Boolean> ENABLED = Setting.boolSetting(
        "index.highlight.accelerated_extended",
        false,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    private ExtendedHighlighting() {}

    /** The index setting for the field being highlighted. */
    public static boolean on(FieldHighlightContext fieldContext) {
        IndexSettings settings = fieldContext.context.getIndexSettings();
        return ENABLED.get(settings.getSettings());
    }
}
