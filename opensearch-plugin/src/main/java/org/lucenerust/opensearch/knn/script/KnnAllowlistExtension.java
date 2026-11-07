/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.script;

import org.lucenerust.opensearch.RustSearchPlugin;
import org.opensearch.painless.spi.Allowlist;
import org.opensearch.painless.spi.AllowlistLoader;
import org.opensearch.painless.spi.PainlessExtension;
import org.opensearch.script.FieldScript;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptedMetricAggContexts;

import java.util.List;
import java.util.Map;

/**
 * k-NN's {@code KNNAllowlistExtension}: the Painless functions over {@code doc['field']} of a {@code
 * knn_vector} field ({@code knn_allowlist.txt}), in the contexts stock allows them. Nothing when the
 * stock k-NN plugin is installed, since it registers the same functions.
 */
public final class KnnAllowlistExtension implements PainlessExtension {
    @Override
    public Map<ScriptContext<?>, List<Allowlist>> getContextAllowlists() {
        if (RustSearchPlugin.stockKnnInstalled()) {
            return Map.of();
        }
        List<Allowlist> list = List.of(AllowlistLoader.loadFromResourceFiles(KnnAllowlistExtension.class, "knn_allowlist.txt"));
        return Map.of(
            ScoreScript.CONTEXT,
            list,
            ScriptedMetricAggContexts.InitScript.CONTEXT,
            list,
            ScriptedMetricAggContexts.MapScript.CONTEXT,
            list,
            ScriptedMetricAggContexts.CombineScript.CONTEXT,
            list,
            ScriptedMetricAggContexts.ReduceScript.CONTEXT,
            list,
            FieldScript.CONTEXT,
            list
        );
    }
}
