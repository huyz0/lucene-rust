/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.neural;

import org.lucenerust.opensearch.neural.rest.RestNeuralStatsAction;
import org.lucenerust.opensearch.neural.settings.NeuralSearchSettingsAccessor;
import org.lucenerust.opensearch.neural.stats.events.EventStatsManager;
import org.lucenerust.opensearch.neural.stats.info.InfoStatsManager;
import org.lucenerust.opensearch.neural.stats.metrics.MetricStatsManager;
import org.lucenerust.opensearch.neural.transport.NeuralStatsAction;
import org.lucenerust.opensearch.neural.transport.NeuralStatsTransportAction;
import org.lucenerust.opensearch.neural.util.NeuralSearchClusterUtil;
import org.lucenerust.opensearch.neural.util.PipelineServiceUtil;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.plugins.ActionPlugin.ActionHandler;
import org.opensearch.rest.RestHandler;

import java.util.List;

/**
 * neural-search's stats API ({@code GET _plugins/_neural/stats}), wired as {@code NeuralSearch}
 * wires it: the cluster helper, the settings accessor the event counters read, and the managers the
 * transport action is injected with.
 */
public final class NeuralStats {
    private static volatile NeuralSearchSettingsAccessor settingsAccessor;

    private NeuralStats() {}

    /** {@code NeuralSearch.createComponents}'s part: the components the transport action needs. */
    public static List<Object> initialize(ClusterService clusterService, IndexNameExpressionResolver resolver, Settings settings) {
        NeuralSearchClusterUtil.instance().initialize(clusterService, resolver);
        settingsAccessor = new NeuralSearchSettingsAccessor(clusterService, settings);
        InfoStatsManager info = new InfoStatsManager(NeuralSearchClusterUtil.instance(), settingsAccessor, new PipelineServiceUtil(clusterService));
        EventStatsManager.instance().initialize(settingsAccessor);
        return List.of(EventStatsManager.instance(), info, new MetricStatsManager());
    }

    public static RestHandler restHandler() {
        return new RestNeuralStatsAction(settingsAccessor, NeuralSearchClusterUtil.instance());
    }

    public static ActionHandler<?, ?> actionHandler() {
        return new ActionHandler<>(NeuralStatsAction.INSTANCE, NeuralStatsTransportAction.class);
    }
}
