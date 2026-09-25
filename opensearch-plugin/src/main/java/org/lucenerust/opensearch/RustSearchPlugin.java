/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.IndexScopedSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.settings.SettingsFilter;
import org.opensearch.plugins.ActionPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.plugins.SearchPlugin;
import org.opensearch.rest.RestController;
import org.opensearch.rest.RestHandler;
import org.opensearch.search.query.QueryPhaseSearcher;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;

/**
 * lucene-rust for OpenSearch: serves the query phase of supported searches from the Rust engine
 * ({@link RustQueryPhaseSearcher}) and everything else from Lucene.
 *
 * <p>The native library is loaded, and its ABI checked, in the constructor: a plugin whose library
 * is missing or stale stops the node at startup with one clear message rather than failing
 * searches later.
 */
public class RustSearchPlugin extends Plugin implements SearchPlugin, ActionPlugin {
    private final NativeReaders readers = new NativeReaders();
    private final SearchStats stats = new SearchStats();
    private final String libraryPath;

    public RustSearchPlugin() {
        this.libraryPath = NativeLibrary.load(pluginDir());
    }

    private static Path pluginDir() {
        try {
            return Path.of(RustSearchPlugin.class.getProtectionDomain().getCodeSource().getLocation().toURI()).getParent();
        } catch (URISyntaxException e) {
            throw new IllegalStateException("lucene-rust: cannot locate the plugin directory", e);
        }
    }

    @Override
    public Optional<QueryPhaseSearcher> getQueryPhaseSearcher() {
        return Optional.of(new RustQueryPhaseSearcher(readers, stats));
    }

    @Override
    public List<Setting<?>> getSettings() {
        return List.of(RustQueryPhaseSearcher.ENABLED, RustQueryPhaseSearcher.NATIVE_SHAPES);
    }

    @Override
    public List<RestHandler> getRestHandlers(
        Settings settings,
        RestController restController,
        ClusterSettings clusterSettings,
        IndexScopedSettings indexScopedSettings,
        SettingsFilter settingsFilter,
        IndexNameExpressionResolver indexNameExpressionResolver,
        Supplier<DiscoveryNodes> nodesInCluster
    ) {
        return List.of(new RestStatsAction(stats, readers, libraryPath));
    }

    @Override
    public void close() {
        readers.closeAll();
    }
}
