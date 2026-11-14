/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.opensearch.action.admin.indices.create.CreateIndexRequest;
import org.opensearch.action.admin.indices.create.CreateIndexResponse;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.action.ActionListener;
import org.opensearch.indices.SystemIndexDescriptor;
import org.opensearch.ResourceAlreadyExistsException;
import org.opensearch.action.support.ContextPreservingActionListener;
import org.opensearch.transport.client.Client;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.function.Supplier;

/**
 * ml-commons' system indices, created on first use with the stock mappings and index settings
 * ({@code MLIndex}, {@code MLIndicesHandler}), so a cluster that ran stock ml-commons keeps its
 * models, groups and tasks, and stock reads what this plugin writes.
 */
public final class MlIndices {
    public static final String MODEL_GROUP = ".plugins-ml-model-group";
    public static final String MODEL = ".plugins-ml-model";
    public static final String TASK = ".plugins-ml-task";
    public static final String CONFIG = ".plugins-ml-config";
    public static final String CONNECTOR = ".plugins-ml-connector";
    public static final String AGENT = ".plugins-ml-agent";
    public static final String MEMORY_META = ".plugins-ml-memory-meta";
    public static final String MEMORY_MESSAGE = ".plugins-ml-memory-message";

    private final Client client;
    private final ClusterService clusterService;

    MlIndices(Client client, ClusterService clusterService) {
        this.client = client;
        this.clusterService = clusterService;
    }

    /** The descriptors {@code MachineLearningPlugin.getSystemIndexDescriptors} registers. */
    public static List<SystemIndexDescriptor> descriptors() {
        return List.of(
            new SystemIndexDescriptor(MODEL_GROUP, "ML Commons Model Group System Index"),
            new SystemIndexDescriptor(MODEL, "ML Commons Model System Index"),
            new SystemIndexDescriptor(TASK, "ML Commons Task System Index"),
            new SystemIndexDescriptor(CONFIG, "ML Commons Configuration System Index"),
            new SystemIndexDescriptor(CONNECTOR, "ML Commons Connector System Index"),
            new SystemIndexDescriptor(AGENT, "ML Commons Agent System Index"),
            new SystemIndexDescriptor(MEMORY_META, "ML Commons Memory Meta System Index"),
            new SystemIndexDescriptor(MEMORY_MESSAGE, "ML Commons Memory Message System Index")
        );
    }

    private static String mapping(String index) {
        String file = "mappings/" + index.substring(".plugins-".length()).replace('-', '_') + ".json";
        try (InputStream in = MlIndices.class.getResourceAsStream(file)) {
            if (in == null) {
                throw new IllegalStateException("no mapping resource " + file);
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static Settings settings(String index) {
        return Settings.builder()
            .put("index.number_of_shards", 1)
            .put("index.auto_expand_replicas", index.equals(CONFIG) ? "0-all" : "0-1")
            .build();
    }

    /** Runs {@code then} once {@code index} exists, creating it if needed. */
    @SuppressWarnings("deprecation")
    public void ensure(String index, ActionListener<Void> then) {
        if (clusterService.state().metadata().hasIndex(index)) {
            then.onResponse(null);
            return;
        }
        CreateIndexRequest request = new CreateIndexRequest(index).mapping(mapping(index), XContentType.JSON).settings(settings(index));
        system(client, l -> client.admin().indices().create(request, l), ActionListener.<CreateIndexResponse>wrap(r -> then.onResponse(null), e -> {
            if (e instanceof ResourceAlreadyExistsException) {
                then.onResponse(null);
            } else {
                then.onFailure(e);
            }
        }));
    }

    /**
     * Runs {@code call} with the thread context stashed (system index access, as ml-commons does for its
     * own indices), answering {@code listener} in the caller's context.
     */
    @SuppressWarnings("try")
    public static <T> void system(Client client, java.util.function.Consumer<ActionListener<T>> call, ActionListener<? super T> listener) {
        ThreadContext context = client.threadPool().getThreadContext();
        Supplier<ThreadContext.StoredContext> restore = context.newRestorableContext(false);
        try (ThreadContext.StoredContext ignored = context.stashContext()) {
            call.accept(new ContextPreservingActionListener<T>(restore, ActionListener.wrap(listener::onResponse, listener::onFailure)));
        }
    }
}
