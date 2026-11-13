/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.logs;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.action.admin.indices.template.put.PutComponentTemplateAction;
import org.opensearch.cluster.ClusterChangedEvent;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.ClusterStateListener;
import org.opensearch.cluster.metadata.ComponentTemplate;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.gateway.GatewayService;
import org.opensearch.transport.client.Client;

/**
 * The logs preset: two component templates, {@code logs_preset@settings} and {@code
 * logs_preset@mappings}, which an index template opts into by naming them in its {@code
 * composed_of}. They change nothing until a template names them.
 *
 * <p>The elected cluster manager puts each one when it is missing or older than this plugin's
 * version of it, so a node upgrade brings new versions in and a template a user saved under its own
 * name is never touched. {@link #ENABLED} off stops that.
 */
public final class LogsPreset implements ClusterStateListener {
    private static final Logger logger = LogManager.getLogger(LogsPreset.class);

    /** Whether the elected cluster manager keeps the preset's component templates registered. */
    public static final Setting<Boolean> ENABLED = Setting.boolSetting(
        "indices.logs_preset.enabled",
        true,
        Setting.Property.NodeScope
    );

    /** The templates' names, in the order an index template's {@code composed_of} lists them. */
    public static final String SETTINGS = "logs_preset@settings";
    public static final String MAPPINGS = "logs_preset@mappings";

    private final Client client;
    private final Map<String, ComponentTemplate> templates;
    private final AtomicBoolean putting = new AtomicBoolean();
    /** Templates whose put failed: logged once, not retried on every state change. */
    private final java.util.Set<String> failed = java.util.concurrent.ConcurrentHashMap.newKeySet();

    public LogsPreset(Client client) {
        this.client = client;
        this.templates = templates();
    }

    /** The preset's templates by name, parsed from the plugin's resources. */
    public static Map<String, ComponentTemplate> templates() {
        Map<String, ComponentTemplate> t = new LinkedHashMap<>();
        for (String name : new String[] { SETTINGS, MAPPINGS }) {
            t.put(name, load(name));
        }
        return t;
    }

    static ComponentTemplate load(String name) {
        try (
            InputStream in = LogsPreset.class.getResourceAsStream(name + ".json");
            XContentParser p = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                in
            )
        ) {
            return ComponentTemplate.parse(p);
        } catch (IOException e) {
            throw new UncheckedIOException("lucene-rust: cannot read the component template [" + name + "]", e);
        }
    }

    /** The names whose installed template is missing or older than the preset's. */
    static java.util.List<String> stale(ClusterState state, Map<String, ComponentTemplate> templates) {
        Map<String, ComponentTemplate> installed = state.metadata().componentTemplates();
        java.util.List<String> out = new java.util.ArrayList<>();
        for (Map.Entry<String, ComponentTemplate> e : templates.entrySet()) {
            ComponentTemplate have = installed.get(e.getKey());
            Long version = have == null ? null : have.version();
            if (version == null || version < e.getValue().version()) {
                out.add(e.getKey());
            }
        }
        return out;
    }

    @Override
    public void clusterChanged(ClusterChangedEvent event) {
        ClusterState state = event.state();
        if (event.localNodeClusterManager() == false || state.blocks().hasGlobalBlock(GatewayService.STATE_NOT_RECOVERED_BLOCK)) {
            return;
        }
        java.util.List<String> names = stale(state, templates);
        names.removeAll(failed);
        if (names.isEmpty() || putting.compareAndSet(false, true) == false) {
            return;
        }
        put(names, 0);
    }

    /** Puts {@code names[i..]} one after another, then lets the next state change check again. */
    private void put(java.util.List<String> names, int i) {
        if (i == names.size()) {
            putting.set(false);
            return;
        }
        String name = names.get(i);
        PutComponentTemplateAction.Request request = new PutComponentTemplateAction.Request(name).componentTemplate(templates.get(name))
            .cause("lucene-rust logs preset");
        // As the node itself, not as whichever request's user triggered this state change.
        ThreadContext.StoredContext restore = client.threadPool().getThreadContext().stashContext();
        try {
            client.execute(PutComponentTemplateAction.INSTANCE, request, ActionListener.wrap(r -> put(names, i + 1), e -> {
                logger.warn("lucene-rust: cannot put the component template [" + name + "]", e);
                failed.add(name);
                put(names, i + 1);
            }));
        } finally {
            restore.restore();
        }
    }
}
