/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.rest;

import org.opensearch.action.ActionType;
import org.opensearch.action.FailedNodeException;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.nodes.BaseNodeResponse;
import org.opensearch.action.support.nodes.BaseNodesRequest;
import org.opensearch.action.support.nodes.BaseNodesResponse;
import org.opensearch.action.support.nodes.TransportNodesAction;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportRequest;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * k-NN's {@code KNNStatsAction}: each node's {@link KnnStats} under {@code nodes}, and the cluster
 * stats beside them, under the stock plugin's action and wire names.
 */
public final class KnnStatsTransport {
    private KnnStatsTransport() {}

    /** {@code KNNStatsAction}. */
    public static final class Action extends ActionType<Response> {
        public static final Action INSTANCE = new Action();
        public static final String NAME = "cluster:admin/knn_stats_action";

        private Action() {
            super(NAME, Response::new);
        }
    }

    /** {@code KNNStatsRequest}: the nodes, and the stats to report. */
    public static final class Request extends BaseNodesRequest<Request> {
        public static final String ALL = "_all";
        private final Set<String> valid;
        private final Set<String> wanted;

        public Request(String... nodeIds) {
            super(nodeIds);
            this.valid = KnnStats.names();
            this.wanted = new HashSet<>();
        }

        public Request(StreamInput in) throws IOException {
            super(in);
            this.valid = in.readSet(StreamInput::readString);
            this.wanted = in.readSet(StreamInput::readString);
        }

        public void all() {
            wanted.addAll(valid);
        }

        /** Asks for {@code stat}; false when it names none. */
        public boolean add(String stat) {
            if (valid.contains(stat)) {
                wanted.add(stat);
                return true;
            }
            return false;
        }

        Set<String> wanted() {
            return wanted;
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeStringCollection(valid);
            out.writeStringCollection(wanted);
        }
    }

    /** {@code KNNStatsNodeRequest}. */
    public static final class NodeRequest extends TransportRequest {
        private final Request request;

        public NodeRequest(Request request) {
            this.request = request;
        }

        public NodeRequest(StreamInput in) throws IOException {
            super(in);
            this.request = new Request(in);
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            request.writeTo(out);
        }
    }

    /** {@code KNNStatsNodeResponse}. */
    public static final class NodeResponse extends BaseNodeResponse {
        private final Map<String, Object> stats;

        public NodeResponse(DiscoveryNode node, Map<String, Object> stats) {
            super(node);
            this.stats = stats;
        }

        public NodeResponse(StreamInput in) throws IOException {
            super(in);
            this.stats = in.readMap(StreamInput::readString, StreamInput::readGenericValue);
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeMap(stats, StreamOutput::writeString, StreamOutput::writeGenericValue);
        }
    }

    /** {@code KNNStatsResponse}: {@code _nodes}, {@code cluster_name}, the cluster stats, {@code nodes}. */
    public static final class Response extends BaseNodesResponse<NodeResponse> implements ToXContentObject {
        private final Map<String, Object> cluster;

        public Response(ClusterName name, List<NodeResponse> nodes, List<FailedNodeException> failures, Map<String, Object> cluster) {
            super(name, nodes, failures);
            this.cluster = cluster;
        }

        public Response(StreamInput in) throws IOException {
            super(new ClusterName(in), in.readList(NodeResponse::new), in.readList(FailedNodeException::new));
            this.cluster = in.readMap();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeMap(cluster);
        }

        @Override
        protected List<NodeResponse> readNodesFrom(StreamInput in) throws IOException {
            return in.readList(NodeResponse::new);
        }

        @Override
        protected void writeNodesTo(StreamOutput out, List<NodeResponse> nodes) throws IOException {
            out.writeList(nodes);
        }

        @Override
        public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
            for (Map.Entry<String, Object> e : cluster.entrySet()) {
                builder.field(e.getKey(), e.getValue());
            }
            builder.startObject("nodes");
            for (NodeResponse node : getNodes()) {
                builder.startObject(node.getNode().getId());
                for (Map.Entry<String, Object> e : node.stats.entrySet()) {
                    builder.field(e.getKey(), e.getValue());
                }
                builder.endObject();
            }
            builder.endObject();
            return builder;
        }
    }

    /** {@code KNNStatsTransportAction}. */
    public static final class TransportAction extends TransportNodesAction<Request, Response, NodeRequest, NodeResponse> {
        @Inject
        public TransportAction(
            ThreadPool threadPool,
            ClusterService clusterService,
            TransportService transportService,
            ActionFilters actionFilters
        ) {
            super(
                Action.NAME,
                threadPool,
                clusterService,
                transportService,
                actionFilters,
                Request::new,
                NodeRequest::new,
                ThreadPool.Names.MANAGEMENT,
                NodeResponse.class
            );
        }

        @Override
        protected Response newResponse(Request request, List<NodeResponse> responses, List<FailedNodeException> failures) {
            Map<String, Object> cluster = new HashMap<>();
            for (String name : KnnStats.CLUSTER) {
                if (request.wanted().contains(name)) {
                    cluster.put(name, KnnStats.value(name));
                }
            }
            return new Response(clusterService.getClusterName(), responses, failures, cluster);
        }

        @Override
        protected NodeRequest newNodeRequest(Request request) {
            return new NodeRequest(request);
        }

        @Override
        protected NodeResponse newNodeResponse(StreamInput in) throws IOException {
            return new NodeResponse(in);
        }

        @Override
        protected NodeResponse nodeOperation(NodeRequest request) {
            Map<String, Object> values = new HashMap<>();
            for (String name : KnnStats.reported()) {
                if (KnnStats.CLUSTER.contains(name) == false && request.request.wanted().contains(name)) {
                    values.put(name, KnnStats.value(name));
                }
            }
            return new NodeResponse(clusterService.localNode(), values);
        }
    }
}
