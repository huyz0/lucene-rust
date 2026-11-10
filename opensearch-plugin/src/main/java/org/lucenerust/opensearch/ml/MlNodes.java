/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

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
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportRequest;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.List;

/**
 * The node-to-node part of the ml-commons port: deploy a model on nodes, undeploy it, and run a
 * prediction on a node that holds it. One request names the operation, the model and a JSON payload;
 * each node answers with a JSON payload or its error message.
 */
public final class MlNodes {
    private MlNodes() {}

    /** What a node is asked to do. */
    public enum Op {
        DEPLOY,
        UNDEPLOY,
        PREDICT,
        PROFILE
    }

    /** Deploy, undeploy and profile, on the deploy pool. */
    public static final class Action extends ActionType<Response> {
        public static final String NAME = "cluster:admin/opensearch/mlinternal/lucene_rust/nodes";
        public static final Action INSTANCE = new Action();

        private Action() {
            super(NAME, Response::new);
        }
    }

    /** Predictions forwarded to a node that holds the model, on the predict pool. */
    public static final class PredictAction extends ActionType<Response> {
        public static final String NAME = "cluster:admin/opensearch/mlinternal/lucene_rust/predict";
        public static final PredictAction INSTANCE = new PredictAction();

        private PredictAction() {
            super(NAME, Response::new);
        }
    }

    public static final class Request extends BaseNodesRequest<Request> {
        final Op op;
        final String modelId;
        final byte[] payload;

        public Request(Op op, String modelId, byte[] payload, String... nodeIds) {
            super(nodeIds);
            this.op = op;
            this.modelId = modelId;
            this.payload = payload;
        }

        public Request(StreamInput in) throws IOException {
            super(in);
            op = in.readEnum(Op.class);
            modelId = in.readOptionalString();
            payload = in.readByteArray();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeEnum(op);
            out.writeOptionalString(modelId);
            out.writeByteArray(payload);
        }
    }

    public static final class NodeRequest extends TransportRequest {
        final Request request;

        NodeRequest(Request request) {
            this.request = request;
        }

        NodeRequest(StreamInput in) throws IOException {
            super(in);
            request = new Request(in);
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            request.writeTo(out);
        }
    }

    public static final class NodeResponse extends BaseNodeResponse {
        /** The node's JSON answer, or {@code null} when it failed. */
        public final byte[] payload;
        /** Why the node failed, or {@code null}. */
        public final String error;
        /** Whether the failure is the caller's (a 400) rather than the node's (a 500). */
        public final boolean badRequest;

        NodeResponse(DiscoveryNode node, byte[] payload, String error, boolean badRequest) {
            super(node);
            this.payload = payload;
            this.error = error;
            this.badRequest = badRequest;
        }

        NodeResponse(StreamInput in) throws IOException {
            super(in);
            payload = in.readBoolean() ? in.readByteArray() : null;
            error = in.readOptionalString();
            badRequest = in.readBoolean();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeBoolean(payload != null);
            if (payload != null) {
                out.writeByteArray(payload);
            }
            out.writeOptionalString(error);
            out.writeBoolean(badRequest);
        }
    }

    public static final class Response extends BaseNodesResponse<NodeResponse> {
        public Response(ClusterName clusterName, List<NodeResponse> nodes, List<FailedNodeException> failures) {
            super(clusterName, nodes, failures);
        }

        public Response(StreamInput in) throws IOException {
            super(in);
        }

        @Override
        protected List<NodeResponse> readNodesFrom(StreamInput in) throws IOException {
            return in.readList(NodeResponse::new);
        }

        @Override
        protected void writeNodesTo(StreamOutput out, List<NodeResponse> nodes) throws IOException {
            out.writeList(nodes);
        }
    }

    public static final class TransportAction extends Base {
        @Inject
        public TransportAction(ThreadPool threadPool, ClusterService clusterService, TransportService transportService, ActionFilters filters) {
            super(Action.NAME, MlCommons.DEPLOY_THREAD_POOL, threadPool, clusterService, transportService, filters);
        }
    }

    public static final class PredictTransportAction extends Base {
        @Inject
        public PredictTransportAction(ThreadPool threadPool, ClusterService clusterService, TransportService transportService, ActionFilters filters) {
            super(PredictAction.NAME, MlCommons.PREDICT_THREAD_POOL, threadPool, clusterService, transportService, filters);
        }
    }

    abstract static class Base extends TransportNodesAction<Request, Response, NodeRequest, NodeResponse> {
        Base(
            String name,
            String executor,
            ThreadPool threadPool,
            ClusterService clusterService,
            TransportService transportService,
            ActionFilters filters
        ) {
            super(
                name,
                threadPool,
                clusterService,
                transportService,
                filters,
                Request::new,
                NodeRequest::new,
                executor,
                NodeResponse.class
            );
        }

        @Override
        protected Response newResponse(Request request, List<NodeResponse> responses, List<FailedNodeException> failures) {
            return new Response(clusterService.getClusterName(), responses, failures);
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
            DiscoveryNode local = clusterService.localNode();
            try {
                byte[] payload = MlCommons.get().onNode(request.request.op, request.request.modelId, request.request.payload);
                return new NodeResponse(local, payload, null, false);
            } catch (IllegalArgumentException e) {
                return new NodeResponse(local, null, e.getMessage(), true);
            } catch (Exception e) {
                return new NodeResponse(local, null, e.getMessage() == null ? e.toString() : e.getMessage(), false);
            }
        }
    }
}
