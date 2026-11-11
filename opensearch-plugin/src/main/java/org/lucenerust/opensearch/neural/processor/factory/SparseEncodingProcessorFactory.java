/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.factory;

import org.opensearch.cluster.service.ClusterService;
import org.opensearch.env.Environment;
import org.opensearch.ingest.AbstractBatchingProcessor;
import org.lucenerust.opensearch.neural.ml.MLCommonsClientAccessor;
import org.lucenerust.opensearch.neural.processor.SparseEncodingProcessor;
import org.lucenerust.opensearch.neural.processor.optimization.TextEmbeddingInferenceFilter;
import org.lucenerust.opensearch.neural.util.prune.PruneType;
import org.lucenerust.opensearch.neural.util.prune.PruneUtils;
import org.opensearch.transport.client.OpenSearchClient;
import java.util.Locale;
import java.util.Map;
import static org.opensearch.ingest.ConfigurationUtils.readBooleanProperty;
import static org.opensearch.ingest.ConfigurationUtils.readDoubleProperty;
import static org.opensearch.ingest.ConfigurationUtils.readMap;
import static org.opensearch.ingest.ConfigurationUtils.readOptionalStringProperty;
import static org.opensearch.ingest.ConfigurationUtils.readStringProperty;
import static org.lucenerust.opensearch.neural.processor.SparseEncodingProcessor.DEFAULT_SKIP_EXISTING;
import static org.lucenerust.opensearch.neural.processor.SparseEncodingProcessor.SKIP_EXISTING;
import static org.lucenerust.opensearch.neural.processor.SparseEncodingProcessor.TYPE;
import static org.lucenerust.opensearch.neural.processor.TextEmbeddingProcessor.FIELD_MAP_FIELD;
import static org.lucenerust.opensearch.neural.processor.TextEmbeddingProcessor.MODEL_ID_FIELD;

/**
 * Factory for sparse encoding ingest processor for ingestion pipeline. Instantiates processor based on user provided input.
 */
public class SparseEncodingProcessorFactory extends AbstractBatchingProcessor.Factory {
    private static final org.apache.logging.log4j.Logger log = org.apache.logging.log4j.LogManager.getLogger(SparseEncodingProcessorFactory.class);
    private final OpenSearchClient openSearchClient;
    private final MLCommonsClientAccessor clientAccessor;
    private final Environment environment;
    private final ClusterService clusterService;

    public SparseEncodingProcessorFactory(OpenSearchClient openSearchClient, MLCommonsClientAccessor clientAccessor, Environment environment, ClusterService clusterService) {
        super(TYPE);
        this.clientAccessor = clientAccessor;
        this.environment = environment;
        this.clusterService = clusterService;
        this.openSearchClient = openSearchClient;
    }

    @Override
    protected AbstractBatchingProcessor newProcessor(String tag, String description, int batchSize, Map<String, Object> config) {
        String modelId = readStringProperty(TYPE, tag, config, MODEL_ID_FIELD);
        Map<String, Object> fieldMap = readMap(TYPE, tag, config, FIELD_MAP_FIELD);
        boolean skipExisting = readBooleanProperty(TYPE, tag, config, SKIP_EXISTING, DEFAULT_SKIP_EXISTING);
        // if the field is miss, will return PruneType.None
        PruneType pruneType = PruneType.fromString(readOptionalStringProperty(TYPE, tag, config, PruneUtils.PRUNE_TYPE_FIELD));
        float pruneRatio = 0;
        if (pruneType != PruneType.NONE) {
            // if we have prune type, then prune ratio field must have value
            // readDoubleProperty will throw exception if value is not present
            pruneRatio = readDoubleProperty(TYPE, tag, config, PruneUtils.PRUNE_RATIO_FIELD).floatValue();
            if (!PruneUtils.isValidPruneRatio(pruneType, pruneRatio)) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "Illegal prune_ratio %f for prune_type: %s. %s", pruneRatio, pruneType.getValue(), PruneUtils.getValidPruneRatioDescription(pruneType)));
            }
        } else if (config.containsKey(PruneUtils.PRUNE_RATIO_FIELD)) {
            // if we don't have prune type, then prune ratio field must not have value
            throw new IllegalArgumentException("prune_ratio field is not supported when prune_type is not provided");
        }
        return new SparseEncodingProcessor(tag, description, batchSize, modelId, fieldMap, skipExisting, skipExisting ? new TextEmbeddingInferenceFilter(fieldMap) : null, pruneType, pruneRatio, openSearchClient, clientAccessor, environment, clusterService);
    }
}
