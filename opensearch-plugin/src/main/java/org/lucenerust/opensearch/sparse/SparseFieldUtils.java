/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.sparse;

import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.mapper.MapperService;

import java.util.Collections;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/** neural-search's {@code SparseFieldUtils}: the {@code sparse_vector} fields an index maps. */
public final class SparseFieldUtils {
    private SparseFieldUtils() {}

    /** The paths of the index's {@code sparse_vector} fields (nested ones by full path); empty unless {@code index.sparse}. */
    public static Set<String> getSparseAnnFields(String index, ClusterService clusterService) {
        return getSparseAnnFields(index, clusterService, getMaxDepth(index, clusterService));
    }

    @SuppressWarnings("unchecked")
    public static Set<String> getSparseAnnFields(String index, ClusterService clusterService, long maxDepth) {
        if (index == null) {
            return Collections.emptySet();
        }
        final IndexMetadata metadata = Optional.ofNullable(clusterService)
            .map(ClusterService::state)
            .map(ClusterState::metadata)
            .map(m -> m.index(index))
            .orElse(null);
        if (metadata == null || SparseSettings.SPARSE_INDEX.get(metadata.getSettings()) == false) {
            return Collections.emptySet();
        }
        MappingMetadata mappingMetadata = metadata.mapping();
        if (mappingMetadata == null || mappingMetadata.sourceAsMap() == null) {
            return Collections.emptySet();
        }
        Object properties = mappingMetadata.sourceAsMap().get("properties");
        if ((properties instanceof Map) == false) {
            return Collections.emptySet();
        }
        Set<String> sparseAnnFields = new HashSet<>();
        collectSparseAnnFields((Map<String, Object>) properties, "", sparseAnnFields, 1, maxDepth);
        return sparseAnnFields;
    }

    /** The index's {@code index.mapping.depth.limit}. */
    public static long getMaxDepth(String index, ClusterService clusterService) {
        Settings settings = Optional.ofNullable(clusterService)
            .map(ClusterService::state)
            .map(ClusterState::metadata)
            .map(metadata -> metadata.index(index))
            .map(IndexMetadata::getSettings)
            .orElse(Settings.EMPTY);
        return MapperService.INDEX_MAPPING_DEPTH_LIMIT_SETTING.get(settings);
    }

    public static boolean isSparseVectorType(String type) {
        return SparseVectorFieldMapper.CONTENT_TYPE.equals(type);
    }

    @SuppressWarnings("unchecked")
    private static void collectSparseAnnFields(Map<String, Object> fields, String parentPath, Set<String> out, int depth, long maxDepth) {
        if (depth > maxDepth) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Field [%s] exceeds maximum mapping depth limit of [%d]", parentPath, maxDepth)
            );
        }
        for (Map.Entry<String, Object> field : fields.entrySet()) {
            if ((field.getValue() instanceof Map) == false) {
                continue;
            }
            Map<String, Object> fieldMap = (Map<String, Object>) field.getValue();
            Object type = fieldMap.get("type");
            String path = parentPath.isEmpty() ? field.getKey() : parentPath + "." + field.getKey();
            if (Objects.nonNull(type) && isSparseVectorType(type.toString())) {
                out.add(path);
            } else if (fieldMap.get("properties") instanceof Map<?, ?> nested) {
                collectSparseAnnFields((Map<String, Object>) nested, path, out, depth + 1, maxDepth);
            }
        }
    }
}
