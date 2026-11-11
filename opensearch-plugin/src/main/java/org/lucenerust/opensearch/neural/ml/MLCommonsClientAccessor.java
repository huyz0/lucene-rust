/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.ml;

import static org.lucenerust.opensearch.neural.processor.TextImageEmbeddingProcessor.INPUT_IMAGE;
import static org.lucenerust.opensearch.neural.processor.TextImageEmbeddingProcessor.INPUT_TEXT;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

import org.lucenerust.opensearch.ml.MlCommons;
import org.lucenerust.opensearch.neural.processor.EmbeddingContentType;
import org.lucenerust.opensearch.neural.processor.InferenceRequest;
import org.lucenerust.opensearch.neural.processor.MapInferenceRequest;
import org.lucenerust.opensearch.neural.processor.SimilarityInferenceRequest;
import org.lucenerust.opensearch.neural.processor.TextInferenceRequest;
import org.lucenerust.opensearch.neural.util.RetryUtil;
import org.opensearch.core.action.ActionListener;

import org.opensearch.common.cache.Cache;
import org.opensearch.common.cache.CacheBuilder;

/**
 * neural-search's {@code MLCommonsClientAccessor}: the predict and get-model calls neural search
 * makes, over this plugin's ml-commons. Requests and results have the shapes of ml-commons' REST
 * predict API ({@code text_docs}, {@code parameters}, {@code target_response}; {@code
 * inference_results[].output[]}).
 */
public class MLCommonsClientAccessor {
    private static final String RESPONSE_FIELD = "response";

    private final Supplier<MlCommons> ml;
    private final Cache<String, MLModel> modelCache = CacheBuilder.<String, MLModel>builder().setMaximumWeight(1000).build();

    public MLCommonsClientAccessor(Supplier<MlCommons> ml) {
        this.ml = ml;
    }

    /** Dense embeddings of the request's texts, in their order. */
    public void inferenceSentences(final TextInferenceRequest inferenceRequest, final ActionListener<List<List<Number>>> listener) {
        checkModelAndThenPredict(
            inferenceRequest.getModelId(),
            listener::onFailure,
            model -> retryableInference(
                inferenceRequest,
                0,
                () -> textEmbeddingInput(model, inferenceRequest.getTargetResponseFilters(), inferenceRequest.getInputTexts(), inferenceRequest),
                this::buildVectorFromResponse,
                listener
            )
        );
    }

    /** Sparse vectors (token to weight) of the request's texts, in their order. */
    public void inferenceSentencesWithMapResult(final TextInferenceRequest inferenceRequest, final ActionListener<List<Map<String, ?>>> listener) {
        checkModelAndThenPredict(
            inferenceRequest.getModelId(),
            listener::onFailure,
            model -> retryableInference(
                inferenceRequest,
                0,
                () -> textEmbeddingInput(model, null, inferenceRequest.getInputTexts(), inferenceRequest),
                this::buildMapResultFromResponse,
                listener
            )
        );
    }

    /** One embedding of a text and an image (the first output). */
    public void inferenceSentencesMap(MapInferenceRequest inferenceRequest, final ActionListener<List<Number>> listener) {
        checkModelAndThenPredict(inferenceRequest.getModelId(), listener::onFailure, model -> {
            List<String> inputText = new ArrayList<>();
            inputText.add(inferenceRequest.getInputObjects().get(INPUT_TEXT));
            if (inferenceRequest.getInputObjects().containsKey(INPUT_IMAGE)) {
                inputText.add(inferenceRequest.getInputObjects().get(INPUT_IMAGE));
            }
            retryableInference(inferenceRequest, 0, () -> {
                Map<String, Object> body = docsInput(inferenceRequest.getTargetResponseFilters(), inputText);
                if (AsymmetricModelDetector.isAsymmetricModel(model) && inferenceRequest.getEmbeddingContentType() != null) {
                    body.put("parameters", contentTypeParameters(inferenceRequest.getEmbeddingContentType()));
                }
                return new Input("text_embedding", body);
            }, this::buildSingleVectorFromResponse, listener);
        });
    }

    /** Cross-encoder scores of the request's texts against its query text, in their order. */
    public void inferenceSimilarity(SimilarityInferenceRequest inferenceRequest, final ActionListener<List<Float>> listener) {
        retryableInference(inferenceRequest, 0, () -> {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("query_text", inferenceRequest.getQueryText());
            body.put("text_docs", inferenceRequest.getInputTexts());
            return new Input("text_similarity", body);
        }, output -> {
            List<Float> scores = new ArrayList<>();
            for (List<Number> v : buildVectorFromResponse(output)) {
                scores.add(v.getFirst().floatValue());
            }
            return scores;
        }, listener);
    }

    public void getModel(final String modelId, final ActionListener<MLModel> listener) {
        retryableGetModel(modelId, 0, listener);
    }

    /**
     * The models of {@code modelIds}, fetched in parallel; fails when any one fails, with every
     * failure's message.
     */
    public void getModels(final Set<String> modelIds, final Consumer<Map<String, MLModel>> onSuccess, final Consumer<Exception> onFailure) {
        if (modelIds.isEmpty()) {
            try {
                onSuccess.accept(Collections.emptyMap());
            } catch (Exception e) {
                onFailure.accept(e);
            }
            return;
        }
        final Map<String, MLModel> modelMap = new ConcurrentHashMap<>();
        final AtomicInteger counter = new AtomicInteger(modelIds.size());
        final AtomicBoolean hasError = new AtomicBoolean(false);
        final List<String> errors = Collections.synchronizedList(new ArrayList<>());
        for (String modelId : modelIds) {
            try {
                getModel(modelId, ActionListener.wrap(model -> {
                    modelMap.put(modelId, model);
                    if (counter.decrementAndGet() == 0) {
                        if (hasError.get()) {
                            onFailure.accept(new RuntimeException(String.join(";", errors)));
                        } else {
                            try {
                                onSuccess.accept(modelMap);
                            } catch (Exception e) {
                                onFailure.accept(e);
                            }
                        }
                    }
                }, e -> handleGetModelException(hasError, errors, modelId, e, counter, onFailure)));
            } catch (Exception e) {
                handleGetModelException(hasError, errors, modelId, e, counter, onFailure);
            }
        }
    }

    public void getCachedModel(String modelId, ActionListener<MLModel> listener) {
        MLModel cached = modelCache.get(modelId);
        if (cached != null) {
            listener.onResponse(cached);
            return;
        }
        fetchModel(modelId, ActionListener.wrap(model -> {
            modelCache.put(modelId, model);
            listener.onResponse(model);
        }, listener::onFailure));
    }

    private void handleGetModelException(
        AtomicBoolean hasError,
        List<String> errors,
        String modelId,
        Exception e,
        AtomicInteger counter,
        Consumer<Exception> onFailure
    ) {
        hasError.set(true);
        errors.add("Failed to fetch model [" + modelId + "]: " + e.getMessage());
        if (counter.decrementAndGet() == 0) {
            onFailure.accept(new RuntimeException(String.join(";", errors)));
        }
    }

    private void retryableGetModel(final String modelId, final int retryTime, final ActionListener<MLModel> listener) {
        fetchModel(
            modelId,
            ActionListener.wrap(
                listener::onResponse,
                e -> RetryUtil.handleRetryOrFailure(e, retryTime, () -> retryableGetModel(modelId, retryTime + 1, listener), listener)
            )
        );
    }

    private void fetchModel(String modelId, ActionListener<MLModel> listener) {
        try {
            ml.get().getModel(modelId, ActionListener.wrap(source -> listener.onResponse(new MLModel(source)), listener::onFailure));
        } catch (Exception e) {
            listener.onFailure(e);
        }
    }

    /** A predict call: the algorithm route and the REST body. */
    private record Input(String algorithm, Map<String, Object> body) {}

    private <T> void retryableInference(
        final InferenceRequest inferenceRequest,
        final int retryTime,
        final Supplier<Input> inputSupplier,
        final Function<Map<String, Object>, T> outputBuilder,
        final ActionListener<T> listener
    ) {
        Input input;
        try {
            input = inputSupplier.get();
        } catch (Exception e) {
            listener.onFailure(e);
            return;
        }
        ActionListener<Map<String, Object>> onOutput = ActionListener.wrap(
            output -> listener.onResponse(outputBuilder.apply(output)),
            e -> RetryUtil.handleRetryOrFailure(
                e,
                retryTime,
                () -> retryableInference(inferenceRequest, retryTime + 1, inputSupplier, outputBuilder, listener),
                listener
            )
        );
        try {
            ml.get().predict(input.algorithm(), inferenceRequest.getModelId(), input.body(), onOutput);
        } catch (Exception e) {
            onOutput.onFailure(e);
        }
    }

    /** {@code NeuralSearchMLInputBuilder.createTextEmbeddingInput}. */
    private static Input textEmbeddingInput(MLModel model, List<String> targetResponseFilters, List<String> inputText, InferenceRequest request) {
        boolean isAsymmetric = AsymmetricModelDetector.isAsymmetricModel(model);
        if (isAsymmetric && model.isRemote()) {
            Map<String, Object> parameters = new LinkedHashMap<>();
            parameters.put(AsymmetricTextEmbeddingConstants.TEXTS_KEY, inputText);
            parameters.put(
                AsymmetricTextEmbeddingConstants.CONTENT_TYPE_KEY,
                request.getEmbeddingContentType().toString().toLowerCase(java.util.Locale.ROOT)
            );
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("parameters", parameters);
            return new Input("remote", body);
        }
        Map<String, Object> body = docsInput(targetResponseFilters, inputText);
        MLAlgoParams params = algoParams(isAsymmetric, request);
        if (params != null) {
            Map<String, Object> p = params.toParameters();
            if (p.isEmpty() == false) {
                body.put("parameters", p);
            }
        }
        return new Input("text_embedding", body);
    }

    private static Map<String, Object> docsInput(List<String> targetResponseFilters, List<String> inputText) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("text_docs", inputText);
        body.put("return_number", true);
        if (targetResponseFilters != null) {
            body.put("target_response", targetResponseFilters);
        }
        return body;
    }

    /** {@code NeuralSearchMLInputBuilder.createMLAlgoParams}. */
    private static MLAlgoParams algoParams(boolean isAsymmetric, InferenceRequest request) {
        if (isAsymmetric == false) {
            return request.getMlAlgoParams();
        }
        EmbeddingContentType contentType = request.getEmbeddingContentType();
        if (contentType == null) {
            throw new IllegalArgumentException("embeddingContentType must be set for asymmetric local models");
        }
        MLAlgoParams preset = request.getMlAlgoParams();
        if (preset != null && (preset instanceof AsymmetricTextEmbeddingParameters) == false) {
            throw new IllegalArgumentException("MLAlgoParams must be AsymmetricTextEmbeddingParameters for asymmetric models");
        }
        return (preset != null ? ((AsymmetricTextEmbeddingParameters) preset).toBuilder() : AsymmetricTextEmbeddingParameters.builder())
            .embeddingContentType(contentType.toMLContentType())
            .build();
    }

    private static Map<String, Object> contentTypeParameters(EmbeddingContentType contentType) {
        return AsymmetricTextEmbeddingParameters.builder().embeddingContentType(contentType.toMLContentType()).build().toParameters();
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> tensors(Map<String, Object> output) {
        List<Map<String, Object>> tensors = new ArrayList<>();
        Object results = output.get("inference_results");
        if (results instanceof List<?> list) {
            for (Object r : list) {
                if (r instanceof Map<?, ?> m && m.get("output") instanceof List<?> outs) {
                    for (Object t : outs) {
                        tensors.add((Map<String, Object>) t);
                    }
                }
            }
        }
        return tensors;
    }

    @SuppressWarnings("unchecked")
    private List<List<Number>> buildVectorFromResponse(Map<String, Object> output) {
        final List<List<Number>> vector = new ArrayList<>();
        for (Map<String, Object> tensor : tensors(output)) {
            Object data = tensor.get("data");
            if (data instanceof List<?> values) {
                vector.add(new ArrayList<>((List<Number>) values));
            } else {
                vector.addAll(extractVectorsFromRemoteEmbeddingResponse(tensor));
            }
        }
        return vector;
    }

    /** Embeddings a remote model's post-process function put under {@code dataAsMap.response}. */
    @SuppressWarnings("unchecked")
    private static List<List<Number>> extractVectorsFromRemoteEmbeddingResponse(Map<String, Object> tensor) {
        String format = ". Ensure the connector's post_process_function returns: {\"response\": [[float, float, ...], ...]}";
        Object dataMapObject = tensor.get("dataAsMap");
        if ((dataMapObject instanceof Map<?, ?>) == false) {
            throw new IllegalStateException(
                "The remote model returned no data. Ensure the connector's post_process_function is configured to return: "
                    + "{\"response\": [[float, float, ...], ...]}"
            );
        }
        Map<String, ?> dataMap = (Map<String, ?>) dataMapObject;
        if (dataMap.containsKey(RESPONSE_FIELD) == false) {
            throw new IllegalStateException(
                "The remote model response is missing the required 'response' key. Ensure the connector's post_process_function returns: "
                    + "{\"response\": [[float, float, ...], ...]}"
            );
        }
        Object responseData = dataMap.get(RESPONSE_FIELD);
        if ((responseData instanceof List) == false) {
            throw new IllegalStateException(
                "The remote model 'response' field must be a list of float arrays, but got: "
                    + (responseData == null ? "null" : responseData.getClass().getSimpleName())
                    + format
            );
        }
        final List<List<Number>> vectors = new ArrayList<>();
        for (Object embedding : (List<?>) responseData) {
            if ((embedding instanceof List) == false) {
                throw new IllegalStateException(
                    "Each element in the remote model 'response' must be a list of floats, but got: "
                        + (embedding == null ? "null" : embedding.getClass().getSimpleName())
                        + format
                );
            }
            List<?> embeddingList = (List<?>) embedding;
            if (embeddingList.isEmpty() == false && (embeddingList.get(0) instanceof Number) == false) {
                throw new IllegalStateException(
                    "Each value in a remote model embedding must be a number (float), but got: "
                        + embeddingList.get(0).getClass().getSimpleName()
                        + format
                );
            }
            vectors.add(new ArrayList<>((List<Number>) embeddingList));
        }
        return vectors;
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, ?>> buildMapResultFromResponse(Map<String, Object> output) {
        List<Map<String, Object>> tensors = tensors(output);
        if (tensors.isEmpty()) {
            throw new IllegalStateException(
                "Empty model result produced. Expected at least [1] tensor output and [1] model tensor, but got [0]"
            );
        }
        List<Map<String, ?>> resultMaps = new ArrayList<>();
        for (Map<String, Object> tensor : tensors) {
            resultMaps.add((Map<String, ?>) tensor.get("dataAsMap"));
        }
        return resultMaps;
    }

    private List<Number> buildSingleVectorFromResponse(final Map<String, Object> output) {
        final List<List<Number>> vector = buildVectorFromResponse(output);
        return vector.isEmpty() ? new ArrayList<>() : vector.get(0);
    }

    private void checkModelAndThenPredict(String modelId, Consumer<Exception> onFailure, Consumer<MLModel> runPrediction) {
        MLModel cached = modelCache.get(modelId);
        if (cached != null) {
            runPrediction.accept(cached);
            return;
        }
        fetchModel(modelId, ActionListener.wrap(model -> {
            modelCache.put(modelId, model);
            runPrediction.accept(model);
        }, onFailure));
    }
}
