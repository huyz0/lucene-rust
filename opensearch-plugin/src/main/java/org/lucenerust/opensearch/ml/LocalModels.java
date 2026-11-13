/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lucenerust.opensearch.NativeBridge;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.get.GetResponse;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.common.util.io.IOUtils;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.transport.client.Client;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * The models deployed on this node: each one's files under {@code <data>/ml_cache/models_cache},
 * and its native handle. Deploying reads the model's chunks back from the model index, checks the
 * zip's SHA-256 against {@code model_content_hash_value}, unpacks it, and opens the model in Rust,
 * as {@code MLModelManager.deployModel} does with DJL.
 */
public final class LocalModels {
    private static final Logger LOGGER = LogManager.getLogger(LocalModels.class);
    /** The local model functions deployed here. */
    static final Set<String> FUNCTIONS = Set.of("TEXT_EMBEDDING", "SPARSE_ENCODING", "SPARSE_TOKENIZE", "TEXT_SIMILARITY");

    /**
     * A deployed model: its handle, the model document it was deployed from, and whether it is a
     * TorchScript model.
     */
    record Deployed(long handle, String functionName, Map<String, Object> model, boolean torch) {
        Deployed(long handle, String functionName, Map<String, Object> model) {
            this(handle, functionName, model, false);
        }
    }

    private final Client client;
    private final RemoteModels remote;
    private final Path cacheDir;
    private final Path runtimeLibrary;
    private final Map<String, Deployed> models = new ConcurrentHashMap<>();
    private final Map<String, RateLimit> limits = new ConcurrentHashMap<>();
    /** The listeners of each deploy in flight here, by model id; guarded by itself. */
    private final Map<String, List<ActionListener<Void>>> deploying = new java.util.HashMap<>();
    private volatile boolean runtimeLoaded;
    private volatile boolean torchLoaded;
    /** Guards loading libtorch, apart from this object so its download holds up no ONNX deploy. */
    private final Object torchLock = new Object();

    LocalModels(Client client, RemoteModels remote, Path cacheDir, Path runtimeLibrary) {
        this.client = client;
        this.remote = remote;
        this.cacheDir = cacheDir;
        this.runtimeLibrary = runtimeLibrary;
    }

    /** The ids of the models deployed here. */
    public Set<String> deployedIds() {
        return Set.copyOf(models.keySet());
    }

    /** Whether {@code modelId} is deployed on this node. */
    public boolean isDeployed(String modelId) {
        return models.containsKey(modelId);
    }

    /** The model document {@code modelId} was deployed from, or {@code null}. */
    public Map<String, Object> model(String modelId) {
        Deployed d = models.get(modelId);
        return d == null ? null : d.model();
    }

    private void loadRuntime() {
        if (runtimeLoaded) {
            return;
        }
        synchronized (this) {
            if (runtimeLoaded == false) {
                initRuntime();
            }
        }
    }

    private void initRuntime() {
        if (NativeBridge.mlRuntimeInit(runtimeLibrary.toString(), 0) != NativeBridge.OK) {
            throw new IllegalStateException("cannot load the ONNX Runtime library: " + NativeBridge.lastError());
        }
        runtimeLoaded = true;
    }

    /**
     * Loads libtorch ({@link LibTorch}, downloading it as stock does when no copy is found) and the
     * plugin's TorchScript shim, built per libtorch C++ string ABI, beside the ONNX Runtime library.
     */
    private void loadTorch() throws IOException {
        if (torchLoaded) {
            return;
        }
        synchronized (torchLock) {
            if (torchLoaded == false) {
                initTorch();
            }
        }
    }

    private void initTorch() throws IOException {
        Path dir = LibTorch.resolve(cacheDir.getParent());
        byte[] shims = encodeTexts(
            List.of(
                runtimeLibrary.resolveSibling("liblucene_torch_cxx11.so").toString(),
                runtimeLibrary.resolveSibling("liblucene_torch_precxx11.so").toString()
            )
        );
        if (NativeBridge.mlTorchInit(dir.toString(), shims, 0) != NativeBridge.OK) {
            throw new IllegalStateException("cannot load libtorch from " + dir + ": " + NativeBridge.lastError());
        }
        torchLoaded = true;
    }

    /**
     * Deploys {@code modelId} on this node: reads its document and chunks, then opens it. The chunk
     * writes, unzip and model load run on the deploy pool, whatever thread answers the reads. A
     * deploy of a model already being deployed here waits for that one rather than racing it over
     * the same files.
     */
    public void deploy(String modelId, ActionListener<Void> listener) {
        if (models.containsKey(modelId)) {
            listener.onResponse(null);
            return;
        }
        synchronized (deploying) {
            List<ActionListener<Void>> waiting = deploying.get(modelId);
            if (waiting != null) {
                waiting.add(listener);
                return;
            }
            waiting = new ArrayList<>();
            waiting.add(listener);
            deploying.put(modelId, waiting);
        }
        ActionListener<Void> all = ActionListener.wrap(v -> finishDeploy(modelId, null), e -> finishDeploy(modelId, e));
        MlIndices.system(client, l -> client.get(new GetRequest(MlIndices.MODEL, modelId), l), onDeployPool(ActionListener.<GetResponse>wrap(r -> {
            if (r.isExists() == false) {
                throw new IllegalArgumentException("Failed to find model");
            }
            Map<String, Object> model = r.getSourceAsMap();
            String function = String.valueOf(model.get("algorithm"));
            String format = String.valueOf(model.get("model_format"));
            if ("REMOTE".equals(function)) {
                remote.open(model, ActionListener.wrap(handle -> {
                    put(modelId, new Deployed(handle, function, model));
                    all.onResponse(null);
                }, all::onFailure));
                return;
            }
            if (FUNCTIONS.contains(function) == false) {
                throw new IllegalArgumentException("lucene-rust does not deploy " + function + " models yet");
            }
            // A sparse tokenize model has no network, only its tokenizer and idf.json, whatever its format says.
            if ("SPARSE_TOKENIZE".equals(function) == false && "ONNX".equals(format) == false && "TORCH_SCRIPT".equals(format) == false) {
                throw new IllegalArgumentException("lucene-rust runs ONNX and TorchScript models; this model's format is " + format);
            }
            int chunks = ((Number) model.getOrDefault("total_chunks", 0)).intValue();
            Path dir = cacheDir.resolve("deploy").resolve(modelId);
            fetchChunks(modelId, chunks, dir, model, ActionListener.wrap(zip -> {
                open(modelId, function, model, zip, dir);
                all.onResponse(null);
            }, all::onFailure));
        }, all::onFailure)));
    }

    private void finishDeploy(String modelId, Exception failure) {
        List<ActionListener<Void>> waiting;
        synchronized (deploying) {
            waiting = deploying.remove(modelId);
        }
        if (waiting == null) {
            return;
        }
        for (ActionListener<Void> l : waiting) {
            if (failure == null) {
                l.onResponse(null);
            } else {
                l.onFailure(failure);
            }
        }
    }

    /** {@code listener}, answered on the deploy pool: its work is file I/O and model loading. */
    private <T> ActionListener<T> onDeployPool(ActionListener<T> listener) {
        return ActionListener.wrap(r -> {
            try {
                client.threadPool().executor(MlCommons.DEPLOY_THREAD_POOL).execute(() -> {
                    try {
                        listener.onResponse(r);
                    } catch (Exception e) {
                        listener.onFailure(e);
                    }
                });
            } catch (Exception rejected) {
                listener.onFailure(rejected);
            }
        }, listener::onFailure);
    }

    /** Writes chunks {@code 0..n} of {@code modelId} to one zip in {@code dir}, in order. */
    private void fetchChunks(String modelId, int n, Path dir, Map<String, Object> model, ActionListener<Path> listener) {
        try {
            IOUtils.rm(dir);
            Files.createDirectories(dir);
            Path zip = dir.resolve(modelId + ".zip");
            OutputStream out = Files.newOutputStream(zip);
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            fetchChunk(modelId, 0, n, out, sha, ActionListener.wrap(v -> {
                out.close();
                String want = (String) model.get("model_content_hash_value");
                String got = HexFormat.of().formatHex(sha.digest());
                if (want != null && want.equals(got) == false) {
                    throw new IllegalArgumentException("model content changed");
                }
                listener.onResponse(zip);
            }, e -> {
                IOUtils.closeWhileHandlingException(out);
                listener.onFailure(e);
            }));
        } catch (IOException | NoSuchAlgorithmException e) {
            listener.onFailure(e);
        }
    }

    private void fetchChunk(String modelId, int i, int n, OutputStream out, MessageDigest sha, ActionListener<Void> listener) {
        if (i == n) {
            listener.onResponse(null);
            return;
        }
        String id = modelId + "_" + i;
        MlIndices.system(client, l -> client.get(new GetRequest(MlIndices.MODEL, id), l), onDeployPool(ActionListener.<GetResponse>wrap(r -> {
            Object content = r.isExists() ? r.getSourceAsMap().get("model_content") : null;
            if (content == null) {
                throw new IllegalArgumentException("Failed to find model chunk " + id);
            }
            byte[] bytes = Base64.getDecoder().decode((String) content);
            sha.update(bytes);
            out.write(bytes);
            fetchChunk(modelId, i + 1, n, out, sha, listener);
        }, listener::onFailure)));
    }

    @SuppressWarnings("unchecked")
    private void open(String modelId, String function, Map<String, Object> model, Path zip, Path dir) throws IOException {
        Path onnx = null;
        Path torchScript = null;
        Path tokenizer = null;
        Path idf = null;
        try (ZipInputStream in = new ZipInputStream(Files.newInputStream(zip))) {
            for (ZipEntry e = in.getNextEntry(); e != null; e = in.getNextEntry()) {
                if (e.isDirectory()) {
                    continue;
                }
                String name = Path.of(e.getName()).getFileName().toString();
                Path target = dir.resolve(name);
                copy(in, target);
                if (name.endsWith(".onnx") || name.endsWith(".pt")) {
                    // DLModel.loadModel: one model file per zip.
                    if (onnx != null || torchScript != null) {
                        throw new IllegalArgumentException("found multiple models");
                    }
                    if (name.endsWith(".pt")) {
                        torchScript = target;
                    } else {
                        onnx = target;
                    }
                } else if (name.equals("tokenizer.json")) {
                    tokenizer = target;
                } else if (name.equals("idf.json")) {
                    idf = target;
                }
            }
        }
        Files.deleteIfExists(zip);
        if ("SPARSE_TOKENIZE".equals(function)) {
            if (tokenizer == null) {
                throw new IllegalArgumentException("found no tokenizer.json in the model zip");
            }
            long[] handle = new long[1];
            int rc = NativeBridge.mlSparseTokenizeOpen(Files.readAllBytes(tokenizer), idf == null ? null : Files.readAllBytes(idf), handle);
            if (rc != NativeBridge.OK) {
                throw new IllegalArgumentException(NativeBridge.lastError());
            }
            put(modelId, new Deployed(handle[0], function, model));
            return;
        }
        boolean torch = "TORCH_SCRIPT".equals(model.get("model_format"));
        if (torch ? torchScript == null : onnx == null) {
            throw new IllegalArgumentException("found no " + (torch ? ".pt" : ".onnx") + " file in the model zip");
        }
        if (tokenizer == null) {
            throw new IllegalArgumentException("found no tokenizer.json in the model zip");
        }
        if (torch) {
            openTorch(modelId, function, model, torchScript, tokenizer);
            return;
        }
        if ("SPARSE_ENCODING".equals(function) || "TEXT_SIMILARITY".equals(function)) {
            loadRuntime();
            long[] handle = new long[1];
            int rc = NativeBridge.mlOnnxModelOpen("SPARSE_ENCODING".equals(function), onnx.toString(), Files.readAllBytes(tokenizer), handle);
            if (rc != NativeBridge.OK) {
                throw new IllegalArgumentException(NativeBridge.lastError());
            }
            put(modelId, new Deployed(handle[0], function, model));
            return;
        }
        Map<String, Object> config = (Map<String, Object>) model.getOrDefault("model_config", Map.of());
        String modelType = String.valueOf(config.getOrDefault("model_type", ""));
        String pooling = String.valueOf(config.getOrDefault("pooling_mode", "MEAN"));
        boolean normalize = Boolean.TRUE.equals(config.get("normalize_result"));
        loadRuntime();
        long[] handle = new long[1];
        int rc = NativeBridge.mlTextEmbeddingOpen(onnx.toString(), Files.readAllBytes(tokenizer), modelType, pooling, normalize, handle);
        if (rc != NativeBridge.OK) {
            throw new IllegalArgumentException(NativeBridge.lastError());
        }
        put(modelId, new Deployed(handle[0], function, model));
    }

    /**
     * Opens a TorchScript model with the translator stock picks for it: a text embedding model by its
     * {@code framework_type} ({@code sentence_transformers}, else Hugging Face with its pooling mode
     * and normalization), a sparse encoding model, or a cross-encoder.
     */
    @SuppressWarnings("unchecked")
    private void openTorch(String modelId, String function, Map<String, Object> model, Path file, Path tokenizer) throws IOException {
        Map<String, Object> config = (Map<String, Object>) model.getOrDefault("model_config", Map.of());
        String pooling = String.valueOf(config.getOrDefault("pooling_mode", "MEAN"));
        boolean normalize = Boolean.TRUE.equals(config.get("normalize_result"));
        int kind = switch (function) {
            case "SPARSE_ENCODING" -> NativeBridge.TORCH_SPARSE_ENCODING;
            case "TEXT_SIMILARITY" -> NativeBridge.TORCH_TEXT_SIMILARITY;
            default -> "sentence_transformers".equalsIgnoreCase(String.valueOf(config.get("framework_type")))
                ? NativeBridge.TORCH_SENTENCE_TRANSFORMERS
                : NativeBridge.TORCH_HUGGING_FACE;
        };
        loadTorch();
        long[] handle = new long[1];
        int rc = NativeBridge.mlTorchOpen(kind, file.toString(), Files.readAllBytes(tokenizer), pooling, normalize, handle);
        if (rc != NativeBridge.OK) {
            throw new IllegalArgumentException(NativeBridge.lastError());
        }
        put(modelId, new Deployed(handle[0], function, model, true));
    }

    /** Whether {@code modelId}, deployed here, is a TorchScript model. */
    public boolean isTorchScript(String modelId) {
        Deployed d = models.get(modelId);
        return d != null && d.torch();
    }

    private void put(String modelId, Deployed deployed) {
        Deployed previous = models.put(modelId, deployed);
        if (previous != null) {
            close(previous);
        }
        LOGGER.info("lucene-rust: deployed model [{}]", modelId);
    }

    /** Takes the model document's new {@code is_enabled} and {@code rate_limiter} into the model deployed here, if any. */
    void refresh(String modelId, Map<String, Object> doc) {
        limits.remove(modelId);
        models.computeIfPresent(modelId, (id, d) -> new Deployed(d.handle(), d.functionName(), doc, d.torch()));
    }

    /** Lets a prediction through, or refuses it as stock does when the model is disabled or over its rate limit. */
    void admit(String modelId) {
        Deployed d = models.get(modelId);
        if (d == null) {
            return;
        }
        Map<String, Object> doc = d.model();
        if (Boolean.FALSE.equals(doc.get("is_enabled"))) {
            throw new OpenSearchStatusException("Model is disabled.", RestStatus.FORBIDDEN);
        }
        if (doc.get("rate_limiter") instanceof Map<?, ?> r && r.get("limit") != null && r.get("unit") != null) {
            String key = r.get("limit") + " " + r.get("unit");
            RateLimit limit = limits.get(modelId);
            if (limit == null || limit.key.equals(key) == false) {
                try {
                    limit = new RateLimit(key, Double.parseDouble(String.valueOf(r.get("limit"))), TimeUnit.valueOf(String.valueOf(r.get("unit"))).toNanos(1));
                } catch (IllegalArgumentException e) {
                    return;
                }
                limits.put(modelId, limit);
            }
            if (limit.take() == false) {
                throw new OpenSearchStatusException("Request is throttled at model level.", RestStatus.TOO_MANY_REQUESTS);
            }
        }
    }

    /** A token bucket of {@code limit} predictions per {@code unitNanos}, full at the start. */
    private static final class RateLimit {
        private final String key;
        private final double capacity;
        private final double perNano;
        private double tokens;
        private long last = System.nanoTime();

        RateLimit(String key, double limit, long unitNanos) {
            this.key = key;
            this.capacity = limit;
            this.perNano = limit / unitNanos;
            this.tokens = limit;
        }

        synchronized boolean take() {
            long now = System.nanoTime();
            tokens = Math.min(capacity, tokens + (now - last) * perNano);
            last = now;
            if (tokens < 1) {
                return false;
            }
            tokens -= 1;
            return true;
        }
    }

    /** The function name ({@code algorithm}) of {@code modelId} as deployed here, or {@code null}. */
    public String function(String modelId) {
        Deployed d = models.get(modelId);
        return d == null ? null : d.functionName();
    }

    private static void copy(InputStream in, Path target) throws IOException {
        Path tmp = target.resolveSibling(target.getFileName() + ".tmp");
        Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    /** Undeploys {@code modelId} from this node; {@code false} if it was not deployed here. */
    public boolean undeploy(String modelId) {
        Deployed d = models.remove(modelId);
        if (d == null) {
            return false;
        }
        close(d);
        try {
            IOUtils.rm(cacheDir.resolve("deploy").resolve(modelId));
        } catch (IOException e) {
            LOGGER.warn("lucene-rust: cannot remove the files of model [{}]", modelId, e);
        }
        return true;
    }

    private void close(Deployed d) {
        NativeBridge.mlModelClose(d.handle());
        if ("REMOTE".equals(d.functionName())) {
            remote.closed(d.handle());
        }
    }

    /** The native handle of {@code modelId}, deployed here. */
    long handle(String modelId) {
        return deployed(modelId).handle();
    }

    /** Undeploys every model, for node shutdown. */
    public void closeAll() {
        for (String id : List.copyOf(models.keySet())) {
            undeploy(id);
        }
    }

    /** One token of a sparse vector: its id, its decoded token and its weight. */
    public record TokenWeight(int id, String token, float weight) {}

    /** The sparse vectors of {@code texts} from sparse tokenize model {@code modelId}, deployed on this node. */
    public List<List<TokenWeight>> sparseTokenize(String modelId, List<String> texts) {
        Deployed d = deployed(modelId);
        ByteBuffer r = result(NativeBridge.mlSparseTokenize(d.handle(), encodeTexts(texts)));
        List<List<TokenWeight>> result = new ArrayList<>(texts.size());
        while (r.hasRemaining()) {
            result.add(readWeights(r));
        }
        return result;
    }

    /** One model output's sparse vector: the output's name in the graph and its non-zero weights. */
    public record SparseOutput(String name, List<TokenWeight> weights) {}

    /**
     * Per text, each output's sparse vector from sparse encoding model {@code modelId}, deployed on
     * this node; keyed by token id ({@code byId}, the tokens are then empty) or by token.
     */
    public List<List<SparseOutput>> sparseEncode(String modelId, List<String> texts, boolean byId) {
        Deployed d = deployed(modelId);
        ByteBuffer r = result(NativeBridge.mlSparseEncode(d.handle(), encodeTexts(texts), byId));
        List<List<SparseOutput>> result = new ArrayList<>(texts.size());
        while (r.hasRemaining()) {
            int n = r.getInt();
            List<SparseOutput> outputs = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                byte[] name = new byte[r.getInt()];
                r.get(name);
                outputs.add(new SparseOutput(new String(name, StandardCharsets.UTF_8), readWeights(r)));
            }
            result.add(outputs);
        }
        return result;
    }

    /** Per document of {@code texts}, the output tensors of cross-encoder {@code modelId} for it against {@code query}. */
    public List<List<Tensor>> similarity(String modelId, String query, List<String> texts) {
        Deployed d = deployed(modelId);
        ByteBuffer r = result(NativeBridge.mlTextSimilarity(d.handle(), query, encodeTexts(texts)));
        List<List<Tensor>> result = new ArrayList<>(texts.size());
        while (r.hasRemaining()) {
            int n = r.getInt();
            List<Tensor> outputs = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                outputs.add(readTensor(r));
            }
            result.add(outputs);
        }
        return result;
    }

    private Deployed deployed(String modelId) {
        Deployed d = models.get(modelId);
        if (d == null) {
            throw new IllegalArgumentException("Model not ready yet. Please deploy the model first.");
        }
        return d;
    }

    /** A model's own prediction failed; the cause says why. */
    static final class InferenceException extends RuntimeException {
        InferenceException(Throwable cause) {
            super(cause.getMessage(), cause);
        }
    }

    /** {@code out} to read; {@code null} is the native call's failure. */
    private static ByteBuffer result(byte[] out) {
        if (out == null) {
            throw new InferenceException(new IllegalStateException(NativeBridge.lastError()));
        }
        return ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
    }

    private static List<TokenWeight> readWeights(ByteBuffer r) {
        int n = r.getInt();
        List<TokenWeight> weights = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int id = r.getInt();
            byte[] token = new byte[r.getInt()];
            r.get(token);
            weights.add(new TokenWeight(id, new String(token, StandardCharsets.UTF_8), r.getFloat()));
        }
        return weights;
    }

    private static Tensor readTensor(ByteBuffer r) {
        int rank = r.getInt();
        long[] shape = new long[rank];
        long n = 1;
        for (int i = 0; i < rank; i++) {
            shape[i] = Integer.toUnsignedLong(r.getInt());
            n *= shape[i];
        }
        float[] data = new float[Math.toIntExact(n)];
        r.asFloatBuffer().get(data);
        r.position(r.position() + data.length * Float.BYTES);
        return new Tensor(shape, data);
    }

    private static byte[] encodeTexts(List<String> texts) {
        int size = 0;
        List<byte[]> utf8 = new ArrayList<>(texts.size());
        for (String t : texts) {
            byte[] b = t.getBytes(StandardCharsets.UTF_8);
            utf8.add(b);
            size += 4 + b.length;
        }
        ByteBuffer in = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
        for (byte[] b : utf8) {
            in.putInt(b.length).put(b);
        }
        return in.array();
    }

    /** A text embedding: {@code shape} and {@code data}, as ml-commons' {@code ModelTensor} carries it. */
    public record Tensor(long[] shape, float[] data) {}

    /**
     * One output of a TorchScript text embedding model, as its translator gives it: its name ({@code
     * null} for an unnamed one), ml-commons data type, shape, values, and whether it carries its bytes
     * ({@code byte_buffer}).
     */
    public record TypedOutput(String name, String dataType, long[] shape, List<Number> data, byte[] bytes) {}

    /**
     * Per text, the outputs of TorchScript text embedding model {@code modelId}, deployed here. A dict
     * the model returns comes back in the iteration order of the {@code ConcurrentHashMap} DJL reads
     * it into, as stock lists it. A sentence-transformers model's outputs carry their little-endian
     * bytes; the one pooled output of a Hugging Face model does not.
     */
    @SuppressWarnings("unchecked")
    public List<List<TypedOutput>> embedTorch(String modelId, List<String> texts) {
        Deployed d = deployed(modelId);
        Map<String, Object> config = (Map<String, Object>) d.model().getOrDefault("model_config", Map.of());
        boolean bytes = "sentence_transformers".equalsIgnoreCase(String.valueOf(config.get("framework_type")));
        ByteBuffer r = result(NativeBridge.mlTorchEmbed(d.handle(), encodeTexts(texts)));
        List<List<TypedOutput>> result = new ArrayList<>(texts.size());
        while (r.hasRemaining()) {
            int n = r.getInt();
            List<TypedOutput> outputs = new ArrayList<>(n);
            Map<String, TypedOutput> dict = null;
            int group = 0;
            for (int i = 0; i < n; i++) {
                byte[] name = new byte[r.getInt()];
                r.get(name);
                int g = r.getInt();
                // A tensor outside a dict has no name; a dict key is one, even when empty.
                String key = g == 0 && name.length == 0 ? null : new String(name, StandardCharsets.UTF_8);
                TypedOutput o = readTyped(r, key, bytes);
                if (g != group && dict != null) {
                    outputs.addAll(dict.values());
                    dict = null;
                }
                group = g;
                if (g == 0) {
                    outputs.add(o);
                } else {
                    if (dict == null) {
                        dict = new ConcurrentHashMap<>();
                    }
                    dict.put(o.name(), o);
                }
            }
            if (dict != null) {
                outputs.addAll(dict.values());
            }
            result.add(outputs);
        }
        return result;
    }

    private static TypedOutput readTyped(ByteBuffer r, String name, boolean withBytes) {
        boolean float32 = r.get() == 0;
        int rank = r.getInt();
        long[] shape = new long[rank];
        long n = 1;
        for (int i = 0; i < rank; i++) {
            shape[i] = Integer.toUnsignedLong(r.getInt());
            n *= shape[i];
        }
        int count = Math.toIntExact(n);
        int width = float32 ? Float.BYTES : Long.BYTES;
        byte[] raw = new byte[Math.multiplyExact(count, width)];
        r.get(raw);
        ByteBuffer values = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        List<Number> data = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            data.add(float32 ? (Number) values.getFloat() : (Number) values.getLong());
        }
        return new TypedOutput(name, float32 ? "FLOAT32" : "INT64", shape, data, withBytes ? raw : null);
    }

    /** The embeddings of {@code texts} from model {@code modelId}, deployed on this node. */
    public List<Tensor> embed(String modelId, List<String> texts) {
        Deployed d = deployed(modelId);
        ByteBuffer r = result(NativeBridge.mlTextEmbeddingEmbed(d.handle(), encodeTexts(texts)));
        List<Tensor> tensors = new ArrayList<>(texts.size());
        while (r.hasRemaining()) {
            tensors.add(readTensor(r));
        }
        return tensors;
    }
}
