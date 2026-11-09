/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lucenerust.opensearch.NativeBridge;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.get.GetResponse;
import org.opensearch.common.util.io.IOUtils;
import org.opensearch.core.action.ActionListener;
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

    /** A deployed model: its handle and the model document it was deployed from. */
    record Deployed(long handle, String functionName, Map<String, Object> model) {}

    private final Client client;
    private final Path cacheDir;
    private final Path runtimeLibrary;
    private final Map<String, Deployed> models = new ConcurrentHashMap<>();
    private volatile boolean runtimeLoaded;

    LocalModels(Client client, Path cacheDir, Path runtimeLibrary) {
        this.client = client;
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

    private synchronized void loadRuntime() {
        if (runtimeLoaded) {
            return;
        }
        if (NativeBridge.mlRuntimeInit(runtimeLibrary.toString(), 0) != NativeBridge.OK) {
            throw new IllegalStateException("cannot load the ONNX Runtime library: " + NativeBridge.lastError());
        }
        runtimeLoaded = true;
    }

    /**
     * Deploys {@code modelId} on this node: reads its document and chunks, then opens it. Runs the
     * blocking part on the caller's thread, which must be a pool thread.
     */
    public void deploy(String modelId, ActionListener<Void> listener) {
        if (models.containsKey(modelId)) {
            listener.onResponse(null);
            return;
        }
        MlIndices.system(client, l -> client.get(new GetRequest(MlIndices.MODEL, modelId), l), ActionListener.<GetResponse>wrap(r -> {
            if (r.isExists() == false) {
                throw new IllegalArgumentException("Failed to find model");
            }
            Map<String, Object> model = r.getSourceAsMap();
            String function = String.valueOf(model.get("algorithm"));
            String format = String.valueOf(model.get("model_format"));
            if ("TEXT_EMBEDDING".equals(function) == false) {
                throw new IllegalArgumentException("lucene-rust does not deploy " + function + " models yet");
            }
            if ("ONNX".equals(format) == false) {
                throw new IllegalArgumentException("lucene-rust runs ONNX models; this model's format is " + format);
            }
            int chunks = ((Number) model.getOrDefault("total_chunks", 0)).intValue();
            Path dir = cacheDir.resolve("deploy").resolve(modelId);
            fetchChunks(modelId, chunks, dir, model, ActionListener.wrap(zip -> {
                open(modelId, function, model, zip, dir);
                listener.onResponse(null);
            }, listener::onFailure));
        }, listener::onFailure));
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
        MlIndices.system(client, l -> client.get(new GetRequest(MlIndices.MODEL, id), l), ActionListener.<GetResponse>wrap(r -> {
            Object content = r.isExists() ? r.getSourceAsMap().get("model_content") : null;
            if (content == null) {
                throw new IllegalArgumentException("Failed to find model chunk " + id);
            }
            byte[] bytes = Base64.getDecoder().decode((String) content);
            sha.update(bytes);
            out.write(bytes);
            fetchChunk(modelId, i + 1, n, out, sha, listener);
        }, listener::onFailure));
    }

    @SuppressWarnings("unchecked")
    private void open(String modelId, String function, Map<String, Object> model, Path zip, Path dir) throws IOException {
        Path onnx = null;
        Path tokenizer = null;
        try (ZipInputStream in = new ZipInputStream(Files.newInputStream(zip))) {
            for (ZipEntry e = in.getNextEntry(); e != null; e = in.getNextEntry()) {
                if (e.isDirectory()) {
                    continue;
                }
                String name = Path.of(e.getName()).getFileName().toString();
                Path target = dir.resolve(name);
                copy(in, target);
                if (name.endsWith(".onnx")) {
                    onnx = target;
                } else if (name.equals("tokenizer.json")) {
                    tokenizer = target;
                }
            }
        }
        Files.deleteIfExists(zip);
        if (onnx == null) {
            throw new IllegalArgumentException("found no .onnx file in the model zip");
        }
        if (tokenizer == null) {
            throw new IllegalArgumentException("found no tokenizer.json in the model zip");
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
        Deployed previous = models.put(modelId, new Deployed(handle[0], function, model));
        if (previous != null) {
            NativeBridge.mlModelClose(previous.handle());
        }
        LOGGER.info("lucene-rust: deployed model [{}]", modelId);
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
        NativeBridge.mlModelClose(d.handle());
        try {
            IOUtils.rm(cacheDir.resolve("deploy").resolve(modelId));
        } catch (IOException e) {
            LOGGER.warn("lucene-rust: cannot remove the files of model [{}]", modelId, e);
        }
        return true;
    }

    /** Undeploys every model, for node shutdown. */
    public void closeAll() {
        for (String id : List.copyOf(models.keySet())) {
            undeploy(id);
        }
    }

    /** A text embedding: {@code shape} and {@code data}, as ml-commons' {@code ModelTensor} carries it. */
    public record Tensor(long[] shape, float[] data) {}

    /** The embeddings of {@code texts} from model {@code modelId}, deployed on this node. */
    public List<Tensor> embed(String modelId, List<String> texts) {
        Deployed d = models.get(modelId);
        if (d == null) {
            throw new IllegalArgumentException("Model not ready yet. Please deploy the model first.");
        }
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
        byte[] out = NativeBridge.mlTextEmbeddingEmbed(d.handle(), in.array());
        if (out == null) {
            throw new IllegalStateException("Failed to inference " + d.functionName() + " model: " + modelId);
        }
        ByteBuffer r = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
        List<Tensor> tensors = new ArrayList<>(texts.size());
        while (r.hasRemaining()) {
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
            tensors.add(new Tensor(shape, data));
        }
        return tensors;
    }
}
