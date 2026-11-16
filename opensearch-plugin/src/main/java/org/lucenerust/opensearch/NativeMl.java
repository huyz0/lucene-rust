/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.lucenerust.opensearch.NativeBridge.*;
import static org.lucenerust.opensearch.NativeCall.*;
import static org.lucenerust.opensearch.NativeSearch.*;
import static org.lucenerust.opensearch.NativeStore.*;
import static org.lucenerust.opensearch.NativeAnalysis.*;
import static org.lucenerust.opensearch.NativeSparse.*;
import static org.lucenerust.opensearch.NativeZstd.*;
import static org.lucenerust.opensearch.NativeLimits.*;
import static org.lucenerust.opensearch.NativeWriter.*;
import static org.lucenerust.opensearch.NativeLive.*;

/** The ml-commons local-model downcalls: ONNX, TorchScript, remote connectors and credentials; see {@link NativeBridge}. */
public final class NativeMl {
    private NativeMl() {}

    private static final class H {
        static final MethodHandle ML_RUNTIME_INIT = bind("ffi_ml_runtime_init", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT));
        static final MethodHandle ML_TEXT_EMBEDDING_OPEN = bind(
            "ffi_ml_text_embedding_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_BYTE, ADDRESS)
        );
        static final MethodHandle ML_TEXT_EMBEDDING_EMBED = bind(
            "ffi_ml_text_embedding_embed_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_SPARSE_TOKENIZE_OPEN = bind(
            "ffi_ml_sparse_tokenize_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle ML_SPARSE_TOKENIZE = bind(
            "ffi_ml_sparse_tokenize_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_HF_ANALYZER_OPEN = bind(
            "ffi_ml_hf_analyzer_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle ML_HF_ANALYZE = bind(
            "ffi_ml_hf_analyze_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_SPARSE_ENCODING_OPEN = bind(
            "ffi_ml_sparse_encoding_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle ML_SPARSE_ENCODING = bind(
            "ffi_ml_sparse_encoding_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_BYTE, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_TEXT_SIMILARITY_OPEN = bind(
            "ffi_ml_text_similarity_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle ML_TEXT_SIMILARITY = bind(
            "ffi_ml_text_similarity_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_HIGHLIGHTER_OPEN = bind(
            "ffi_ml_highlighter_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_BYTE, ADDRESS)
        );
        static final MethodHandle ML_HIGHLIGHT = bind(
            "ffi_ml_highlight_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_MODEL_CLOSE = bind("ffi_ml_model_close", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        static final MethodHandle ML_MASTER_KEY = bind("ffi_ml_master_key_alloc", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        static final MethodHandle ML_CREDENTIAL_ENCRYPT = bind(
            "ffi_ml_credential_encrypt_alloc",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_CREDENTIAL_DECRYPT = bind(
            "ffi_ml_credential_decrypt_alloc",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_REMOTE_OPEN = bind(
            "ffi_ml_remote_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle ML_REMOTE_CALL = bind(
            "ffi_ml_remote_call_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_TORCH_INIT = bind(
            "ffi_ml_torch_init",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT)
        );
        static final MethodHandle ML_TORCH_OPEN = bind(
            "ffi_ml_torch_open",
            FunctionDescriptor.of(JAVA_INT, JAVA_BYTE, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_BYTE, ADDRESS)
        );
        static final MethodHandle ML_TORCH_TEXT_EMBEDDING = bind(
            "ffi_ml_torch_text_embedding_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
    }

    // ---- ml-commons local models (ml.rs). ----

    /**
     * Loads the ONNX Runtime library at {@code path}, with {@code threads} intra-op threads shared by
     * every model (0: one per core). Later calls keep the first library.
     */
    public static int mlRuntimeInit(String path, int threads) {
        byte[] p = path.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            return (int) H.ML_RUNTIME_INIT.invokeExact(s.bytes(p), (long) p.length, threads);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Opens the ONNX text-embedding model at {@code modelPath} with the {@code tokenizer.json} bytes,
     * ml-commons' {@code model_type} and {@code pooling_mode}, and whether results are normalized;
     * its handle goes to {@code outHandle[0]}.
     */
    public static int mlTextEmbeddingOpen(
        String modelPath,
        byte[] tokenizer,
        String modelType,
        String pooling,
        boolean normalize,
        long[] outHandle
    ) {
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        byte[] model = modelPath.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] type = modelType.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] mode = pooling.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            int rc = (int) H.ML_TEXT_EMBEDDING_OPEN.invokeExact(
                s.bytes(model),
                (long) model.length,
                s.bytes(tokenizer),
                (long) tokenizer.length,
                s.bytes(type),
                (long) type.length,
                s.bytes(mode),
                (long) mode.length,
                (byte) (normalize ? 1 : 0),
                handle
            );
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Embeds {@code texts} (each a little-endian {@code u32} byte length then UTF-8) with model {@code
     * handle}: per text a {@code u32} rank, the dimensions, then the {@code f32} values, all little
     * endian. {@code null} on failure, with the reason in {@link #lastError}.
     */
    public static byte[] mlTextEmbeddingEmbed(long handle, byte[] texts) {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_TEXT_EMBEDDING_EMBED.invokeExact(handle, s.bytes(texts), (long) texts.length, ptrOf(slot), lenOf(slot));
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Opens a sparse tokenize model from its {@code tokenizer.json} bytes and its {@code idf.json}
     * bytes ({@code null} when it has none); its handle goes to {@code outHandle[0]}.
     */
    public static int mlSparseTokenizeOpen(byte[] tokenizer, byte[] idf, long[] outHandle) {
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            MemorySegment idfBytes = idf == null ? MemorySegment.NULL : s.bytes(idf);
            long idfLength = idf == null ? 0L : idf.length;
            int rc = (int) H.ML_SPARSE_TOKENIZE_OPEN.invokeExact(s.bytes(tokenizer), (long) tokenizer.length, idfBytes, idfLength, handle);
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The token weights of {@code texts} (encoded as for {@link #mlTextEmbeddingEmbed}) from sparse
     * tokenize model {@code handle}: per text a {@code u32} count, then per token its {@code u32} id,
     * {@code u32} byte length, UTF-8 token and {@code f32} weight, all little endian. {@code null} on
     * failure, with the reason in {@link #lastError}.
     */
    public static byte[] mlSparseTokenize(long handle, byte[] texts) {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_SPARSE_TOKENIZE.invokeExact(handle, s.bytes(texts), (long) texts.length, ptrOf(slot), lenOf(slot));
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Opens a {@code bert-uncased}-style analyzer from its {@code tokenizer.json} bytes and its
     * {@code idf.json} bytes ({@code null} when it has none); its handle goes to {@code outHandle[0]}.
     */
    public static int mlHfAnalyzerOpen(byte[] tokenizer, byte[] idf, long[] outHandle) {
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            MemorySegment idfBytes = idf == null ? MemorySegment.NULL : s.bytes(idf);
            long idfLength = idf == null ? 0L : idf.length;
            int rc = (int) H.ML_HF_ANALYZER_OPEN.invokeExact(s.bytes(tokenizer), (long) tokenizer.length, idfBytes, idfLength, handle);
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The token stream of the UTF-8 {@code text} from analyzer {@code handle}: a {@code u32} count,
     * then per token its {@code u32} id, {@code u32} start and end offsets in code points,
     * {@code u32} byte length, UTF-8 token and {@code f32} weight, all little endian. {@code null}
     * on failure, with the reason in {@link #lastError}.
     */
    public static byte[] mlHfAnalyze(long handle, byte[] text) {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_HF_ANALYZE.invokeExact(handle, s.bytes(text), (long) text.length, ptrOf(slot), lenOf(slot));
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Opens the ONNX sparse encoding model ({@code sparse} true) or cross-encoder at {@code modelPath}
     * with the {@code tokenizer.json} bytes; its handle goes to {@code outHandle[0]}.
     */
    public static int mlOnnxModelOpen(boolean sparse, String modelPath, byte[] tokenizer, long[] outHandle) {
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        byte[] model = modelPath.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        MethodHandle open = sparse ? H.ML_SPARSE_ENCODING_OPEN : H.ML_TEXT_SIMILARITY_OPEN;
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            int rc = (int) open.invokeExact(s.bytes(model), (long) model.length, s.bytes(tokenizer), (long) tokenizer.length, handle);
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The sparse vectors of {@code texts} (encoded as for {@link #mlTextEmbeddingEmbed}) from sparse
     * encoding model {@code handle}, keyed by token id when {@code byId}: per text a {@code u32}
     * output count, then per output its {@code u32} name length, UTF-8 name and its weights as {@link
     * #mlSparseTokenize} gives a text's. {@code null} on failure, with the reason in {@link #lastError}.
     */
    public static byte[] mlSparseEncode(long handle, byte[] texts, boolean byId) {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_SPARSE_ENCODING.invokeExact(
                handle,
                s.bytes(texts),
                (long) texts.length,
                (byte) (byId ? 1 : 0),
                ptrOf(slot),
                lenOf(slot)
            );
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Scores each of {@code texts} (encoded as for {@link #mlTextEmbeddingEmbed}) against {@code query}
     * with cross-encoder {@code handle}: per text a {@code u32} output count, then each output as
     * {@link #mlTextEmbeddingEmbed} gives a vector. {@code null} on failure, with the reason in {@link
     * #lastError}.
     */
    public static byte[] mlTextSimilarity(long handle, String query, byte[] texts) {
        byte[] q = query.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_TEXT_SIMILARITY.invokeExact(
                handle,
                s.bytes(q),
                (long) q.length,
                s.bytes(texts),
                (long) texts.length,
                ptrOf(slot),
                lenOf(slot)
            );
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@link #mlHighlighterOpen} flag: a context longer than the token limit is read past its first chunk. */
    public static final int HIGHLIGHT_OVERFLOWING = 1;

    /** {@link #mlHighlighterOpen} flag: each chunk is padded. */
    public static final int HIGHLIGHT_PADDING = 2;

    /**
     * Opens the TorchScript sentence highlighting model at {@code modelPath} with the {@code
     * tokenizer.json} bytes, cutting a context into chunks of {@code maxLength} tokens that overlap by
     * {@code stride}, as the {@code HIGHLIGHT_*} bits of {@code flags} say. Its handle goes to {@code
     * outHandle[0]}. {@link #mlTorchInit} must have succeeded.
     */
    public static int mlHighlighterOpen(String modelPath, byte[] tokenizer, int maxLength, int stride, int flags, long[] outHandle) {
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        byte[] model = modelPath.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            int rc = (int) H.ML_HIGHLIGHTER_OPEN.invokeExact(
                s.bytes(model),
                (long) model.length,
                s.bytes(tokenizer),
                (long) tokenizer.length,
                maxLength,
                stride,
                (byte) flags,
                handle
            );
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The sentences of {@code context} that answer {@code question}, from highlighter {@code handle}: a
     * {@code u32} count, then per sentence its {@code u32} position, start and end (UTF-16 units), the
     * {@code u32} byte length of its text and the UTF-8 text. {@code null} on failure, with the reason
     * in {@link #lastError}.
     */
    public static byte[] mlHighlight(long handle, String question, String context) {
        byte[] q = question.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] c = context.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_HIGHLIGHT.invokeExact(handle, s.bytes(q), (long) q.length, s.bytes(c), (long) c.length, ptrOf(slot), lenOf(slot));
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Loads libtorch from directory {@code dir} and the first TorchScript shim of {@code shims}
     * (encoded as texts are for {@link #mlTextEmbeddingEmbed}) that runs against it, with {@code
     * threads} intra-op threads shared by every model (0: one per physical core). Later calls keep
     * the first.
     */
    public static int mlTorchInit(String dir, byte[] shims, int threads) {
        byte[] d = dir.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            return (int) H.ML_TORCH_INIT.invokeExact(s.bytes(d), (long) d.length, s.bytes(shims), (long) shims.length, threads);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@link #mlTorchOpen} functions: a {@code sentence_transformers} text embedding model. */
    public static final int TORCH_SENTENCE_TRANSFORMERS = 0;

    /** A {@code huggingface_transformers} text embedding model, pooled natively. */
    public static final int TORCH_HUGGING_FACE = 1;

    /** A sparse encoding model, read with {@link #mlSparseEncode}. */
    public static final int TORCH_SPARSE_ENCODING = 2;

    /** A cross-encoder, read with {@link #mlTextSimilarity}. */
    public static final int TORCH_TEXT_SIMILARITY = 3;

    /**
     * Opens the TorchScript model at {@code modelPath} with the {@code tokenizer.json} bytes for
     * {@code function} (one of the {@code TORCH_*} constants); a Hugging Face text embedding model is
     * pooled by the {@code pooling} mode name and normalized when {@code normalize}. Its handle goes to
     * {@code outHandle[0]}. {@link #mlTorchInit} must have succeeded.
     */
    public static int mlTorchOpen(int function, String modelPath, byte[] tokenizer, String pooling, boolean normalize, long[] outHandle) {
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        byte[] model = modelPath.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] mode = pooling.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            int rc = (int) H.ML_TORCH_OPEN.invokeExact(
                (byte) function,
                s.bytes(model),
                (long) model.length,
                s.bytes(tokenizer),
                (long) tokenizer.length,
                s.bytes(mode),
                (long) mode.length,
                (byte) (normalize ? 1 : 0),
                handle
            );
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The outputs of TorchScript text embedding model {@code handle} for each of {@code texts}
     * (encoded as for {@link #mlTextEmbeddingEmbed}): per text a {@code u32} output count, then per
     * output its {@code u32} name length and UTF-8 name, {@code u32} dict number (0 outside a dict),
     * {@code u8} data type (0 {@code FLOAT32}, 1 {@code INT64}), {@code u32} rank and dimensions, and
     * the values, all little endian. {@code null} on failure, with the reason in {@link #lastError}.
     */
    public static byte[] mlTorchEmbed(long handle, byte[] texts) {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_TORCH_TEXT_EMBEDDING.invokeExact(handle, s.bytes(texts), (long) texts.length, ptrOf(slot), lenOf(slot));
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** A new ml-commons master key (base64), or {@code null} with the reason in {@link #lastError}. */
    public static String mlMasterKey() {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_MASTER_KEY.invokeExact(ptrOf(slot), lenOf(slot));
            return rc == OK ? new String(take(slot), java.nio.charset.StandardCharsets.UTF_8) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Credential value {@code plain} encrypted under the base64 master key {@code key}, as
     * ml-commons stores it; {@code null} on failure, with the reason in {@link #lastError}.
     */
    public static String mlCredentialEncrypt(String key, String plain) {
        byte[] k = key.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] p = plain.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_CREDENTIAL_ENCRYPT.invokeExact(s.bytes(k), (long) k.length, s.bytes(p), (long) p.length, ptrOf(slot), lenOf(slot));
            return rc == OK ? new String(take(slot), java.nio.charset.StandardCharsets.UTF_8) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Stored credential value {@code stored} decrypted under the base64 master key {@code key};
     * {@code null} on failure, with the reason in {@link #lastError}.
     */
    public static String mlCredentialDecrypt(String key, String stored) {
        byte[] k = key.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] p = stored.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_CREDENTIAL_DECRYPT.invokeExact(s.bytes(k), (long) k.length, s.bytes(p), (long) p.length, ptrOf(slot), lenOf(slot));
            return rc == OK ? new String(take(slot), java.nio.charset.StandardCharsets.UTF_8) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Opens connector document {@code connectorJson} (credentials as stored, decrypted under the
     * base64 master key {@code key}) as a remote model; its handle goes to {@code outHandle[0]}.
     */
    public static int mlRemoteOpen(String connectorJson, String key, long[] outHandle) {
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        byte[] d = connectorJson.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] k = key.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            int rc = (int) H.ML_REMOTE_OPEN.invokeExact(s.bytes(d), (long) d.length, s.bytes(k), (long) k.length, handle);
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * One step of a call to remote model {@code handle}: the JSON request and answer that
     * {@code ml_remote.rs} documents. {@code null} on failure, with the status in {@code status[0]}
     * and the reason in {@link #lastError}.
     */
    public static String mlRemoteCall(long handle, String request, int[] status) {
        byte[] r = request.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_REMOTE_CALL.invokeExact(handle, s.bytes(r), (long) r.length, ptrOf(slot), lenOf(slot));
            status[0] = rc;
            return rc == OK ? new String(take(slot), java.nio.charset.StandardCharsets.UTF_8) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Releases a handle from any of the {@code ml*Open} calls. */
    public static int mlModelClose(long handle) {
        try {
            return (int) H.ML_MODEL_CLOSE.invokeExact(handle);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
