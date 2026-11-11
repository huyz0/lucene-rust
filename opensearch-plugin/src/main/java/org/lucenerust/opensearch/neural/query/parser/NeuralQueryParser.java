/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.query.parser;

import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

public class NeuralQueryParser {
    /**
     * @param in stream input
     * @return vector supplier if it exists
     * @throws IOException
     */
    public static Supplier<float[]> vectorSupplierStreamInput(final StreamInput in) throws IOException {
        java.util.Objects.requireNonNull(in, "in is marked non-null but is null");
        if (in.readBoolean() == false) {
            return null;
        } else {
            final float[] vector = in.readFloatArray();
            return () -> vector;
        }
    }

    /**
     * Write the vector to the stream if it exists.
     * @param out output stream
     * @param vectorSupplier vector supplier
     * @throws IOException
     */
    public static void vectorSupplierStreamOutput(final StreamOutput out, final Supplier<float[]> vectorSupplier) throws IOException {
        java.util.Objects.requireNonNull(out, "out is marked non-null but is null");
        if (vectorSupplier == null || vectorSupplier.get() == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            out.writeFloatArray(vectorSupplier.get());
        }
    }

    /**
     * @param in stream input
     * @return modelIdToVectorSupplierMap if it exists
     * @throws IOException
     */
    public static Map<String, Supplier<float[]>> modelIdToVectorSupplierMapStreamInput(final StreamInput in) throws IOException {
        java.util.Objects.requireNonNull(in, "in is marked non-null but is null");
        if (in.readBoolean() == false) {
            return null;
        } else {
            final Map<String, Supplier<float[]>> modelIdToVectorSupplierMap = new HashMap<>();
            final int size = in.readVInt();
            for (int i = 0; i < size; i++) {
                final String modelId = in.readString();
                final float[] vector = in.readFloatArray();
                final Supplier<float[]> vectorSupplier = () -> vector;
                modelIdToVectorSupplierMap.put(modelId, vectorSupplier);
            }
            return modelIdToVectorSupplierMap;
        }
    }

    /**
     * Write the model id to vector map to the stream if all the vector suppliers have a value. All the vector suppliers
     * have values means all the async inference API calls are done it's safe to copy over the values to create the query
     * in another node. Otherwise, we should keep modelIdToVectorSupplierMap as null to trigger the inference calls
     * after the query is passed to another node.
     * @param out stream output
     * @param modelIdToVectorSupplierMap model id to vector supplier map
     * @throws IOException
     */
    public static void modelIdToVectorSupplierMapStreamOutput(final StreamOutput out, final Map<String, Supplier<float[]>> modelIdToVectorSupplierMap) throws IOException {
        java.util.Objects.requireNonNull(out, "out is marked non-null but is null");
        if (modelIdToVectorSupplierMap == null) {
            out.writeBoolean(false);
            return;
        }
        out.writeBoolean(true);
        int size = modelIdToVectorSupplierMap.size();
        out.writeVInt(size);
        for (Map.Entry<String, Supplier<float[]>> entry : modelIdToVectorSupplierMap.entrySet()) {
            try {
                out.writeString(entry.getKey());
                out.writeFloatArray(entry.getValue().get());
            } catch (Exception e) {
                throw new RuntimeException(String.format(Locale.ROOT, "Failed to write modelIdToVectorSupplierMap to StreamOutput for the neural query due to: %s", e.getMessage()), e);
            }
        }
    }

    /**
     * @param in stream input
     * @return queryTokensMapSupplier if it exists
     * @throws IOException
     */
    public static Supplier<Map<String, Float>> queryTokensMapSupplierStreamInput(final StreamInput in) throws IOException {
        java.util.Objects.requireNonNull(in, "in is marked non-null but is null");
        if (in.readBoolean() == false) {
            return null;
        } else {
            final Map<String, Float> queryTokens = in.readMap(StreamInput::readString, StreamInput::readFloat);
            return () -> queryTokens;
        }
    }

    /**
     * Write it to the stream if it exists.
     * @param out stream output
     * @param queryTokensMapSupplier query token map supplier for the sparse model
     * @throws IOException
     */
    public static void queryTokensMapSupplierStreamOutput(final StreamOutput out, final Supplier<Map<String, Float>> queryTokensMapSupplier) throws IOException {
        java.util.Objects.requireNonNull(out, "out is marked non-null but is null");
        if (queryTokensMapSupplier == null || queryTokensMapSupplier.get() == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            out.writeMap(queryTokensMapSupplier.get(), StreamOutput::writeString, StreamOutput::writeFloat);
        }
    }

    /**
     * @param in
     * @return
     * @throws IOException
     */
    public static Map<String, Supplier<Map<String, Float>>> modelIdToQueryTokensSupplierMapStreamInput(final StreamInput in) throws IOException {
        java.util.Objects.requireNonNull(in, "in is marked non-null but is null");
        if (in.readBoolean() == false) {
            return null;
        } else {
            final Map<String, Supplier<Map<String, Float>>> modelIdToQueryTokensSupplierMap = new HashMap<>();
            final int size = in.readVInt();
            for (int i = 0; i < size; i++) {
                final String modelId = in.readString();
                final Map<String, Float> queryTokenMap = in.readMap(StreamInput::readString, StreamInput::readFloat);
                final Supplier<Map<String, Float>> queryTokenMapSupplier = () -> queryTokenMap;
                modelIdToQueryTokensSupplierMap.put(modelId, queryTokenMapSupplier);
            }
            return modelIdToQueryTokensSupplierMap;
        }
    }

    /**
     * Write the model id to query token map to the stream if all the query token suppliers have a value. All the query
     * token suppliers have values means all the async inference API calls are done it's safe to copy over the values
     * to create the query in another node. Otherwise, we should keep modelIdToVectorSupplierMap as null to trigger
     * the inference calls after the query is passed to another node.
     * @param out stream output
     * @param modelIdToQueryTokensMapSupplier model id to query token map
     * @throws IOException
     */
    public static void modelIdToQueryTokensSupplierMapStreamOutput(final StreamOutput out, final Map<String, Supplier<Map<String, Float>>> modelIdToQueryTokensMapSupplier) throws IOException {
        java.util.Objects.requireNonNull(out, "out is marked non-null but is null");
        if (modelIdToQueryTokensMapSupplier == null) {
            out.writeBoolean(false);
            return;
        }
        out.writeBoolean(true);
        int size = modelIdToQueryTokensMapSupplier.size();
        out.writeVInt(size);
        for (Map.Entry<String, Supplier<Map<String, Float>>> entry : modelIdToQueryTokensMapSupplier.entrySet()) {
            try {
                out.writeString(entry.getKey());
                out.writeMap(entry.getValue().get(), StreamOutput::writeString, StreamOutput::writeFloat);
            } catch (Exception e) {
                throw new RuntimeException(String.format(Locale.ROOT, "Failed to write modelIdToQueryTokensSupplierMap to StreamOutput for the neural query due to: %s", e.getMessage()), e);
            }
        }
    }

    /**
     * modelIdToTwoPhaseSharedQueryTokenStreamInput
     * @param in input stream
     * @return modelIdToTwoPhaseSharedQueryToken
     * @throws IOException
     */
    public static Map<String, Map<String, Float>> modelIdToTwoPhaseSharedQueryTokenStreamInput(final StreamInput in) throws IOException {
        java.util.Objects.requireNonNull(in, "in is marked non-null but is null");
        if (in.readBoolean() == false) {
            return null;
        } else {
            final Map<String, Map<String, Float>> modelIdToTwoPhaseSharedQueryToken = new HashMap<>();
            final int size = in.readVInt();
            for (int i = 0; i < size; i++) {
                final String modelId = in.readString();
                final Map<String, Float> twoPhaseSharedQueryToken = in.readMap(StreamInput::readString, StreamInput::readFloat);
                modelIdToTwoPhaseSharedQueryToken.put(modelId, twoPhaseSharedQueryToken);
            }
            return modelIdToTwoPhaseSharedQueryToken;
        }
    }

    /**
     * modelIdToTwoPhaseSharedQueryTokenStreamOutput
     * @param out output stream
     * @param modelIdToTwoPhaseSharedQueryToken
     * @throws IOException
     */
    public static void modelIdToTwoPhaseSharedQueryTokenStreamOutput(final StreamOutput out, final Map<String, Map<String, Float>> modelIdToTwoPhaseSharedQueryToken) throws IOException {
        java.util.Objects.requireNonNull(out, "out is marked non-null but is null");
        if (modelIdToTwoPhaseSharedQueryToken == null) {
            out.writeBoolean(false);
            return;
        }
        out.writeBoolean(true);
        int size = modelIdToTwoPhaseSharedQueryToken.size();
        out.writeVInt(size);
        for (Map.Entry<String, Map<String, Float>> entry : modelIdToTwoPhaseSharedQueryToken.entrySet()) {
            try {
                out.writeString(entry.getKey());
                out.writeMap(entry.getValue(), StreamOutput::writeString, StreamOutput::writeFloat);
            } catch (Exception e) {
                throw new RuntimeException(String.format(Locale.ROOT, "Failed to write modelIdToTwoPhaseSharedQueryToken to StreamOutput for the neural query due to: %s", e.getMessage()), e);
            }
        }
    }

    /**
     * modelIdToTwoPhaseSharedQueryTokenSupplierStreamInput
     * @param in input stream
     * @return modelIdToTwoPhaseSharedQueryTokenSupplier
     * @throws IOException
     */
    public static Supplier<Map<String, Map<String, Float>>> modelIdToTwoPhaseSharedQueryTokenSupplierStreamInput(final StreamInput in) throws IOException {
        java.util.Objects.requireNonNull(in, "in is marked non-null but is null");
        if (in.readBoolean() == false) {
            return null;
        }
        Map<String, Map<String, Float>> modelIdToTwoPhaseSharedQueryToken = modelIdToTwoPhaseSharedQueryTokenStreamInput(in);
        if (modelIdToTwoPhaseSharedQueryToken == null) {
            return null;
        } else {
            return () -> modelIdToTwoPhaseSharedQueryToken;
        }
    }

    /**
     * modelIdToTwoPhaseSharedQueryTokenSupplierStreamOutput
     * @param out output stream
     * @param modelIdToTwoPhaseSharedQueryTokenSupplier
     * @throws IOException
     */
    public static void modelIdToTwoPhaseSharedQueryTokenSupplierStreamOutput(final StreamOutput out, final Supplier<Map<String, Map<String, Float>>> modelIdToTwoPhaseSharedQueryTokenSupplier) throws IOException {
        java.util.Objects.requireNonNull(out, "out is marked non-null but is null");
        if (modelIdToTwoPhaseSharedQueryTokenSupplier == null) {
            out.writeBoolean(false);
            return;
        }
        out.writeBoolean(true);
        try {
            modelIdToTwoPhaseSharedQueryTokenStreamOutput(out, modelIdToTwoPhaseSharedQueryTokenSupplier.get());
        } catch (Exception e) {
            throw new RuntimeException(String.format(Locale.ROOT, "Failed to write modelIdToTwoPhaseSharedQueryTokenSupplier to StreamOutput for the neural query due to: %s", e.getMessage()), e);
        }
    }
}
