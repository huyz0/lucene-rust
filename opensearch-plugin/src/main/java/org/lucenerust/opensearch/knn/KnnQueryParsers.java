/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.opensearch.common.ValidationException;
import org.opensearch.core.ParseField;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ObjectParser;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * k-NN's {@code MethodParametersParser}, {@code MethodParameter} and {@code RescoreParser}: the
 * {@code method_parameters} and {@code rescore} objects of a query that carries knn options (the
 * {@code neural} query), with stock's messages and wire layout.
 */
public final class KnnQueryParsers {
    public static final ParseField METHOD_PARAMS_FIELD = new ParseField("method_parameters");
    public static final String RESCORE_PARAMETER = "rescore";
    public static final String RESCORE_OVERSAMPLE_PARAMETER = "oversample_factor";

    private static final String EF_SEARCH = "ef_search";
    private static final String NPROBES = "nprobes";
    /** {@code MethodParameter.values()} in declaration order: the wire order. */
    private static final List<String> PARAMETERS = List.of(EF_SEARCH, NPROBES);
    /**
     * This plugin's own opt-in method parameter, which stock does not have: the graph walk's adaptive
     * beam stop ({@code KnnFloatVectorQuery::exploration_slack}), a number {@code >= 0}. It travels in
     * the stream's map of other parameters, so the stock wire order is unchanged.
     */
    public static final String EXPLORATION_SLACK = "exploration_slack";

    private static Float parseSlack(Object value) {
        final float v;
        try {
            v = Float.parseFloat(String.valueOf(value));
        } catch (final NumberFormatException e) {
            throw new IllegalArgumentException(EXPLORATION_SLACK + " value must be a number");
        }
        if (Float.isFinite(v) == false || v < 0) {
            throw new IllegalArgumentException(EXPLORATION_SLACK + " should be a finite number >= 0");
        }
        return v;
    }

    private static final ObjectParser<float[], Void> RESCORE_PARSER = new ObjectParser<>(RESCORE_PARAMETER, () -> new float[] {
        RescoreContext.DEFAULT_OVERSAMPLE_FACTOR });
    static {
        RESCORE_PARSER.declareFloat((f, v) -> f[0] = v, new ParseField(RESCORE_OVERSAMPLE_PARAMETER));
    }

    private KnnQueryParsers() {}

    private static Integer parseInteger(Object value, String name) {
        try {
            return Integer.parseInt(String.valueOf(value));
        } catch (final NumberFormatException e) {
            throw new IllegalArgumentException(name + " value must be an integer");
        }
    }

    /** {@code MethodParameter.parse}: nprobes reports itself as ef_search, as stock does. */
    private static Integer parse(String name, Object value) {
        return parseInteger(value, EF_SEARCH);
    }

    /** {@code MethodParametersParser.fromXContent}. */
    public static Map<String, ?> methodParametersFromXContent(XContentParser parser) throws IOException {
        final Map<String, Object> json = parser.map();
        if (json.isEmpty()) {
            throw new ParsingException(parser.getTokenLocation(), METHOD_PARAMS_FIELD.getPreferredName() + " cannot be empty");
        }
        final Map<String, Object> methodParameters = new HashMap<>();
        for (Map.Entry<String, Object> e : json.entrySet()) {
            if (EXPLORATION_SLACK.equals(e.getKey())) {
                try {
                    methodParameters.put(e.getKey(), parseSlack(e.getValue()));
                } catch (final Exception exception) {
                    throw new ParsingException(parser.getTokenLocation(), exception.getMessage());
                }
                continue;
            }
            if (PARAMETERS.contains(e.getKey()) == false) {
                throw new ParsingException(
                    parser.getTokenLocation(),
                    "[" + KnnQueryBuilder.NAME + "] unknown method parameter found [" + e.getKey() + "]"
                );
            }
            try {
                methodParameters.put(e.getKey(), parse(e.getKey(), e.getValue()));
            } catch (final Exception exception) {
                throw new ParsingException(parser.getTokenLocation(), exception.getMessage());
            }
        }
        return methodParameters.isEmpty() ? null : methodParameters;
    }

    /** {@code MethodParametersParser.validateMethodParameters}. */
    static ValidationException validateMethodParameters(final Map<String, ?> methodParameters) {
        final List<String> errors = new ArrayList<>();
        for (final Map.Entry<String, ?> p : methodParameters.entrySet()) {
            if (EXPLORATION_SLACK.equals(p.getKey())) {
                try {
                    parseSlack(p.getValue());
                } catch (final IllegalArgumentException e) {
                    errors.add("Validation Failed: 1: " + e.getMessage() + ";");
                }
            } else if (PARAMETERS.contains(p.getKey())) {
                Integer v = parse(p.getKey(), p.getValue());
                if (v == null || v <= 0) {
                    errors.add("Validation Failed: 1: " + p.getKey() + " should be greater than 0;");
                }
            } else {
                errors.add(p.getKey() + " is not a valid method parameter");
            }
        }
        if (errors.isEmpty()) {
            return null;
        }
        ValidationException validationException = new ValidationException();
        validationException.addValidationErrors(errors);
        return validationException;
    }

    /** {@code exploration_slack} of a parsed method parameter map, or null. */
    public static Float explorationSlack(Map<String, ?> methodParameters) {
        Object v = methodParameters == null ? null : methodParameters.get(EXPLORATION_SLACK);
        return v == null ? null : parseSlack(v);
    }

    /** {@code MethodParametersParser.streamInput}. */
    public static Map<String, ?> methodParametersStreamInput(StreamInput in) throws IOException {
        if (in.readBoolean() == false) {
            return null;
        }
        final Map<String, Object> methodParameters = new HashMap<>();
        for (int i = 0; i < PARAMETERS.size(); i++) {
            String name = in.readString();
            Object value = in.readGenericValue();
            if (value != null) {
                methodParameters.put(name, parse(name, value));
            }
        }
        methodParameters.putAll(in.readMap(StreamInput::readString, StreamInput::readGenericValue));
        return methodParameters.isEmpty() ? null : methodParameters;
    }

    /** {@code MethodParametersParser.streamOutput}. */
    public static void methodParametersStreamOutput(StreamOutput out, Map<String, ?> methodParameters) throws IOException {
        if (methodParameters == null || methodParameters.isEmpty()) {
            out.writeBoolean(false);
            return;
        }
        out.writeBoolean(true);
        Map<String, Object> others = new HashMap<>();
        for (Map.Entry<String, ?> e : methodParameters.entrySet()) {
            if (PARAMETERS.contains(e.getKey()) == false) {
                others.put(e.getKey(), e.getValue());
            }
        }
        for (String name : PARAMETERS) {
            out.writeString(name);
            out.writeGenericValue(methodParameters.get(name));
        }
        out.writeMap(others, StreamOutput::writeString, StreamOutput::writeGenericValue);
    }

    /** {@code MethodParametersParser.doXContent}. */
    public static void methodParametersDoXContent(XContentBuilder builder, Map<String, ?> methodParameters) throws IOException {
        if (methodParameters == null || methodParameters.isEmpty()) {
            return;
        }
        builder.startObject(METHOD_PARAMS_FIELD.getPreferredName());
        for (Map.Entry<String, ?> e : methodParameters.entrySet()) {
            if (e.getKey() != null && e.getValue() != null) {
                builder.field(e.getKey(), e.getValue());
            }
        }
        builder.endObject();
    }

    /** {@code RescoreParser.fromXContent}: an object with an optional {@code oversample_factor}. */
    public static RescoreContext rescoreFromXContent(XContentParser parser) {
        return new RescoreContext(RESCORE_PARSER.apply(parser, null)[0], true);
    }

    /** {@code RescoreParser.validate}. */
    static ValidationException validateRescore(RescoreContext rescore) {
        String message = null;
        if (rescore.getOversampleFactor() < RescoreContext.MIN_OVERSAMPLE_FACTOR) {
            message = String.format(
                Locale.ROOT,
                "Oversample factor [%f] cannot be less than [%f]",
                rescore.getOversampleFactor(),
                RescoreContext.MIN_OVERSAMPLE_FACTOR
            );
        } else if (rescore.getOversampleFactor() > RescoreContext.MAX_OVERSAMPLE_FACTOR) {
            message = String.format(
                Locale.ROOT,
                "Oversample factor [%f] cannot be more than [%f]",
                rescore.getOversampleFactor(),
                RescoreContext.MAX_OVERSAMPLE_FACTOR
            );
        }
        if (message == null) {
            return null;
        }
        ValidationException e = new ValidationException();
        e.addValidationError(message);
        return e;
    }

    /** {@code RescoreParser.streamInput}. */
    public static RescoreContext rescoreStreamInput(StreamInput in) throws IOException {
        Float oversample = in.readOptionalFloat();
        if (oversample == null) {
            return null;
        }
        return new RescoreContext(oversample, in.readBoolean());
    }

    /** {@code RescoreParser.streamOutput}. */
    public static void rescoreStreamOutput(StreamOutput out, RescoreContext rescore) throws IOException {
        out.writeOptionalFloat(rescore == null ? null : rescore.getOversampleFactor());
        if (rescore != null) {
            out.writeBoolean(rescore.isRescoreEnabled());
        }
    }

    /** {@code RescoreParser.doXContent}. */
    public static void rescoreDoXContent(XContentBuilder builder, RescoreContext rescore) throws IOException {
        builder.startObject(RESCORE_PARAMETER);
        builder.field(RESCORE_OVERSAMPLE_PARAMETER, rescore.getOversampleFactor());
        builder.endObject();
    }
}
