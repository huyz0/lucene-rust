/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.lucenerust.opensearch.painless;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Scorable;
import org.lucenerust.opensearch.NativeBridge;

import org.opensearch.OpenSearchException;
import org.opensearch.Version;
import org.opensearch.index.mapper.BooleanFieldMapper;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.painless.CompilerSettings;
import org.opensearch.painless.PainlessScript;
import org.opensearch.painless.PainlessScriptEngine;
import org.opensearch.painless.WriterConstants;
import org.opensearch.painless.lookup.PainlessLookup;
import org.opensearch.script.FilterScript;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptException;
import org.opensearch.script.ScriptService;

/**
 * The Painless scripts the native engine runs in place of the classes Painless generates: each
 * script compiled once to IR ({@link PainlessIr}) with the node's own Painless engine, its
 * context's allowlist and its compiler settings.
 */
public final class PainlessScore {

    /** What a field reads as in a script's {@code doc}, in the native node's numbering. */
    public static final byte LONG = 0, DOUBLE = 1, FLOAT = 2, HALF_FLOAT = 3, BOOLEAN = 4, KEYWORD = 5, UNSUPPORTED = (byte) 254,
        UNMAPPED = (byte) 255;

    /** Scripts kept compiled; past this many the cache starts over. */
    private static final int CACHE_LIMIT = 1024;

    private static volatile PainlessScriptEngine engine;
    private static volatile CompilerSettings settings;

    private static final ConcurrentHashMap<Key, Object> CACHE = new ConcurrentHashMap<>();

    private record Key(ScriptContext<?> context, String source) {}

    /** {@link #check} by node: "" when the native runtime runs it, else why not. */
    private static final ConcurrentHashMap<CheckKey, String> CHECKED = new ConcurrentHashMap<>();

    /**
     * What a check depends on: the compiled script (by identity: one per source in {@link
     * #CACHE}), its context ({@link #SCORE}, {@link #SCORE_WITH_SCORE} or {@link #FILTER}), its
     * params and its fields' kinds.
     */
    private record CheckKey(PainlessIr.Compiled compiled, byte context, ByteBuffer params, ByteBuffer kinds) {
        @Override
        public boolean equals(Object o) {
            return o instanceof CheckKey k && k.compiled == compiled && k.context == context && k.params.equals(params)
                && k.kinds.equals(kinds);
        }

        @Override
        public int hashCode() {
            return (System.identityHashCode(compiled) * 31 + params.hashCode()) * 31 + kinds.hashCode() + context;
        }
    }

    /** A script's context on the wire: a score script, one reading {@code _score}, a filter script. */
    public static final byte SCORE = 0, SCORE_WITH_SCORE = 1, FILTER = 2;

    /** A script the request sends to the native runtime, as Java makes it again for its error. */
    private interface Run {
        /** The script for {@code leaf}. */
        Object script(LeafReaderContext leaf) throws IOException;

        /** The {@code ScriptException} {@code script} raises on {@code doc} scored {@code score}, or null if it runs. */
        ScriptException rethrown(Object script, int doc, float score);
    }

    /**
     * A score script: its factory, and the index name, shard and index version {@code
     * ScriptScoreQuery} gives each script it makes.
     */
    private record ScoreRun(ScoreScript.LeafFactory factory, String index, int shard, Version version) implements Run {
        @Override
        public Object script(LeafReaderContext leaf) throws IOException {
            ScoreScript script = factory.newInstance(leaf);
            script._setIndexName(index);
            script._setShard(shard);
            script._setIndexVersion(version);
            return script;
        }

        @Override
        public ScriptException rethrown(Object s, int doc, float score) {
            ScoreScript script = (ScoreScript) s;
            script.setScorer(new Scorable() {
                @Override
                public float score() {
                    return score;
                }
            });
            script.setDocument(doc);
            try {
                script.execute(null);
            } catch (ScriptException e) {
                return e;
            }
            return null;
        }
    }

    /** A {@code script} query's filter script, as {@code ScriptQuery} makes it. */
    private record FilterRun(FilterScript.LeafFactory factory) implements Run {
        @Override
        public Object script(LeafReaderContext leaf) throws IOException {
            return factory.newInstance(leaf);
        }

        @Override
        public ScriptException rethrown(Object s, int doc, float score) {
            FilterScript script = (FilterScript) s;
            script.setDocument(doc);
            try {
                script.execute();
            } catch (ScriptException e) {
                return e;
            }
            return null;
        }
    }

    /** The scripts the request on this thread sends to the native runtime, by run number. */
    private static final ThreadLocal<List<Run>> RUNS = ThreadLocal.withInitial(ArrayList::new);

    /** What a script's error starts with in the native runtime's message ({@code SCRIPT_ERROR}). */
    private static final String SCRIPT_ERROR = "\u0001painless-script-error\u0001";

    /**
     * What a script's invalid (negative or NaN) score starts with in the native runtime's message
     * ({@code SCORE_ERROR}); {@code ScriptScoreQuery} raises it as an IllegalArgumentException.
     */
    private static final String SCORE_ERROR = "\u0001painless-score-error\u0001";

    /**
     * What an {@code OpenSearchException} of a query holding a native script starts with
     * ({@code OPENSEARCH_ERROR}): {@code function_score}'s missing value or invalid final score.
     */
    private static final String OPENSEARCH_ERROR = "\u0001opensearch-error\u0001";

    private PainlessScore() {}

    /** Takes the node's Painless engine from {@code scripts}; without it every script runs in Java. */
    public static void initialize(ScriptService scripts) {
        try {
            Field f = ScriptService.class.getDeclaredField("engines");
            f.setAccessible(true);
            if (((Map<?, ?>) f.get(scripts)).get(PainlessScriptEngine.NAME) instanceof PainlessScriptEngine p) {
                initialize(p);
            }
        } catch (ReflectiveOperationException | RuntimeException e) {
            engine = null;
        }
    }

    /** Uses {@code e} as the node's engine, with its default compiler settings. */
    public static void initialize(PainlessScriptEngine e) throws ReflectiveOperationException {
        Field s = PainlessScriptEngine.class.getDeclaredField("defaultCompilerSettings");
        s.setAccessible(true);
        settings = (CompilerSettings) s.get(e);
        engine = e;
        CACHE.clear();
        CHECKED.clear();
    }

    /**
     * {@code source} compiled for the score context: a {@link PainlessIr.Compiled}, or a string
     * saying why the native runtime cannot have it.
     */
    public static Object score(String source) {
        return compile(ScoreScript.CONTEXT, ScoreScript.class, source);
    }

    /** {@code source} compiled for the filter context, as {@link #score}. */
    public static Object filter(String source) {
        return compile(FilterScript.CONTEXT, FilterScript.class, source);
    }

    private static Object compile(ScriptContext<?> context, Class<?> scriptClass, String source) {
        PainlessScriptEngine e = engine;
        if (e == null) {
            return "no Painless engine";
        }
        Key key = new Key(context, source);
        Object hit = CACHE.get(key);
        if (hit != null) {
            return hit;
        }
        PainlessLookup lookup = e.getContextsToLookups().get(context);
        Object compiled;
        if (lookup == null) {
            compiled = "no lookup for the context";
        } else {
            try {
                compiled = PainlessIr.compile(lookup, scriptClass, source, settings);
            } catch (PainlessIr.Unsupported u) {
                compiled = u.getMessage();
            } catch (RuntimeException r) {
                // Painless refused the script: the search raises its own error in Java.
                compiled = "does not compile";
            }
        }
        if (CACHE.size() >= CACHE_LIMIT) {
            CACHE.clear();
        }
        CACHE.put(key, compiled);
        return compiled;
    }

    /**
     * Whether the native runtime runs {@code compiled} in {@code context} with {@code params} and
     * fields read as {@code kinds} exactly as Painless would: null when it does, else why not. The check is
     * static, over the whole script before any of it runs, on the node {@code spec} writes (its
     * fields after the sub-query), and kept per script, params and kinds, refusals too.
     */
    public static String check(PainlessIr.Compiled compiled, byte context, byte[] params, byte[] kinds, Supplier<byte[]> spec) {
        CheckKey key = new CheckKey(compiled, context, ByteBuffer.wrap(params), ByteBuffer.wrap(kinds));
        String hit = CHECKED.get(key);
        if (hit == null) {
            byte[][] out = new byte[1][];
            int rc = NativeBridge.painlessCheck(spec.get(), out);
            hit = rc != NativeBridge.OK ? "check failed: " + NativeBridge.lastError()
                : out[0] == null ? "" : new String(out[0], java.nio.charset.StandardCharsets.UTF_8);
            if (CHECKED.size() >= CACHE_LIMIT) {
                CHECKED.clear();
            }
            CHECKED.put(key, hit);
        }
        return hit.isEmpty() ? null : hit;
    }

    /** Starts a request on this thread: no script sent yet. */
    public static void beginRequest() {
        RUNS.get().clear();
        NativeBridge.clearLastError();
    }

    /** Ends the request on this thread. */
    public static void endRequest() {
        RUNS.get().clear();
    }

    /** The number in this request, for its errors, of a script {@code factory} makes for an index's shard. */
    public static int run(ScoreScript.LeafFactory factory, String index, int shard, Version version) {
        return run(new ScoreRun(factory, index, shard, version));
    }

    /** The number in this request, for its errors, of a {@code script} query's filter script. */
    public static int run(FilterScript.LeafFactory factory) {
        return run(new FilterRun(factory));
    }

    private static int run(Run run) {
        List<Run> runs = RUNS.get();
        runs.add(run);
        return runs.size() - 1;
    }

    /** Whether {@code error} is a script's own exception or invalid score, not an internal failure. */
    public static boolean scriptFailed(String error) {
        return error != null && (error.contains(SCRIPT_ERROR) || error.contains(SCORE_ERROR) || error.contains(OPENSEARCH_ERROR));
    }

    /** Whether this request sent a script to the native runtime. */
    public static boolean ran() {
        return RUNS.get().isEmpty() == false;
    }

    /**
     * What a native search that ran this request's scripts fails with, from its {@code error}: the
     * {@code ScriptException} Painless raises for a script's exception, or else an internal error.
     * Never a reason to run the search again in Java: the scripts already ran. Painless's own
     * exception (its message, the Java frames in its script stack) comes from the script run once
     * more in Java on the one document it threw at, with the score it had; the static check let
     * through only scripts with no effect but their score, so that run changes nothing.
     */
    public static RuntimeException failure(String error, IndexReader reader) {
        int score = error == null ? -1 : error.indexOf(SCORE_ERROR);
        if (score >= 0) {
            return new IllegalArgumentException(error.substring(score + SCORE_ERROR.length()));
        }
        int opensearch = error == null ? -1 : error.indexOf(OPENSEARCH_ERROR);
        if (opensearch >= 0) {
            return new OpenSearchException(error.substring(opensearch + OPENSEARCH_ERROR.length()));
        }
        int at = error == null ? -1 : error.indexOf(SCRIPT_ERROR);
        if (at < 0) {
            return new IllegalStateException("native script_score search failed: " + error);
        }
        // Run number, segment, document, score bits, source offset, exception class, message.
        String[] parts = error.substring(at + SCRIPT_ERROR.length()).split("\u0001", 7);
        try {
            Run run = RUNS.get().get(Integer.parseInt(parts[0]));
            int leaf = Integer.parseInt(parts[1]);
            int doc = Integer.parseInt(parts[2]);
            float docScore = Float.intBitsToFloat(Integer.parseUnsignedInt(parts[3]));
            int offset = Integer.parseInt(parts[4]);
            String message = parts[6].startsWith("1") ? parts[6].substring(1) : null;
            List<LeafReaderContext> leaves = reader.leaves();
            if (leaf >= leaves.size()) {
                return new IllegalStateException("native script_score search failed: " + error);
            }
            Object script = run.script(leaves.get(leaf));
            ScriptException painless = run.rethrown(script, doc, docScore);
            if (painless != null) {
                return painless;
            }
            Throwable cause = exception(parts[5], message);
            // Painless numbers a script's lines by source offset + 1.
            cause.setStackTrace(new StackTraceElement[] { new StackTraceElement(WriterConstants.CLASS_NAME, "execute", null, offset < 0 ? -1 : offset + 1) });
            return ((PainlessScript) script).convertToScriptException(cause, Map.of());
        } catch (IOException | RuntimeException e) {
            return new IllegalStateException("native script_score search failed: " + error, e);
        }
    }

    /** An exception of class {@code name} with {@code message} (null for none). */
    private static Throwable exception(String name, String message) {
        try {
            Class<?> c = Class.forName(name, false, PainlessScore.class.getClassLoader());
            if (Throwable.class.isAssignableFrom(c)) {
                if (message == null) {
                    try {
                        return (Throwable) c.getConstructor().newInstance();
                    } catch (NoSuchMethodException e) {
                        // Falls through to the message constructor.
                    }
                }
                return (Throwable) c.getConstructor(String.class).newInstance(message);
            }
        } catch (ReflectiveOperationException | LinkageError e) {
            // An exception the plugin cannot build stands as an IllegalStateException naming it.
        }
        return new IllegalStateException(name + (message == null ? "" : ": " + message));
    }

    /**
     * How a field reads in {@code doc}: as {@code ScriptDocValues.Longs}, {@code Doubles}, {@code
     * Booleans} or {@code Strings} from its doc values, else {@link #UNSUPPORTED} (mapped, read in
     * Java) or {@link #UNMAPPED}.
     */
    public static byte fieldKind(MappedFieldType ft) {
        if (ft == null) {
            return UNMAPPED;
        }
        if (ft.hasDocValues() == false) {
            return UNSUPPORTED;
        }
        if (ft.getClass() == NumberFieldMapper.NumberFieldType.class) {
            return switch (((NumberFieldMapper.NumberFieldType) ft).numberType()) {
                case LONG, INTEGER, SHORT, BYTE -> LONG;
                case DOUBLE -> DOUBLE;
                case FLOAT -> FLOAT;
                case HALF_FLOAT -> HALF_FLOAT;
                default -> UNSUPPORTED;
            };
        }
        if (ft.getClass() == BooleanFieldMapper.BooleanFieldType.class) {
            return BOOLEAN;
        }
        if (ft.getClass() == KeywordFieldMapper.KeywordFieldType.class) {
            return KEYWORD;
        }
        return UNSUPPORTED;
    }
}
