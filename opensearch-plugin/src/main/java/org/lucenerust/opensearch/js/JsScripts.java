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

package org.lucenerust.opensearch.js;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.lucenerust.opensearch.NativeBridge;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.script.AggregationScript;
import org.opensearch.script.FilterScript;
import org.opensearch.script.NumberSortScript;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptEngine;
import org.opensearch.script.ScriptException;
import org.opensearch.script.ScriptService;
import org.opensearch.script.TermsSetQueryScript;
import org.opensearch.search.lookup.SearchLookup;

/**
 * Search scripts in JavaScript ({@code "lang": "js"}), which the native engine runs on V8: an
 * opt-in script engine ({@link #ENABLED}, off by default) that needs a native library built with
 * the {@code js} feature ({@code ./gradlew -Pjs}).
 *
 * <p>A script is an expression or a function body that returns, reading {@code doc['f'].value},
 * {@code doc['f'].size()}, {@code params} and {@code _score}, in the score, filter, number sort,
 * aggregation and {@code terms_set} contexts. It runs only natively: compiling it checks its
 * syntax on V8, and a search the native engine does not run fails before any script runs, never
 * falling back to Java.
 */
public final class JsScripts {

    /** The language's name. */
    public static final String NAME = "js";

    /** Whether this node serves {@code "lang": "js"} scripts. */
    public static final Setting<Boolean> ENABLED = Setting.boolSetting("lucene_rust.script.js.enabled", false, Setting.Property.NodeScope);

    /** How long one search may run a script, over all its documents. */
    public static final Setting<TimeValue> TIMEOUT = Setting.timeSetting(
        "lucene_rust.script.js.timeout",
        TimeValue.timeValueSeconds(10),
        TimeValue.timeValueMillis(1),
        Setting.Property.NodeScope
    );

    /** The heap each search thread's V8 isolate may grow to. */
    public static final Setting<ByteSizeValue> MAX_HEAP = Setting.byteSizeSetting(
        "lucene_rust.script.js.max_heap",
        new ByteSizeValue(128, ByteSizeUnit.MB),
        new ByteSizeValue(16, ByteSizeUnit.MB),
        new ByteSizeValue(2, ByteSizeUnit.GB),
        Setting.Property.NodeScope
    );

    /** The settings this class reads. */
    public static List<Setting<?>> settings() {
        return List.of(ENABLED, TIMEOUT, MAX_HEAP);
    }

    /** The context byte's flag for a JavaScript script ({@code jvm_painless_score.rs}). */
    public static final byte JS_SCRIPT = (byte) 0x80;

    private static volatile int timeoutMillis = 10_000;
    private static volatile int heapMegabytes = 128;

    private JsScripts() {}

    /**
     * Adds the {@code js} engine to {@code scripts} when {@code settings} enable it: refused, at
     * node start, when the native library has no V8.
     */
    @SuppressWarnings("unchecked")
    public static void install(ScriptService scripts, Settings settings) {
        if (ENABLED.get(settings) == false) {
            return;
        }
        timeoutMillis = (int) Math.min(Integer.MAX_VALUE, Math.max(1, TIMEOUT.get(settings).millis()));
        heapMegabytes = (int) Math.max(1, MAX_HEAP.get(settings).getMb());
        String refused = check("1", (byte) 0);
        if (refused != null) {
            throw new IllegalStateException("[" + ENABLED.getKey() + "] is set, but js scripts cannot run on this node: " + refused);
        }
        try {
            Field f = ScriptService.class.getDeclaredField("engines");
            f.setAccessible(true);
            Map<String, ScriptEngine> engines = new HashMap<>((Map<String, ScriptEngine>) f.get(scripts));
            engines.put(NAME, new Engine());
            f.set(scripts, Collections.unmodifiableMap(engines));
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("cannot add the js script engine", e);
        }
    }

    /** The per-search time limit in milliseconds, and the heap limit in megabytes. */
    public static int timeoutMillis() {
        return timeoutMillis;
    }

    public static int heapMegabytes() {
        return heapMegabytes;
    }

    /** A {@code js} script's leaf factory: what the native engine runs. */
    public interface Leaf {
        String source();

        Map<String, Object> params();

        boolean needsScore();
    }

    /** What a search the native engine does not run raises when Java would run its script. */
    private static IllegalArgumentException javaRun() {
        return new IllegalArgumentException("[js] scripts run only in the native engine, which does not run this search");
    }

    private record ScoreLeaf(String source, Map<String, Object> params, boolean needsScore) implements Leaf, ScoreScript.LeafFactory {
        @Override
        public boolean needs_score() {
            return needsScore;
        }

        @Override
        public ScoreScript newInstance(LeafReaderContext ctx) {
            throw javaRun();
        }
    }

    private record FilterLeaf(String source, Map<String, Object> params, boolean needsScore) implements Leaf, FilterScript.LeafFactory {
        @Override
        public FilterScript newInstance(LeafReaderContext ctx) {
            throw javaRun();
        }
    }

    private record SortLeaf(String source, Map<String, Object> params, boolean needsScore) implements Leaf, NumberSortScript.LeafFactory {
        @Override
        public boolean needs_score() {
            return needsScore;
        }

        @Override
        public NumberSortScript newInstance(LeafReaderContext ctx) {
            throw javaRun();
        }
    }

    private record AggLeaf(String source, Map<String, Object> params, boolean needsScore) implements Leaf, AggregationScript.LeafFactory {
        @Override
        public boolean needs_score() {
            return needsScore;
        }

        @Override
        public AggregationScript newInstance(LeafReaderContext ctx) {
            throw javaRun();
        }
    }

    private record TermsSetLeaf(String source, Map<String, Object> params, boolean needsScore) implements Leaf,
        TermsSetQueryScript.LeafFactory {
        @Override
        public TermsSetQueryScript newInstance(LeafReaderContext ctx) {
            throw javaRun();
        }
    }

    /** The engine: each script checked on V8 as it compiles. */
    static final class Engine implements ScriptEngine {
        @Override
        public String getType() {
            return NAME;
        }

        @Override
        public <T> T compile(String name, String code, ScriptContext<T> context, Map<String, String> options) {
            byte probe;
            if (context == ScoreScript.CONTEXT) {
                probe = 0;
            } else if (context == FilterScript.CONTEXT) {
                probe = 2;
            } else if (context == NumberSortScript.CONTEXT) {
                probe = 3;
            } else if (context == AggregationScript.CONTEXT) {
                probe = 5;
            } else if (context == TermsSetQueryScript.CONTEXT) {
                probe = 6;
            } else {
                throw new IllegalArgumentException("[js] scripts cannot be used for context [" + context.name + "]");
            }
            String refused = check(code, probe);
            if (refused != null) {
                throw new ScriptException("compile error", new IllegalArgumentException(refused), List.of(code), code, NAME);
            }
            boolean score = names(code).contains("_score");
            Object factory;
            if (context == ScoreScript.CONTEXT) {
                factory = new ScoreFactory(code, score);
            } else if (context == FilterScript.CONTEXT) {
                factory = (FilterFactory) (params, lookup) -> new FilterLeaf(code, params, false);
            } else if (context == NumberSortScript.CONTEXT) {
                factory = (SortFactory) (params, lookup) -> new SortLeaf(code, params, score);
            } else if (context == AggregationScript.CONTEXT) {
                factory = (AggFactory) (params, lookup) -> new AggLeaf(code, params, score);
            } else {
                factory = (TermsSetFactory) (params, lookup) -> new TermsSetLeaf(code, params, false);
            }
            return context.factoryClazz.cast(factory);
        }

        @Override
        public Set<ScriptContext<?>> getSupportedContexts() {
            return Set.of(
                ScoreScript.CONTEXT,
                FilterScript.CONTEXT,
                NumberSortScript.CONTEXT,
                AggregationScript.CONTEXT,
                TermsSetQueryScript.CONTEXT
            );
        }
    }

    private record ScoreFactory(String code, boolean score) implements ScoreScript.Factory {
        @Override
        public ScoreScript.LeafFactory newFactory(Map<String, Object> params, SearchLookup lookup, IndexSearcher searcher) {
            return new ScoreLeaf(code, params, score);
        }
    }

    private interface FilterFactory extends FilterScript.Factory {}

    private interface SortFactory extends NumberSortScript.Factory {}

    private interface AggFactory extends AggregationScript.Factory {}

    private interface TermsSetFactory extends TermsSetQueryScript.Factory {}

    /** {@link #check} by context and source: "" when it passes. */
    private static final ConcurrentHashMap<String, String> CHECKED = new ConcurrentHashMap<>();

    private static final int CACHE_LIMIT = 1024;

    /** Whether the native engine runs {@code source} in {@code context}: null, or why not. */
    public static String check(String source, byte context) {
        String key = context + ":" + source;
        String hit = CHECKED.get(key);
        if (hit == null) {
            ByteArrayOutputStream spec = new ByteArrayOutputStream();
            writeSpec(spec, context, source, new byte[] { 12, 0, 0, 0, 0 }, List.of(), List.of(), new byte[0], "", 0, 0);
            byte[][] out = new byte[1][];
            int rc = NativeBridge.painlessCheck(spec.toByteArray(), out);
            hit = rc != NativeBridge.OK ? "check failed: " + NativeBridge.lastError()
                : out[0] == null ? "" : new String(out[0], StandardCharsets.UTF_8);
            if (CHECKED.size() >= CACHE_LIMIT) {
                CHECKED.clear();
            }
            CHECKED.put(key, hit);
        }
        return hit.isEmpty() ? null : hit;
    }

    /**
     * A JavaScript script's node fields as {@code jvm_painless_score.rs} reads them: the context
     * with {@link #JS_SCRIPT}, the source where Painless's IR goes, the params, index, shard, run
     * number and fields, then the limits.
     */
    public static void writeSpec(
        ByteArrayOutputStream out,
        byte context,
        String source,
        byte[] params,
        List<String> names,
        List<String> reads,
        byte[] kinds,
        String index,
        int shard,
        int run
    ) {
        out.write(context | JS_SCRIPT);
        writeBytes(out, source.getBytes(StandardCharsets.UTF_8));
        writeBytes(out, params);
        writeBytes(out, index.getBytes(StandardCharsets.UTF_8));
        writeInt(out, shard);
        writeInt(out, run);
        writeInt(out, names.size());
        for (int i = 0; i < names.size(); i++) {
            writeBytes(out, names.get(i).getBytes(StandardCharsets.UTF_8));
            writeBytes(out, reads.get(i).getBytes(StandardCharsets.UTF_8));
            out.write(kinds[i]);
        }
        writeInt(out, timeoutMillis);
        writeInt(out, heapMegabytes);
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }

    private static void writeBytes(ByteArrayOutputStream out, byte[] b) {
        writeInt(out, b.length);
        out.write(b, 0, b.length);
    }

    /** Most names {@link #names} takes from one script. */
    private static final int MAX_NAMES = 256;

    /**
     * The names {@code source} may read a field by: its string literals (a template literal's
     * when it has no substitutions) and identifiers, in order, at most {@link #MAX_NAMES}. A
     * script reads a field only by a name its source spells out; a name built at run time reads
     * as a field the mapping lacks.
     */
    public static Set<String> names(String source) {
        Set<String> out = new LinkedHashSet<>();
        int n = source.length();
        int i = 0;
        while (i < n && out.size() < MAX_NAMES) {
            char c = source.charAt(i);
            if (c == '/' && i + 1 < n && source.charAt(i + 1) == '/') {
                while (i < n && source.charAt(i) != '\n') {
                    i++;
                }
            } else if (c == '/' && i + 1 < n && source.charAt(i + 1) == '*') {
                int end = source.indexOf("*/", i + 2);
                i = end < 0 ? n : end + 2;
            } else if (c == '\'' || c == '"' || c == '`') {
                StringBuilder s = new StringBuilder();
                boolean plain = true;
                i++;
                while (i < n && source.charAt(i) != c) {
                    char d = source.charAt(i);
                    if (d == '\\' && i + 1 < n) {
                        char e = source.charAt(i + 1);
                        s.append(switch (e) {
                            case 'n' -> '\n';
                            case 't' -> '\t';
                            case 'r' -> '\r';
                            default -> e;
                        });
                        i += 2;
                        continue;
                    }
                    if (c == '`' && d == '$' && i + 1 < n && source.charAt(i + 1) == '{') {
                        plain = false;
                    }
                    s.append(d);
                    i++;
                }
                i++;
                if (plain) {
                    out.add(s.toString());
                }
            } else if (Character.isJavaIdentifierStart(c)) {
                int start = i;
                while (i < n && Character.isJavaIdentifierPart(source.charAt(i))) {
                    i++;
                }
                out.add(source.substring(start, i));
            } else {
                i++;
            }
        }
        return out;
    }

    /** The names of {@link #names} the mapping has, in order. */
    public static List<String> fields(String source, java.util.function.Function<String, ?> mapped) {
        List<String> out = new ArrayList<>();
        for (String name : names(source)) {
            if (mapped.apply(name) != null) {
                out.add(name);
            }
        }
        return out;
    }
}
