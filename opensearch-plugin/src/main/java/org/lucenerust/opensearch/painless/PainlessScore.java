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

import java.lang.reflect.Field;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.opensearch.index.mapper.BooleanFieldMapper;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.painless.CompilerSettings;
import org.opensearch.painless.PainlessScriptEngine;
import org.opensearch.painless.lookup.PainlessLookup;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptContext;
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
    }

    /**
     * {@code source} compiled for the score context: a {@link PainlessIr.Compiled}, or a string
     * saying why the native runtime cannot have it.
     */
    public static Object score(String source) {
        return compile(ScoreScript.CONTEXT, ScoreScript.class, source);
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
