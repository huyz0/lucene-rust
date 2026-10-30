/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.AnalyzerWrapper;
import org.apache.lucene.analysis.hunspell.Dictionary;
import org.opensearch.core.util.FileSystemUtils;
import org.opensearch.env.Environment;
import org.opensearch.index.analysis.AnalyzerComponentsProvider;
import org.opensearch.index.analysis.HunspellTokenFilterFactory;
import org.opensearch.index.analysis.NamedAnalyzer;
import org.opensearch.index.analysis.TokenFilterFactory;
import org.opensearch.indices.analysis.HunspellService;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The files behind a hunspell {@link Dictionary} an OpenSearch analyzer uses. A {@link Dictionary}
 * keeps none of its sources, so they are found the way {@link HunspellService} loaded them: the
 * {@link HunspellTokenFilterFactory} holding the dictionary names its locale (and {@code ref_path}),
 * and the service's cache must still hold that very instance under that key, so the files on disk
 * are the ones it was parsed from (a plugin's own dictionary, or one reloaded since, has no such
 * entry and stays in Java).
 */
final class HunspellSources {
    private HunspellSources() {}

    /** {@code d}'s affix and dictionary files, in the order the service read them, or null. */
    static AnalyzerSpec.HunspellFiles find(Analyzer analyzer, String field, Dictionary d) {
        try {
            for (TokenFilterFactory f : factories(analyzer, field)) {
                if (f instanceof HunspellTokenFilterFactory h && field(h, "dictionary") == d) {
                    return files(h, d);
                }
            }
        } catch (IOException | ReflectiveOperationException | RuntimeException e) {
            return null;
        }
        return null;
    }

    /** The token filter factories of the OpenSearch analyzer {@code field} is analyzed with. */
    private static List<TokenFilterFactory> factories(Analyzer a, String field) throws ReflectiveOperationException {
        for (int depth = 0; depth < 8 && a != null; depth++) {
            if (a instanceof AnalyzerComponentsProvider p) {
                return Arrays.asList(p.getComponents().getTokenFilters());
            }
            if (a instanceof NamedAnalyzer n) {
                a = n.analyzer();
            } else if (a instanceof AnalyzerWrapper w) {
                Method m = wrapped(w.getClass());
                if (m == null) {
                    break;
                }
                a = (Analyzer) m.invoke(w, field);
            } else {
                break;
            }
        }
        return List.of();
    }

    private static Method wrapped(Class<?> c) {
        for (Class<?> k = c; k != null && k != Analyzer.class; k = k.getSuperclass()) {
            try {
                Method m = k.getDeclaredMethod("getWrappedAnalyzer", String.class);
                m.setAccessible(true);
                return m;
            } catch (NoSuchMethodException e) {
                // keep looking up
            }
        }
        return null;
    }

    private static AnalyzerSpec.HunspellFiles files(HunspellTokenFilterFactory h, Dictionary d) throws IOException,
        ReflectiveOperationException {
        String refPath = (String) field(h, "refPath");
        String locale = (String) field(h, "locale");
        Object service = field(h, "hunspellService");
        if (locale == null || service == null) {
            return null;
        }
        Map<?, ?> cache = (Map<?, ?>) field(service, "dictionaries");
        Path dir;
        if (refPath == null) {
            Map<?, ?> known = (Map<?, ?>) field(service, "knownDictionaries");
            if (known.get(locale) != null || cache.get(locale) != d) {
                return null;
            }
            dir = ((Path) field(service, "hunspellDir")).resolve(locale);
        } else {
            if (cache.get(HunspellService.buildRefPathCacheKey(refPath, locale)) != d) {
                return null;
            }
            Environment env = (Environment) field(service, "env");
            dir = env.configDir().resolve(refPath).resolve("hunspell").resolve(locale);
        }
        Path[] affixes = FileSystemUtils.files(dir, "*.aff");
        if (affixes.length != 1) {
            return null;
        }
        List<byte[]> dictionaries = new ArrayList<>();
        for (Path dic : FileSystemUtils.files(dir, "*.dic")) {
            dictionaries.add(Files.readAllBytes(dic));
        }
        return new AnalyzerSpec.HunspellFiles(Files.readAllBytes(affixes[0]), dictionaries);
    }

    private static Object field(Object o, String name) throws ReflectiveOperationException {
        Field f = Reflect.inHierarchy(o.getClass(), name);
        if (f == null) {
            throw new NoSuchFieldException(o.getClass().getName() + "." + name);
        }
        return f.get(o);
    }
}
