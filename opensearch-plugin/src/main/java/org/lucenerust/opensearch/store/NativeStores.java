/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.nio.file.Path;
import java.util.Set;
import java.util.function.Predicate;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;

/**
 * Which store directories the native engine may open by path. The engine reads and writes the
 * shard's files itself, so a directory that changes the bytes on their way to disk -- an encrypting
 * or compressing store type from another plugin -- must not be bypassed: the engine would write
 * plaintext beside it, or fail to read it. Only directories whose bytes on disk are what Lucene
 * wrote, and {@link RustEncryptedDirectory} (whose format the engine reads and writes too), pass.
 */
public final class NativeStores {
    /** Store types ({@code index.store.type}) the native engine can serve. */
    public static final Set<String> STORE_TYPES = Set.of("", "fs", "hybridfs", "mmapfs", "niofs", "simplefs", RustEncryptedDirectoryFactory.TYPE);

    private NativeStores() {}

    /**
     * Whether {@code c} is Lucene's or OpenSearch core's own: loaded by the class loader that loaded
     * Lucene's {@link Directory} or OpenSearch's {@code Store}. Those pass bytes through unchanged
     * (OpenSearch's {@code Store$StoreDirectory}, {@code ByteSizeCachingDirectory}, {@code
     * BucketedCompositeDirectory}, the hybrid FS directory). A plugin's classes have the plugin's own
     * loader, and a plugin's directory may change the bytes (encryption, compression), so it is refused.
     */
    static boolean core(Class<?> c) {
        ClassLoader l = c.getClassLoader();
        return l == Directory.class.getClassLoader() || l == org.opensearch.index.store.Store.class.getClassLoader();
    }

    /** Why the native engine must not open {@code dir} by path, or {@code null} when it may. */
    public static String refusal(Directory dir) {
        return refusal(dir, NativeStores::core);
    }

    static String refusal(Directory dir, Predicate<Class<?>> trusted) {
        Directory d = dir;
        boolean encrypted = false;
        while (d instanceof FilterDirectory f) {
            if (f instanceof RustEncryptedDirectory) {
                if (encrypted) {
                    return "the store is encrypted twice";
                }
                encrypted = true;
            } else if (trusted.test(f.getClass()) == false) {
                return "the store directory is wrapped by [" + f.getClass().getName() + "], which the accelerated engine cannot read through";
            }
            d = f.getDelegate();
        }
        if (d instanceof FSDirectory == false) {
            return "the accelerated engine needs a filesystem directory, and the store's is a [" + d.getClass().getName() + "]";
        }
        if (trusted.test(d.getClass()) == false) {
            return "the store directory is a [" + d.getClass().getName() + "], which the accelerated engine cannot read through";
        }
        return null;
    }

    /** The path the native engine opens for {@code dir}, after {@link #refusal} allowed it. */
    public static Path path(Directory dir) {
        String why = refusal(dir);
        if (why != null) {
            throw new IllegalArgumentException(why);
        }
        return ((FSDirectory) FilterDirectory.unwrap(dir)).getDirectory();
    }
}
