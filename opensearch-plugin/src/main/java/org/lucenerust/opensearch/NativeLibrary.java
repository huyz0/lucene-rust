/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Finds and loads {@code liblucene_ffi.so}, then checks that it speaks the ABI this jar was built
 * for.
 *
 * <p>The plugin zip carries one library per platform under {@code native/<os>-<arch>/}. The
 * library is loaded from the installed plugin directory rather than extracted from the jar: the
 * plugin directory is already on disk, so extraction would only add a temp file to clean up. The
 * system property {@value #PATH_PROPERTY} overrides the location, for tests.
 *
 * <p>Every failure is a {@link IllegalStateException} naming what was looked for and why it did not
 * work, so that a node refusing to start says so in one line rather than as a {@link
 * UnsatisfiedLinkError} three calls later.
 */
public final class NativeLibrary {
    public static final String PATH_PROPERTY = "lucene_rust.library.path";
    public static final String LIBRARY_FILE = "liblucene_ffi.so";

    private static volatile String loaded;

    private NativeLibrary() {}

    /** {@code linux-x86_64} or {@code linux-aarch64}; anything else has no library. */
    public static String platform() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        String osPart = os.startsWith("linux") ? "linux" : os.replace(' ', '_');
        String archPart = switch (arch) {
            case "amd64", "x86_64" -> "x86_64";
            case "aarch64", "arm64" -> "aarch64";
            default -> arch;
        };
        return osPart + "-" + archPart;
    }

    /** Loads the library once per JVM; later calls return the path it was loaded from. */
    public static synchronized String load(Path pluginDir) {
        if (loaded != null) {
            return loaded;
        }
        String override = System.getProperty(PATH_PROPERTY);
        Path lib = override != null
            ? Path.of(override)
            : pluginDir.resolve("native").resolve(platform()).resolve(LIBRARY_FILE);
        if (Files.isRegularFile(lib) == false) {
            throw new IllegalStateException(
                "lucene-rust: no native library for platform ["
                    + platform()
                    + "] at ["
                    + lib
                    + "]; this plugin build supports linux-x86_64 and linux-aarch64"
            );
        }
        try {
            System.load(lib.toAbsolutePath().toString());
        } catch (UnsatisfiedLinkError e) {
            throw new IllegalStateException("lucene-rust: failed to load [" + lib + "]: " + e.getMessage(), e);
        }
        int abi;
        try {
            abi = NativeBridge.abiVersion();
        } catch (UnsatisfiedLinkError e) {
            throw new IllegalStateException("lucene-rust: [" + lib + "] has no JNI entry points; is it a lucene-ffi build?", e);
        }
        if (abi != NativeBridge.EXPECTED_ABI_VERSION) {
            throw new IllegalStateException(
                "lucene-rust: ["
                    + lib
                    + "] speaks ABI "
                    + abi
                    + " but this plugin expects "
                    + NativeBridge.EXPECTED_ABI_VERSION
                    + "; the plugin jar and native library come from different builds"
            );
        }
        loaded = lib.toAbsolutePath().toString();
        return loaded;
    }
}
