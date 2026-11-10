/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.common.util.io.IOUtils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Finds the libtorch TorchScript models run on, as DJL's PyTorch engine does for stock ml-commons:
 * {@code PYTORCH_LIBRARY_PATH} first, then DJL's cache directory under {@code <data>/ml_cache} (so
 * a node that ran stock ml-commons reuses its download), else a download into that directory. Stock
 * asks DJL for libtorch 2.5.1 built with the pre-C++11 string ABI ({@code PYTORCH_PRECXX11}), whose
 * flavor {@code PYTORCH_FLAVOR} overrides.
 */
final class LibTorch {
    private static final Logger LOGGER = LogManager.getLogger(LibTorch.class);
    static final String VERSION = "2.5.1";
    private static final String NATIVE_LIB = "libtorch.so";
    private static final String DJL_SITE = "https://publish.djl.ai/pytorch/" + VERSION;
    private static final String MAVEN = "https://repo1.maven.org/maven2/ai/djl/pytorch/";
    /** SHA-256 of DJL's Maven jars of the default flavor, the download used when DJL's own site fails. */
    private static final Map<String, String> MAVEN_SHA256 = Map.of(
        "cpu-precxx11-linux-x86_64",
        "5503bb5addcdf54498f53520a410bb53c15b781531c14b15b0c49bc8520db753",
        "cpu-precxx11-linux-aarch64",
        "3722754bb5b0ae41f4fdfad00460af922f1eb3dbaea250b17152bf3711a9737b"
    );

    private LibTorch() {}

    /** DJL's {@code Utils.getEnvOrSystemProperty}. */
    private static String setting(String name) {
        String env = System.getenv(name);
        return env != null ? env : System.getProperty(name);
    }

    /** The libtorch directory to load, downloading it into {@code mlCache} when needed. */
    static Path resolve(Path mlCache) throws IOException {
        String override = setting("PYTORCH_LIBRARY_PATH");
        if (override != null) {
            for (String p : override.split(File.pathSeparator)) {
                Path path = Path.of(p);
                if (Files.isRegularFile(path) && NATIVE_LIB.equals(path.getFileName().toString())) {
                    return path.toAbsolutePath().getParent();
                }
                if (Files.isRegularFile(path.resolve(NATIVE_LIB))) {
                    return path.toAbsolutePath();
                }
            }
        }
        String flavor = setting("PYTORCH_FLAVOR");
        if (flavor == null || flavor.isEmpty()) {
            flavor = "cpu-precxx11";
        }
        String classifier = "linux-" + arch();
        Path cache = mlCache.resolve("pytorch");
        Path dir = cache.resolve(VERSION + "-" + flavor + "-" + classifier);
        if (Files.isRegularFile(dir.resolve(NATIVE_LIB))) {
            return dir;
        }
        Files.createDirectories(cache);
        Path tmp = Files.createTempDirectory(cache, "tmp");
        try {
            try {
                fromDjl(flavor, classifier, tmp);
            } catch (IOException djl) {
                String sha = MAVEN_SHA256.get(flavor + "-" + classifier);
                if (sha == null) {
                    throw djl;
                }
                LOGGER.warn("lucene-rust: cannot download libtorch from {}, trying Maven Central: {}", DJL_SITE, djl.toString());
                IOUtils.rm(tmp);
                Files.createDirectories(tmp);
                fromMaven(flavor, classifier, sha, tmp);
            }
            if (Files.isRegularFile(tmp.resolve(NATIVE_LIB)) == false) {
                throw new IOException("the libtorch download for " + flavor + "/" + classifier + " has no " + NATIVE_LIB);
            }
            try {
                Files.move(tmp, dir, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException raced) {
                // Another deploy on this node finished the same download first.
                if (Files.isRegularFile(dir.resolve(NATIVE_LIB)) == false) {
                    throw raced;
                }
            }
            return dir;
        } finally {
            IOUtils.rm(tmp);
        }
    }

    private static String arch() {
        String arch = System.getProperty("os.arch").toLowerCase(Locale.ROOT);
        return switch (arch) {
            case "amd64", "x86_64" -> "x86_64";
            case "aarch64", "arm64" -> "aarch64";
            default -> arch;
        };
    }

    /** DJL's {@code LibUtils.downloadPyTorch}: the site's file index, then each gzipped library of the flavor. */
    private static void fromDjl(String flavor, String classifier, Path into) throws IOException {
        List<String> lines;
        try (InputStream in = URI.create(DJL_SITE + "/files.txt").toURL().openStream()) {
            lines = new String(in.readAllBytes(), StandardCharsets.UTF_8).lines().toList();
        }
        String prefix = flavor + "/" + classifier + "/";
        boolean found = false;
        for (String line : lines) {
            if (line.startsWith(prefix) == false || line.endsWith(".gz") == false) {
                continue;
            }
            found = true;
            String name = URLDecoder.decode(line.substring(line.lastIndexOf('/') + 1, line.length() - 3), StandardCharsets.UTF_8);
            Path target = into.resolve(name).normalize();
            if (target.getParent().equals(into) == false) {
                throw new IOException("unexpected libtorch file name [" + name + "]");
            }
            LOGGER.info("lucene-rust: downloading {}/{}", DJL_SITE, line);
            try (InputStream in = new GZIPInputStream(URI.create(DJL_SITE + "/" + line).toURL().openStream())) {
                Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
            }
        }
        if (found == false) {
            throw new IOException("No PyTorch native library matches " + flavor + "/" + classifier);
        }
    }

    /** DJL's Maven jar of the flavor, checked against its pinned SHA-256, unpacked. */
    private static void fromMaven(String flavor, String classifier, String sha256, Path into) throws IOException {
        String artifact = "pytorch-native-" + flavor;
        String url = MAVEN + artifact + "/" + VERSION + "/" + artifact + "-" + VERSION + "-" + classifier + ".jar";
        Path jar = into.resolve("libtorch.jar");
        MessageDigest sha;
        try {
            sha = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
        LOGGER.info("lucene-rust: downloading {}", url);
        try (InputStream in = new DigestInputStream(URI.create(url).toURL().openStream(), sha)) {
            Files.copy(in, jar);
        }
        if (HexFormat.of().formatHex(sha.digest()).equals(sha256) == false) {
            throw new IOException("checksum mismatch for " + url);
        }
        String prefix = "pytorch/" + flavor + "/" + classifier + "/";
        try (ZipInputStream zip = new ZipInputStream(Files.newInputStream(jar))) {
            for (ZipEntry e = zip.getNextEntry(); e != null; e = zip.getNextEntry()) {
                String name = e.getName();
                if (e.isDirectory() || name.startsWith(prefix) == false || name.indexOf('/', prefix.length()) >= 0) {
                    continue;
                }
                Path target = into.resolve(name.substring(prefix.length())).normalize();
                if (target.getParent().equals(into) == false) {
                    throw new IOException("unexpected libtorch file name [" + name + "]");
                }
                Files.copy(zip, target, StandardCopyOption.REPLACE_EXISTING);
            }
        }
        Files.delete(jar);
    }
}
