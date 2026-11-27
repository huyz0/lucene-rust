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
import java.net.URLConnection;
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
    /**
     * Where DJL's Maven jars are downloaded from, in order: Google's mirror of Maven Central first
     * (it does not throttle shared address ranges), then Central. A repository that answers with an
     * error, such as HTTP 429, or cannot be reached passes the download to the next one.
     */
    static final List<String> MAVEN_REPOSITORIES = List.of(
        "https://maven-central.storage-download.googleapis.com/maven2/ai/djl/pytorch/",
        "https://repo1.maven.org/maven2/ai/djl/pytorch/"
    );
    /**
     * SHA-256 of DJL's Maven jars of the default flavor. These flavors download from Maven Central,
     * checked against the pin, before DJL's own site; other flavors come from DJL's site unchecked,
     * as DJL itself downloads them.
     */
    private static final Map<String, String> MAVEN_SHA256 = Map.of(
        "cpu-precxx11-linux-x86_64",
        "5503bb5addcdf54498f53520a410bb53c15b781531c14b15b0c49bc8520db753",
        "cpu-precxx11-linux-aarch64",
        "3722754bb5b0ae41f4fdfad00460af922f1eb3dbaea250b17152bf3711a9737b"
    );

    /** How long a download may wait to connect, and between bytes. */
    private static final int TIMEOUT_MILLIS = 60_000;

    private LibTorch() {}

    private static InputStream open(String url) throws IOException {
        URLConnection connection = URI.create(url).toURL().openConnection();
        connection.setConnectTimeout(TIMEOUT_MILLIS);
        connection.setReadTimeout(TIMEOUT_MILLIS);
        return connection.getInputStream();
    }

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
            String sha = MAVEN_SHA256.get(flavor + "-" + classifier);
            if (sha == null) {
                fromDjl(flavor, classifier, tmp);
            } else {
                try {
                    fromMaven(flavor, classifier, sha, tmp);
                } catch (IOException maven) {
                    LOGGER.warn("accelerated-search: cannot download libtorch from Maven, trying {}: {}", DJL_SITE, maven.toString());
                    IOUtils.rm(tmp);
                    Files.createDirectories(tmp);
                    fromDjl(flavor, classifier, tmp);
                }
            }
            if (Files.isRegularFile(tmp.resolve(NATIVE_LIB)) == false) {
                throw new IOException("the libtorch download for " + flavor + "/" + classifier + " has no " + NATIVE_LIB);
            }
            try {
                Files.move(tmp, dir, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException raced) {
                // Another deploy on this node finished the same download first, or an interrupted
                // download left the directory without the library, which this one replaces.
                if (Files.isRegularFile(dir.resolve(NATIVE_LIB)) == false) {
                    IOUtils.rm(dir);
                    Files.move(tmp, dir, StandardCopyOption.ATOMIC_MOVE);
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
        try (InputStream in = open(DJL_SITE + "/files.txt")) {
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
            LOGGER.info("accelerated-search: downloading {}/{}", DJL_SITE, line);
            try (InputStream in = new GZIPInputStream(open(DJL_SITE + "/" + line))) {
                Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
            }
        }
        if (found == false) {
            throw new IOException("No PyTorch native library matches " + flavor + "/" + classifier);
        }
    }

    private static void fromMaven(String flavor, String classifier, String sha256, Path into) throws IOException {
        fromMaven(MAVEN_REPOSITORIES, flavor, classifier, sha256, into);
    }

    /**
     * DJL's Maven jar of the flavor, from the first repository that serves a copy matching the
     * pinned SHA-256, unpacked. A wrong checksum counts as a failed repository, so a damaged mirror
     * falls through to Central.
     */
    static void fromMaven(List<String> repositories, String flavor, String classifier, String sha256, Path into)
        throws IOException {
        String artifact = "pytorch-native-" + flavor;
        Path jar = into.resolve("libtorch.jar");
        IOException last = new IOException("no Maven repository configured");
        boolean downloaded = false;
        for (String repository : repositories) {
            String url = repository + artifact + "/" + VERSION + "/" + artifact + "-" + VERSION + "-" + classifier + ".jar";
            MessageDigest sha;
            try {
                sha = MessageDigest.getInstance("SHA-256");
            } catch (NoSuchAlgorithmException e) {
                throw new IllegalStateException(e);
            }
            LOGGER.info("accelerated-search: downloading {}", url);
            try (InputStream in = new DigestInputStream(open(url), sha)) {
                Files.copy(in, jar, StandardCopyOption.REPLACE_EXISTING);
                if (HexFormat.of().formatHex(sha.digest()).equals(sha256) == false) {
                    throw new IOException("checksum mismatch for " + url);
                }
                downloaded = true;
                break;
            } catch (IOException e) {
                LOGGER.warn("accelerated-search: {} failed, trying the next repository: {}", url, e.toString());
                Files.deleteIfExists(jar);
                last = e;
            }
        }
        if (downloaded == false) {
            throw last;
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
