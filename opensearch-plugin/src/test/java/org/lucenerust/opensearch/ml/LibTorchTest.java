/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;

import com.sun.net.httpserver.HttpServer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.junit.After;
import org.junit.Test;

/** The libtorch jar download moves to the next repository when one throttles, is damaged or is down. */
public class LibTorchTest {
    private static final String FLAVOR = "cpu-precxx11";
    private static final String CLASSIFIER = "linux-x86_64";
    private final List<HttpServer> servers = new ArrayList<>();

    @After
    public void stop() {
        servers.forEach(s -> s.stop(0));
    }

    private static byte[] jar() throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ZipOutputStream zip = new ZipOutputStream(bytes)) {
            zip.putNextEntry(new ZipEntry("pytorch/" + FLAVOR + "/" + CLASSIFIER + "/libtorch.so"));
            zip.write("native".getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
        }
        return bytes.toByteArray();
    }

    private static String sha(byte[] bytes) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }

    /** A repository answering every request with the status, or with the body when the status is 200. */
    private String serve(int status, byte[] body, AtomicInteger requests) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            requests.incrementAndGet();
            exchange.sendResponseHeaders(status, status == 200 ? body.length : -1);
            if (status == 200) {
                exchange.getResponseBody().write(body);
            }
            exchange.close();
        });
        server.start();
        servers.add(server);
        return "http://127.0.0.1:" + server.getAddress().getPort() + "/";
    }

    @Test
    public void aThrottledRepositoryPassesTheDownloadToTheNext() throws Exception {
        byte[] jar = jar();
        AtomicInteger throttled = new AtomicInteger();
        AtomicInteger good = new AtomicInteger();
        List<String> repos = List.of(serve(429, null, throttled), serve(200, jar, good));
        Path dir = Files.createTempDirectory("libtorch");
        LibTorch.fromMaven(repos, FLAVOR, CLASSIFIER, sha(jar), dir);
        assertEquals(1, throttled.get());
        assertEquals(1, good.get());
        assertArrayEquals("native".getBytes(StandardCharsets.UTF_8), Files.readAllBytes(dir.resolve("libtorch.so")));
        assertFalse(Files.exists(dir.resolve("libtorch.jar")));
    }

    @Test
    public void aDamagedCopyFailsItsChecksumAndTheNextRepositoryIsUsed() throws Exception {
        byte[] jar = jar();
        List<String> repos = List.of(serve(200, "damaged".getBytes(StandardCharsets.UTF_8), new AtomicInteger()), serve(200, jar, new AtomicInteger()));
        Path dir = Files.createTempDirectory("libtorch");
        LibTorch.fromMaven(repos, FLAVOR, CLASSIFIER, sha(jar), dir);
        assertEquals(1, Files.list(dir).filter(p -> p.getFileName().toString().equals("libtorch.so")).count());
    }

    @Test
    public void anUnreachableRepositoryPassesTheDownloadToTheNext() throws Exception {
        byte[] jar = jar();
        String dead;
        try (ServerSocket closed = new ServerSocket(0, 1, InetAddress.getLoopbackAddress())) {
            dead = "http://127.0.0.1:" + closed.getLocalPort() + "/";
        }
        Path dir = Files.createTempDirectory("libtorch");
        LibTorch.fromMaven(List.of(dead, serve(200, jar, new AtomicInteger())), FLAVOR, CLASSIFIER, sha(jar), dir);
        assertEquals(1, Files.list(dir).filter(p -> p.getFileName().toString().equals("libtorch.so")).count());
    }

    @Test
    public void everyRepositoryFailingIsAnError() throws Exception {
        Path dir = Files.createTempDirectory("libtorch");
        List<String> repos = List.of(serve(429, null, new AtomicInteger()), serve(503, null, new AtomicInteger()));
        assertThrows(IOException.class, () -> LibTorch.fromMaven(repos, FLAVOR, CLASSIFIER, "00", dir));
    }

    @Test
    public void theDefaultsListTheMirrorBeforeCentral() {
        assertEquals(2, LibTorch.MAVEN_REPOSITORIES.size());
        assertEquals(true, LibTorch.MAVEN_REPOSITORIES.get(0).contains("storage-download.googleapis.com"));
        assertEquals(true, LibTorch.MAVEN_REPOSITORIES.get(1).contains("repo1.maven.org"));
    }
}
