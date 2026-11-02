/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.io.IOException;
import java.io.InputStream;
import java.security.GeneralSecurityException;
import java.security.Provider;
import java.security.SecureRandom;
import java.util.Properties;

import javax.crypto.AEADBadTagException;
import javax.crypto.Cipher;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

import org.lucenerust.opensearch.NativeBridge;

/**
 * The JVM side's AES-256-GCM and random numbers for the encrypted store.
 *
 * <ul>
 *   <li>The default build uses the JDK's own providers ({@code Cipher.getInstance} with no provider
 *       named, which is SunJCE).
 *   <li>The FIPS build (the plugin zip built with {@code -Pfips}, whose {@code lucene-rust-build.properties}
 *       says {@code fips=true}) uses the Amazon Corretto Crypto Provider's FIPS build, which is AWS-LC's
 *       FIPS 140-3 module: the same module the native library's {@code fips} build links. SunJCE is not
 *       FIPS validated, so it is never used there. Startup checks that ACCP loaded, is the FIPS build,
 *       passed its self tests, and that the native library is in FIPS mode too; otherwise the
 *       encrypted store refuses to open, rather than encrypt with an unvalidated module.
 * </ul>
 *
 * Sealing lets the provider draw each GCM nonce ({@code Cipher.init} with no parameters), the
 * FIPS-approved way to use GCM; a provider that will not falls back to a nonce from this class's
 * {@link SecureRandom} (the provider's DRBG in the FIPS build).
 */
public final class StoreCrypto {
    static final String ACCP_CLASS = "com.amazon.corretto.crypto.provider.AmazonCorrettoCryptoProvider";
    private static final String TRANSFORMATION = "AES/GCM/NoPadding";
    private static final int TAG_BITS = Lrse.TAG_LEN * 8;

    private static volatile StoreCrypto instance;

    private final boolean fips;
    private final Provider provider;
    private final SecureRandom random;
    private final String description;

    private StoreCrypto(boolean fips, Provider provider, SecureRandom random, String description) {
        this.fips = fips;
        this.provider = provider;
        this.random = random;
        this.description = description;
    }

    /** The JVM's instance, chosen once from the build properties. */
    public static StoreCrypto get() {
        StoreCrypto c = instance;
        if (c == null) {
            synchronized (StoreCrypto.class) {
                c = instance;
                if (c == null) {
                    c = instance = create(buildIsFips());
                }
            }
        }
        return c;
    }

    /** Whether this plugin was built as the FIPS variant ({@code lucene-rust-build.properties}). */
    public static boolean buildIsFips() {
        try (InputStream in = StoreCrypto.class.getResourceAsStream("/lucene-rust-build.properties")) {
            if (in == null) {
                return false;
            }
            Properties p = new Properties();
            p.load(in);
            return Boolean.parseBoolean(p.getProperty("fips", "false"));
        } catch (IOException e) {
            throw new IllegalStateException("lucene-rust: cannot read lucene-rust-build.properties", e);
        }
    }

    static StoreCrypto create(boolean fips) {
        if (fips == false) {
            return new StoreCrypto(false, null, new SecureRandom(), "JDK default providers");
        }
        Provider accp = loadAccp();
        if (NativeBridge.storeIsFips() == false) {
            throw new IllegalStateException(
                "lucene-rust: this is the FIPS build, but the native library's encryption is not in FIPS mode "
                    + "(build it with scripts/build-fips.sh, or the plugin with -Pfips)"
            );
        }
        try {
            SecureRandom random = SecureRandom.getInstance("DEFAULT", accp);
            return new StoreCrypto(true, accp, random, accp.getName() + " " + accp.getVersionStr() + " (FIPS)");
        } catch (GeneralSecurityException e) {
            throw new IllegalStateException("lucene-rust: the Corretto provider offers no DRBG", e);
        }
    }

    /**
     * The Amazon Corretto Crypto Provider, loaded from the plugin's class path, checked to be the FIPS
     * build and healthy.
     */
    private static Provider loadAccp() {
        try {
            Class<?> c = Class.forName(ACCP_CLASS, true, StoreCrypto.class.getClassLoader());
            Provider p = (Provider) c.getField("INSTANCE").get(null);
            c.getMethod("assertHealthy").invoke(p);
            boolean isFips = (Boolean) c.getMethod("isFips").invoke(p);
            if (isFips == false) {
                throw new IllegalStateException(
                    "lucene-rust: the FIPS build needs the Corretto provider's FIPS build (AmazonCorrettoCryptoProvider-FIPS), and "
                        + p.getName()
                        + " "
                        + p.getVersionStr()
                        + " is not"
                );
            }
            return p;
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException(
                "lucene-rust: the FIPS build needs the Amazon Corretto Crypto Provider (FIPS) on the plugin's class path",
                e
            );
        } catch (ReflectiveOperationException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new IllegalStateException("lucene-rust: the Corretto provider is not usable: " + cause, cause);
        }
    }

    public boolean fips() {
        return fips;
    }

    /** Which provider seals and opens, for logs and the stats endpoint. */
    public String description() {
        return description;
    }

    public SecureRandom random() {
        return random;
    }

    /** A fresh AES-GCM cipher (not thread safe: one per stream). */
    public Cipher newCipher() throws IOException {
        try {
            return provider == null ? Cipher.getInstance(TRANSFORMATION) : Cipher.getInstance(TRANSFORMATION, provider);
        } catch (GeneralSecurityException e) {
            throw new IOException("lucene-rust: no " + TRANSFORMATION + " cipher from " + description, e);
        }
    }

    /**
     * Encrypts {@code in[off..off+len)} under {@code key} with associated data {@code aad}, under a
     * nonce the provider draws. Writes the nonce to {@code nonce} and returns ciphertext then tag.
     */
    public byte[] seal(Cipher cipher, SecretKey key, byte[] aad, byte[] in, int off, int len, byte[] nonce) throws IOException {
        try {
            byte[] iv;
            try {
                cipher.init(Cipher.ENCRYPT_MODE, key, random);
                iv = cipher.getIV();
            } catch (GeneralSecurityException | IllegalStateException e) {
                iv = null;
            }
            if (iv == null || iv.length != Lrse.NONCE_LEN) {
                iv = new byte[Lrse.NONCE_LEN];
                random.nextBytes(iv);
                cipher.init(Cipher.ENCRYPT_MODE, key, new GCMParameterSpec(TAG_BITS, iv), random);
            }
            System.arraycopy(iv, 0, nonce, 0, Lrse.NONCE_LEN);
            cipher.updateAAD(aad);
            return cipher.doFinal(in, off, len);
        } catch (GeneralSecurityException e) {
            throw new IOException("lucene-rust: sealing a chunk failed", e);
        }
    }

    /**
     * Decrypts {@code sealed[off..off+len)} (ciphertext then tag) into {@code out} at {@code outOff},
     * returning the plaintext's length. A failed authentication is an {@link AEADBadTagException}.
     */
    public int open(Cipher cipher, SecretKey key, byte[] nonce, byte[] aad, byte[] sealed, int off, int len, byte[] out, int outOff)
        throws GeneralSecurityException {
        cipher.init(Cipher.DECRYPT_MODE, key, new GCMParameterSpec(TAG_BITS, nonce));
        cipher.updateAAD(aad);
        return cipher.doFinal(sealed, off, len, out, outOff);
    }
}
