/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.Arrays;

import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;

import org.opensearch.common.settings.SecureSetting;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.settings.SecureString;

/**
 * A key-encryption key held in the node's keystore ({@code lucene_rust.store.local_key}), for tests
 * and for clusters without a KMS: AES-256-GCM key wrapping, the key id as associated data. A real
 * deployment uses a KMS through {@link MasterKeySource}, so the key-encryption key never sits on the
 * node.
 */
public final class LocalKeySource implements KeySource {
    /** The key provider type that selects this source. */
    public static final String TYPE = "lucene_rust_local";
    /** The node keystore entry holding the key-encryption key, base64 of 32 bytes. */
    public static final String KEY_SETTING = "lucene_rust.store.local_key";
    public static final Setting<SecureString> KEY = SecureSetting.secureString(KEY_SETTING, null);
    /** The id written into each file's header for that key; change it when the key changes. */
    public static final Setting<String> KEY_ID = Setting.simpleString(
        "lucene_rust.store.local_key_id",
        "local-v1",
        Setting.Property.NodeScope
    );

    /** The node's local key-encryption key, read at startup (the keystore closes after). */
    public record Config(String keyId, byte[] key) {
        /** The node's configuration, or {@code null} when its keystore has no local key. */
        public static Config fromNode(Settings settings) {
            if (KEY.exists(settings) == false) {
                return null;
            }
            byte[] key;
            try (SecureString b64 = KEY.get(settings)) {
                key = java.util.Base64.getDecoder().decode(b64.toString().trim());
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("[" + KEY_SETTING + "] is not base64", e);
            }
            if (key.length != Lrse.DATA_KEY_LEN) {
                throw new IllegalArgumentException("[" + KEY_SETTING + "] must be 32 bytes (base64), not " + key.length);
            }
            return new Config(KEY_ID.get(settings), key);
        }

        @Override
        public String toString() {
            return "Config[keyId=" + keyId + "]";
        }
    }

    private static final byte[] AAD_PREFIX = "lucene-rust local key wrap v1:".getBytes(StandardCharsets.UTF_8);

    private final String keyId;
    private final SecretKeySpec kek;
    private final StoreCrypto crypto;

    public LocalKeySource(String keyId, byte[] kek, StoreCrypto crypto) {
        if (kek.length != Lrse.DATA_KEY_LEN) {
            throw new IllegalArgumentException("the local key-encryption key must be 32 bytes, not " + kek.length);
        }
        this.keyId = keyId;
        this.kek = new SecretKeySpec(kek, "AES");
        this.crypto = crypto;
    }

    @Override
    public DataKey generate() throws IOException {
        byte[] raw = new byte[Lrse.DATA_KEY_LEN];
        crypto.random().nextBytes(raw);
        byte[] nonce = new byte[Lrse.NONCE_LEN];
        byte[] sealed = crypto.seal(crypto.newCipher(), kek, aad(keyId), raw, 0, raw.length, nonce);
        byte[] wrapped = new byte[nonce.length + sealed.length];
        System.arraycopy(nonce, 0, wrapped, 0, nonce.length);
        System.arraycopy(sealed, 0, wrapped, nonce.length, sealed.length);
        return new DataKey(keyId, raw, wrapped);
    }

    @Override
    public byte[] unwrap(String keyId, byte[] wrapped) throws IOException {
        if (this.keyId.equals(keyId) == false) {
            throw new IOException("unknown key-encryption key [" + keyId + "]; this node's local key is [" + this.keyId + "]");
        }
        if (wrapped.length != Lrse.NONCE_LEN + Lrse.DATA_KEY_LEN + Lrse.TAG_LEN) {
            throw new IOException("a wrapped key of " + wrapped.length + " bytes is not one this key wrapped");
        }
        byte[] nonce = Arrays.copyOf(wrapped, Lrse.NONCE_LEN);
        byte[] raw = new byte[Lrse.DATA_KEY_LEN];
        try {
            Cipher c = crypto.newCipher();
            crypto.open(c, kek, nonce, aad(keyId), wrapped, Lrse.NONCE_LEN, wrapped.length - Lrse.NONCE_LEN, raw, 0);
        } catch (GeneralSecurityException e) {
            throw new IOException("the data key does not unwrap under [" + keyId + "] (wrong or rotated key?)", e);
        }
        return raw;
    }

    private static byte[] aad(String keyId) {
        byte[] id = keyId.getBytes(StandardCharsets.UTF_8);
        byte[] aad = Arrays.copyOf(AAD_PREFIX, AAD_PREFIX.length + id.length);
        System.arraycopy(id, 0, aad, AAD_PREFIX.length, id.length);
        return aad;
    }
}
