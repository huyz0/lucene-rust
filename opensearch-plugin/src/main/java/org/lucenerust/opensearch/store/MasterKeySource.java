/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.io.IOException;

import org.opensearch.common.crypto.DataKeyPair;
import org.opensearch.common.crypto.MasterKeyProvider;

/**
 * Data keys from an OpenSearch {@link MasterKeyProvider}: a {@code CryptoKeyProviderPlugin} such as
 * {@code crypto-kms} (AWS KMS), configured by the index's {@code index.store.encrypted.*}
 * settings. The KMS draws each data key ({@code generateDataPair}) and unwraps it ({@code
 * decryptKey}); the key-encryption key never leaves it. For a FIPS deployment the provider must use
 * the KMS's FIPS endpoints.
 */
public final class MasterKeySource implements KeySource {
    private final MasterKeyProvider provider;

    public MasterKeySource(MasterKeyProvider provider) {
        this.provider = provider;
    }

    @Override
    public DataKey generate() throws IOException {
        try {
            DataKeyPair pair = provider.generateDataPair();
            return new DataKey(provider.getKeyId(), pair.getRawKey(), pair.getEncryptedKey());
        } catch (RuntimeException e) {
            throw new IOException("the key provider could not generate a data key: " + e.getMessage(), e);
        }
    }

    @Override
    public byte[] unwrap(String keyId, byte[] wrapped) throws IOException {
        try {
            return provider.decryptKey(wrapped);
        } catch (RuntimeException e) {
            throw new IOException("the key provider could not decrypt a data key wrapped under [" + keyId + "]: " + e.getMessage(), e);
        }
    }

    @Override
    public void close() throws IOException {
        provider.close();
    }
}
