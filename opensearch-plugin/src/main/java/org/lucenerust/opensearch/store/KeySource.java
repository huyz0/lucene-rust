/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.io.Closeable;
import java.io.IOException;
import java.util.Arrays;

/**
 * Where the encrypted store's data keys come from: a key-encryption key in a KMS (BYOK), behind
 * OpenSearch's {@code MasterKeyProvider} ({@link MasterKeySource}), or one held locally for tests
 * ({@link LocalKeySource}). Both the JVM's directory and the native engine (through the key upcall,
 * {@link KeyHandles}) use the same source, so either side reads what the other wrote.
 */
public interface KeySource extends Closeable {
    /** A new data key for a file: its raw bytes, and the same wrapped by the key-encryption key {@code keyId}. */
    DataKey generate() throws IOException;

    /** The raw data key {@code wrapped} holds, wrapped under {@code keyId}. A revoked or unknown key fails. */
    byte[] unwrap(String keyId, byte[] wrapped) throws IOException;

    @Override
    default void close() throws IOException {}

    /** A data key, raw and wrapped. */
    record DataKey(String keyId, byte[] raw, byte[] wrapped) {
        public DataKey {
            if (raw.length != Lrse.DATA_KEY_LEN) {
                throw new IllegalArgumentException("a data key is " + Lrse.DATA_KEY_LEN + " bytes, not " + raw.length);
            }
            if (wrapped.length == 0) {
                throw new IllegalArgumentException("an empty wrapped key");
            }
        }

        @Override
        public String toString() {
            return "DataKey[keyId=" + keyId + ", wrapped=" + wrapped.length + " bytes]";
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof DataKey k && keyId.equals(k.keyId) && Arrays.equals(raw, k.raw) && Arrays.equals(wrapped, k.wrapped);
        }

        @Override
        public int hashCode() {
            return keyId.hashCode() * 31 + Arrays.hashCode(wrapped);
        }
    }
}
