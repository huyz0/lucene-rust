/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.LockFactory;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.cluster.metadata.CryptoMetadata;
import org.opensearch.common.crypto.MasterKeyProvider;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.crypto.CryptoHandlerRegistry;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.shard.ShardPath;
import org.opensearch.index.store.FsDirectoryFactory;
import org.opensearch.plugins.CryptoKeyProviderPlugin;
import org.opensearch.plugins.IndexStorePlugin;

/**
 * The {@code encrypted} store type: a shard's files encrypted at rest with per-file data keys
 * from a KMS (BYOK), readable and writable by both Lucene and the Rust engine.
 *
 * <pre>
 * index.store.type: encrypted
 * index.store.encrypted.key_provider_type: aws-kms          # a CryptoKeyProviderPlugin's type
 * index.store.encrypted.key_provider: my-kms                # its name
 * index.store.encrypted.kms.key_arn: arn:aws:kms:...        # what that provider reads
 * index.store.encrypted.kms.encryption_context: tenant=a
 * </pre>
 *
 * The key provider type {@code local_keystore} instead wraps data keys with a key held in the
 * node's keystore ({@code indices.store.encrypted.local_key}), for tests and clusters without a KMS.
 */
public final class RustEncryptedDirectoryFactory implements IndexStorePlugin.DirectoryFactory {
    public static final String TYPE = "encrypted";
    private static final String PREFIX = "index.store." + TYPE + ".";

    public static final Setting<String> KEY_PROVIDER_TYPE = Setting.simpleString(
        PREFIX + "key_provider_type",
        "aws-kms",
        Setting.Property.IndexScope,
        Setting.Property.Final
    );
    public static final Setting<String> KEY_PROVIDER = Setting.simpleString(
        PREFIX + "key_provider",
        Setting.Property.IndexScope,
        Setting.Property.Final
    );
    public static final Setting<String> KMS_KEY_ARN = Setting.simpleString(
        PREFIX + "kms.key_arn",
        Setting.Property.IndexScope,
        Setting.Property.Final
    );
    public static final Setting<String> KMS_ENCRYPTION_CONTEXT = Setting.simpleString(
        PREFIX + "kms.encryption_context",
        Setting.Property.IndexScope,
        Setting.Property.Final
    );
    public static final Setting<ByteSizeValue> CHUNK_SIZE = Setting.byteSizeSetting(
        PREFIX + "chunk_size",
        new ByteSizeValue(Lrse.DEFAULT_CHUNK_SIZE),
        new ByteSizeValue(4, ByteSizeUnit.KB),
        new ByteSizeValue(Lrse.MAX_CHUNK_SIZE),
        Setting.Property.IndexScope,
        Setting.Property.Final
    );
    /** How long a generated data key is reused for new files (one KMS call per period, not per file). */
    public static final Setting<TimeValue> DATA_KEY_TTL = Setting.timeSetting(
        PREFIX + "data_key_ttl",
        TimeValue.timeValueHours(1),
        TimeValue.timeValueSeconds(0),
        Setting.Property.IndexScope,
        Setting.Property.Final
    );
    /** How long an unwrapped data key is cached: a revoked key stops working within this. */
    public static final Setting<TimeValue> KEY_CACHE_TTL = Setting.timeSetting(
        PREFIX + "key_cache_ttl",
        TimeValue.timeValueMinutes(5),
        TimeValue.timeValueSeconds(0),
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    public static List<Setting<?>> settings() {
        return List.of(KEY_PROVIDER_TYPE, KEY_PROVIDER, KMS_KEY_ARN, KMS_ENCRYPTION_CONTEXT, CHUNK_SIZE, DATA_KEY_TTL, KEY_CACHE_TTL);
    }

    private final LocalKeySource.Config local;

    /** {@code local} is the node's local key-encryption key, or {@code null} when it has none. */
    public RustEncryptedDirectoryFactory(LocalKeySource.Config local) {
        this.local = local;
    }

    @Override
    public Directory newDirectory(IndexSettings indexSettings, ShardPath shardPath) throws IOException {
        Path location = shardPath.resolveIndex();
        Files.createDirectories(location);
        return newFSDirectory(location, indexSettings.getValue(FsDirectoryFactory.INDEX_LOCK_FACTOR_SETTING), indexSettings);
    }

    @Override
    public Directory newFSDirectory(Path location, LockFactory lockFactory, IndexSettings indexSettings) throws IOException {
        StoreCrypto crypto = StoreCrypto.get();
        Settings s = indexSettings.getSettings();
        KeySource keys = new CachingKeySource(
            keySource(s, crypto),
            DATA_KEY_TTL.get(s).nanos(),
            KEY_CACHE_TTL.get(s).nanos(),
            4096
        );
        FSDirectory fs = null;
        try {
            fs = new MMapDirectory(location, lockFactory);
            return new RustEncryptedDirectory(fs, keys, crypto, Math.toIntExact(CHUNK_SIZE.get(s).getBytes()));
        } catch (IOException | RuntimeException e) {
            try {
                keys.close();
            } finally {
                if (fs != null) {
                    fs.close();
                }
            }
            throw e;
        }
    }

    private KeySource keySource(Settings s, StoreCrypto crypto) {
        String type = KEY_PROVIDER_TYPE.get(s);
        if (LocalKeySource.TYPE.equals(type)) {
            if (local == null) {
                throw new IllegalArgumentException(
                    "[" + KEY_PROVIDER_TYPE.getKey() + "] is [" + type + "], but the node's keystore has no [" + LocalKeySource.KEY_SETTING + "]"
                );
            }
            return new LocalKeySource(local.keyId(), local.key(), crypto);
        }
        String name = KEY_PROVIDER.get(s);
        if (name.isEmpty()) {
            throw new IllegalArgumentException("[" + KEY_PROVIDER.getKey() + "] is required for the [" + TYPE + "] store");
        }
        CryptoKeyProviderPlugin plugin = CryptoHandlerRegistry.getInstance() == null
            ? null
            : CryptoHandlerRegistry.getInstance().getCryptoKeyProviderPlugin(type);
        if (plugin == null) {
            throw new IllegalArgumentException(
                "no key provider plugin of type [" + type + "] is installed (for AWS KMS, install crypto-kms)"
            );
        }
        MasterKeyProvider provider = plugin.createKeyProvider(new CryptoMetadata(name, type, s.getByPrefix(PREFIX)));
        return new MasterKeySource(provider);
    }
}
