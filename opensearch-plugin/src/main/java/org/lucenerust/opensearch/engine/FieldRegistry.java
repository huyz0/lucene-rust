/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The shard's field numbers: {@code FieldInfos.FieldNumbers}. A field is registered with the Rust
 * writer the first time a document uses it, with that document's schema; every later document must
 * declare the same schema, which is Lucene's rule too.
 */
final class FieldRegistry {
    /** Registers a new field's schema with the writer and returns its global number. */
    interface Registrar {
        int register(FieldSchema schema) throws IOException;
    }

    record Entry(FieldSchema schema, int number) {}

    private final ConcurrentHashMap<String, Entry> fields = new ConcurrentHashMap<>();
    private final Registrar registrar;

    FieldRegistry(Registrar registrar) {
        this.registrar = registrar;
    }

    /** The registered entry for a document's schema of a field, registering it on first use. */
    Entry resolve(FieldSchema docSchema) throws IOException {
        Entry e = fields.get(docSchema.name);
        if (e == null) {
            synchronized (this) {
                e = fields.get(docSchema.name);
                if (e == null) {
                    docSchema.checkSupported();
                    e = new Entry(docSchema, registrar.register(docSchema));
                    fields.put(docSchema.name, e);
                    return e;
                }
            }
        }
        docSchema.assertSameAs(e.schema);
        return e;
    }

    int size() {
        return fields.size();
    }
}
