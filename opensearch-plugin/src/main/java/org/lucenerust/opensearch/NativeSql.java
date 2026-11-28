/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.lucenerust.opensearch.NativeBridge.*;
import static org.lucenerust.opensearch.NativeCall.*;

/** The SQL downcalls ({@code jvm_sql.rs}); see {@link NativeBridge}. */
public final class NativeSql {
    private NativeSql() {}

    private static final class H {
        static final MethodHandle TABLE = bind("ffi_jvm_sql_table", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS));
        static final MethodHandle QUERY = bind(
            "ffi_jvm_sql_query",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle CONFIGURE = bind("ffi_jvm_sql_configure", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT));
        static final MethodHandle STATS = bind("ffi_jvm_sql_stats", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    }

    /** A statement the native side refused or could not run; the message is its reason. */
    public static final class Failure extends RuntimeException {
        private final int status;

        Failure(int status, String message) {
            super(message);
            this.status = status;
        }

        /** The native status code ({@link NativeBridge#INVALID_ARGUMENT} and the like). */
        public int status() {
            return status;
        }
    }

    private static Failure failure(int rc) {
        return new Failure(rc, NativeBridge.lastError());
    }

    /** Sets what the SQL engines are built with; zero keeps a default. */
    public static void configure(long maxRows, long memoryLimit, int workerThreads) {
        try {
            int rc = (int) H.CONFIGURE.invokeExact(maxRows, memoryLimit, workerThreads);
            if (rc != OK) {
                throw failure(rc);
            }
        } catch (Failure e) {
            throw e;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** The name of the table {@code sql} reads. */
    public static String table(String sql) {
        byte[] utf8 = sql.getBytes(StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.TABLE.invokeExact(s.bytes(utf8), (long) utf8.length, ptrOf(result), lenOf(result));
            if (rc != OK) {
                throw failure(rc);
            }
            byte[] name = take(result);
            if (name == null) {
                throw new Failure(BUFFER_TOO_SMALL, "the table name is larger than a Java array holds");
            }
            return new String(name, StandardCharsets.UTF_8);
        } catch (Failure e) {
            throw e;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs {@code sql} over reader {@code handle} read as the table {@code table} describes
     * ({@link SqlTables}): the result as the SQL plugin's JSON response.
     */
    public static byte[] query(long handle, byte[] table, String sql) {
        byte[] utf8 = sql.getBytes(StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.QUERY.invokeExact(
                handle,
                s.bytes(table),
                (long) table.length,
                s.bytes(utf8),
                (long) utf8.length,
                ptrOf(result),
                lenOf(result)
            );
            if (rc != OK) {
                throw failure(rc);
            }
            byte[] json = take(result);
            if (json == null) {
                throw new Failure(BUFFER_TOO_SMALL, "the result is larger than a Java array holds");
            }
            return json;
        } catch (Failure e) {
            throw e;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Statements the router and DataFusion answered since the library loaded: {@code {router, standard}}. */
    public static long[] stats() {
        try (Scratch s = Scratch.open()) {
            MemorySegment out = s.alloc(2L * Long.BYTES);
            int rc = (int) H.STATS.invokeExact(out);
            if (rc != OK) {
                throw failure(rc);
            }
            return new long[] { out.get(JAVA_LONG, 0), out.get(JAVA_LONG, Long.BYTES) };
        } catch (Failure e) {
            throw e;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
