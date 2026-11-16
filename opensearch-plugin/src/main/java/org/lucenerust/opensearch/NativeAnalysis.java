/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.lucenerust.opensearch.NativeBridge.*;
import static org.lucenerust.opensearch.NativeCall.*;
import static org.lucenerust.opensearch.NativeSearch.*;
import static org.lucenerust.opensearch.NativeStore.*;
import static org.lucenerust.opensearch.NativeMl.*;
import static org.lucenerust.opensearch.NativeSparse.*;
import static org.lucenerust.opensearch.NativeZstd.*;
import static org.lucenerust.opensearch.NativeLimits.*;
import static org.lucenerust.opensearch.NativeWriter.*;
import static org.lucenerust.opensearch.NativeLive.*;

/** The analysis, highlighting and Painless check downcalls; see {@link NativeBridge}. */
public final class NativeAnalysis {
    private NativeAnalysis() {}

    private static final class H {
        static final MethodHandle THAI_SET_BREAK_UPCALL = bind("ffi_thai_set_break_upcall", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        static final MethodHandle HIGHLIGHT_UNIFIED = bind(
            "ffi_jvm_highlight_unified_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle HIGHLIGHT_PLAIN = bind(
            "ffi_jvm_highlight_plain_alloc",
            FunctionDescriptor.of(
                JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS
            )
        );
        static final MethodHandle HIGHLIGHT_FVH = bind(
            "ffi_jvm_highlight_fvh_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ANALYZE = bind(
            "ffi_jvm_analyze_alloc",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle PAINLESS_CHECK = bind(
            "ffi_jvm_painless_check",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle HIGHLIGHT_SET_THAI_UPCALL = bind("ffi_jvm_highlight_set_thai_upcall", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    }

    /** Registers the JVM's Thai word-break upcall ({@code thai_break.rs}); only the first counts. */
    static int thaiSetBreakUpcall(MemorySegment upcall) {
        try {
            return (int) H.THAI_SET_BREAK_UPCALL.invokeExact(upcall);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * One hit's {@code plain} fragments ({@code jvm_highlight_plain.rs}): {@code values} highlighted
     * as {@code request} asks, and with no fragment, the end of {@code summary}'s {@code
     * noMatchSize} summary ({@code summary} null for none); {@code waitForPos} is the span
     * fragmenter's state the field's previous call returned (-1 at first).
     */
    public static int highlightPlain(
        byte[] request,
        java.util.List<String> values,
        int noMatchSize,
        String summary,
        int waitForPos,
        byte[][] out
    ) {
        if (request == null || values == null) {
            return nullArgument(request == null ? "request" : "values");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            long total = 0;
            for (String v : values) {
                total += v.length();
            }
            MemorySegment chars = s.alloc(total * Character.BYTES);
            MemorySegment lengths = s.alloc((long) values.size() * Integer.BYTES);
            long at = 0;
            for (int i = 0; i < values.size(); i++) {
                String v = values.get(i);
                MemorySegment.copy(v.toCharArray(), 0, chars, java.lang.foreign.ValueLayout.JAVA_CHAR_UNALIGNED, at * Character.BYTES, v.length());
                lengths.setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, i, v.length());
                at += v.length();
            }
            MemorySegment summaryChars = MemorySegment.NULL;
            long summaryLen = -1;
            if (summary != null) {
                summaryChars = s.alloc((long) summary.length() * Character.BYTES);
                MemorySegment.copy(summary.toCharArray(), 0, summaryChars, java.lang.foreign.ValueLayout.JAVA_CHAR_UNALIGNED, 0, summary.length());
                summaryLen = summary.length();
            }
            int rc = (int) H.HIGHLIGHT_PLAIN.invokeExact(
                s.bytes(request),
                (long) request.length,
                chars,
                total,
                lengths,
                (long) values.size(),
                noMatchSize,
                summaryChars,
                summaryLen,
                waitForPos,
                ptrOf(result),
                lenOf(result)
            );
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the fragments");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * One hit's {@code fvh} fragments ({@code jvm_highlight_fvh.rs}): segment {@code segment}'s
     * document {@code doc} of {@code handle}'s reader, whose values are {@code values}, highlighted
     * as the field's {@code request} asks; the result in {@code out[0]}.
     */
    public static int highlightFvh(long handle, byte[] request, int segment, int doc, java.util.List<String> values, byte[][] out) {
        if (request == null || values == null) {
            return nullArgument(request == null ? "request" : "values");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            long total = 0;
            for (String v : values) {
                total += v.length();
            }
            MemorySegment chars = s.alloc(total * Character.BYTES);
            MemorySegment lengths = s.alloc((long) values.size() * Integer.BYTES);
            long at = 0;
            for (int i = 0; i < values.size(); i++) {
                String v = values.get(i);
                MemorySegment.copy(v.toCharArray(), 0, chars, java.lang.foreign.ValueLayout.JAVA_CHAR_UNALIGNED, at * Character.BYTES, v.length());
                lengths.setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, i, v.length());
                at += v.length();
            }
            int rc = (int) H.HIGHLIGHT_FVH.invokeExact(
                handle,
                s.bytes(request),
                (long) request.length,
                segment,
                doc,
                chars,
                total,
                lengths,
                (long) values.size(),
                ptrOf(result),
                lenOf(result)
            );
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the fragments");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * One hit's unified-highlighter snippets ({@code jvm_highlight.rs}): segment {@code segment}'s
     * document {@code doc} of {@code handle}'s reader, whose loaded value is {@code content},
     * highlighted as the field's {@code request} asks; the result in {@code out[0]}.
     */
    public static int highlightUnified(long handle, byte[] request, int segment, int doc, String content, byte[][] out) {
        if (request == null || content == null) {
            return nullArgument(request == null ? "request" : "content");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int n = content.length();
            MemorySegment chars = s.alloc((long) n * Character.BYTES);
            MemorySegment.copy(content.toCharArray(), 0, chars, java.lang.foreign.ValueLayout.JAVA_CHAR_UNALIGNED, 0, n);
            int rc = (int) H.HIGHLIGHT_UNIFIED.invokeExact(
                handle,
                s.bytes(request),
                (long) request.length,
                segment,
                doc,
                chars,
                (long) n,
                ptrOf(result),
                lenOf(result)
            );
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the snippets");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code text} analyzed as field {@code field} by the analyzer {@code spec} describes ({@code
     * analysis.rs}; {@link AnalyzerSpec} writes the spec), its tokens in {@code out[0]}.
     */
    public static int analyze(byte[] spec, byte[] field, byte[] text, byte[][] out) {
        if (spec == null || field == null || text == null) {
            return nullArgument(spec == null ? "spec" : field == null ? "field" : "text");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.ANALYZE.invokeExact(
                s.bytes(spec),
                (long) spec.length,
                s.bytes(field),
                (long) field.length,
                s.bytes(text),
                (long) text.length,
                ptrOf(result),
                lenOf(result)
            );
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the tokens");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Whether the native runtime runs the Painless score script {@code spec} describes (the node's
     * fields after its sub-query, {@code painless_score.rs}) exactly as Painless would, decided
     * before anything runs: {@code out[0]} empty when it does, else the reason as UTF-8.
     */
    public static int painlessCheck(byte[] spec, byte[][] out) {
        if (spec == null) {
            return nullArgument("spec");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.PAINLESS_CHECK.invokeExact(s.bytes(spec), (long) spec.length, ptrOf(result), lenOf(result));
            if (rc == OK) {
                byte[] reason = take(result);
                if (reason == null) {
                    return tooLarge("the reason");
                }
                out[0] = reason;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Registers the JDK's Thai word iterator ({@code ThaiWordsUpcall} in {@code jvm_highlight.rs}),
     * which the highlighters ask for the boundaries of a text in the Thai locale.
     */
    public static int highlightSetThaiUpcall(MemorySegment upcall) {
        try {
            return (int) H.HIGHLIGHT_SET_THAI_UPCALL.invokeExact(upcall);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
