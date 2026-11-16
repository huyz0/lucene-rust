/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.highlight;

import org.lucenerust.opensearch.NativeBridge;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.text.BreakIterator;
import java.util.Locale;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_CHAR_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * The JDK's word {@link BreakIterator} for Thai, which segments by a dictionary the library does
 * not carry: a request in the Thai locale asks for the word boundaries of each text it cuts
 * ({@code ThaiWordsUpcall} in {@code jvm_highlight.rs}), and gets what OpenSearch's highlighter
 * would cut at.
 */
final class ThaiWords {
    private static final Locale THAI = Locale.forLanguageTag("th");
    private static volatile boolean installed;

    private ThaiWords() {}

    /** Registers the upcall, once per process (the stub has ThaiWordsUpcall's signature in jvm_highlight.rs). */
    static void install() {
        if (installed == false) {
            installOnce();
        }
    }

    @SuppressWarnings("restricted")
    private static synchronized void installOnce() {
        if (installed) {
            return;
        }
        try {
            MethodHandle target = MethodHandles.lookup()
                .findStatic(
                    ThaiWords.class,
                    "boundaries",
                    MethodType.methodType(long.class, MemorySegment.class, long.class, MemorySegment.class, long.class)
                );
            MemorySegment stub = Linker.nativeLinker()
                .upcallStub(target, FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG), Arena.global());
            if (NativeBridge.highlightSetThaiUpcall(stub) != NativeBridge.OK) {
                throw new IllegalStateException("lucene-rust: registering the Thai word upcall failed: " + NativeBridge.lastError());
            }
            installed = true;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Every boundary of {@code text} into {@code out}: how many, or -1 when they do not fit or the iterator fails. */
    @SuppressWarnings("restricted") // the segments are raw addresses of the caller's buffers
    static long boundaries(MemorySegment text, long len, MemorySegment out, long cap) {
        try {
            char[] chars = new char[Math.toIntExact(len)];
            MemorySegment.copy(text.reinterpret(len * 2), JAVA_CHAR_UNALIGNED, 0, chars, 0, chars.length);
            BreakIterator words = BreakIterator.getWordInstance(THAI);
            words.setText(new String(chars));
            MemorySegment o = out.reinterpret(cap * Integer.BYTES);
            long n = 0;
            for (int b = words.first(); b != BreakIterator.DONE; b = words.next()) {
                if (n >= cap) {
                    return -1;
                }
                o.setAtIndex(JAVA_INT_UNALIGNED, n++, b);
            }
            return n;
        } catch (Throwable t) {
            return -1;
        }
    }
}
