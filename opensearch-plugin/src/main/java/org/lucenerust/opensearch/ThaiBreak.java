package org.lucenerust.opensearch;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.text.BreakIterator;
import java.text.StringCharacterIterator;
import java.util.Locale;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_CHAR;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * The JVM's Thai word iterator, offered to the native analysis. Lucene's {@code ThaiTokenizer} cuts
 * Thai runs with {@code BreakIterator.getWordInstance(th)} over the JDK's own dictionary; the native
 * tokenizer asks this upcall for each sentence's boundaries so both sides cut the same words
 * ({@code thai_break.rs}).
 */
final class ThaiBreak {
    private static final ThreadLocal<BreakIterator> ITERATOR =
        ThreadLocal.withInitial(() -> BreakIterator.getWordInstance(Locale.forLanguageTag("th")));
    private static volatile boolean installed;

    private ThaiBreak() {}

    /** Registers the upcall with the native library, once per JVM. */
    @SuppressWarnings("restricted") // an upcall stub is the point
    static synchronized void install() {
        if (installed) {
            return;
        }
        try {
            MethodHandle target = MethodHandles.lookup()
                .findStatic(
                    ThaiBreak.class,
                    "boundaries",
                    MethodType.methodType(int.class, MemorySegment.class, int.class, MemorySegment.class, int.class)
                );
            FunctionDescriptor fd = FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT);
            MemorySegment stub = Linker.nativeLinker().upcallStub(target, fd, Arena.global());
            int rc = NativeAnalysis.thaiSetBreakUpcall(stub);
            if (rc != NativeBridge.OK) {
                throw new IllegalStateException("lucene-rust: registering the Thai break upcall failed: " + NativeBridge.lastError());
            }
            installed = true;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * The word boundaries of the {@code len} UTF-16 units at {@code text}, written to {@code out}
     * while there is room; the number of boundaries, or -1 when the iterator failed.
     */
    @SuppressWarnings("restricted") // the segments are the native caller's, sized by its contract
    static int boundaries(MemorySegment text, int len, MemorySegment out, int cap) {
        try {
            char[] chars = new char[len];
            MemorySegment.copy(text.reinterpret((long) len * Character.BYTES), JAVA_CHAR, 0, chars, 0, len);
            BreakIterator bi = ITERATOR.get();
            bi.setText(new StringCharacterIterator(new String(chars)));
            MemorySegment slots = out.reinterpret((long) cap * Integer.BYTES);
            int n = 0;
            for (int b = bi.first(); b != BreakIterator.DONE; b = bi.next()) {
                if (n < cap) {
                    slots.setAtIndex(JAVA_INT, n, b);
                }
                n++;
            }
            return n;
        } catch (Throwable t) {
            return -1;
        }
    }
}
