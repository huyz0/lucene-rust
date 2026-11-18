/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.core.rest.RestStatus;

public class MlLimitsTest {
    private static final long GB = 1L << 30;

    @Test
    public void heapAtThresholdOpensTheMemoryBreaker() {
        assertEquals(MlLimits.HEAP_OPEN, MlLimits.open(85, 100, 85, 0, GB, 90));
        assertNull(MlLimits.open(84, 100, 85, 0, GB, 90));
    }

    @Test
    public void nativeAnonymousMemoryOpensItsOwnBreaker() {
        assertEquals(MlLimits.NATIVE_OPEN, MlLimits.open(0, 100, 85, 90, 100, 90));
        assertNull(MlLimits.open(0, 100, 85, 89, 100, 90));
    }

    @Test
    public void aThresholdOf100NeverOpensAndUnreadableNativeMemoryNeverOpens() {
        assertNull(MlLimits.open(100, 100, 100, GB, GB, 100));
        assertNull(MlLimits.open(0, 100, 85, -1, GB, 90));
        assertNull(MlLimits.open(0, 100, 85, GB, -1, 90));
    }

    @Test
    public void anAnswerCostsHeapPerFloat() {
        assertEquals(20_000L * 384 * MlLimits.HEAP_PER_FLOAT, MlLimits.answerHeap(20_000, 384));
        assertEquals(0, MlLimits.answerHeap(0, 384));
        assertEquals(0, MlLimits.answerHeap(10, -1));
    }

    @Test
    public void theModelCountRefusesTheOneOverTheLimitWith429() {
        AtomicInteger trips = new AtomicInteger();
        MlLimits limits = new MlLimits(() -> 100, () -> 100, () -> 2, trips);
        limits.checkModelCount(false, 1);
        OpenSearchStatusException e = assertThrows(OpenSearchStatusException.class, () -> limits.checkModelCount(false, 2));
        assertEquals(RestStatus.TOO_MANY_REQUESTS, e.status());
        assertTrue(e.getMessage().contains("local"));
        assertEquals(1, trips.get());
        assertTrue(assertThrows(OpenSearchStatusException.class, () -> limits.checkModelCount(true, 5)).getMessage().contains("remote"));
    }

    @Test
    public void aHugeAnswerOpensTheBreakerBeforeItIsBuilt() {
        AtomicInteger trips = new AtomicInteger();
        MlLimits limits = new MlLimits(() -> 1, () -> 100, () -> 10, trips);
        OpenSearchStatusException e = assertThrows(OpenSearchStatusException.class, () -> limits.checkBreakers(Long.MAX_VALUE / 2));
        assertEquals(MlLimits.HEAP_OPEN, e.getMessage());
        assertEquals(1, trips.get());
    }
}
