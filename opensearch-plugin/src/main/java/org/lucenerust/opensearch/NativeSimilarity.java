/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.similarities.Similarity;

/**
 * The extension point for a plugin's own {@link Similarity}: one that scores exactly as a built-in
 * similarity (BM25, classic, boolean, DFR, IB, DFI, LM Dirichlet, LM Jelinek-Mercer, scripted with a
 * native script) names that built-in here, and its searches run natively instead of falling back as
 * {@code field_similarity}.
 *
 * <p>The contract is exactness: {@link #nativeEquivalent()} must score every document as this
 * similarity does, norms included. A similarity that computes anything else cannot run natively.
 */
public interface NativeSimilarity {
    /** The built-in similarity this one scores as. Never itself, and never another {@code NativeSimilarity}. */
    Similarity nativeEquivalent();
}
