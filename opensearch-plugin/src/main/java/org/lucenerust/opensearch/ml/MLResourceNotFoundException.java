/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.opensearch.OpenSearchException;

/** A model the call named is not there, as ml-commons' {@code MLResourceNotFoundException} reports it: type {@code m_l_resource_not_found_exception}, status 500. */
public final class MLResourceNotFoundException extends OpenSearchException {
    MLResourceNotFoundException(String message) {
        super(message);
    }
}
