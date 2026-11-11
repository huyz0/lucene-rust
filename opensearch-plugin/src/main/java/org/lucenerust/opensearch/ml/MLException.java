/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.opensearch.OpenSearchException;

/**
 * A failure while running a model, as ml-commons' own {@code MLException} reports it: type {@code
 * m_l_exception} (from the class name), status 500.
 */
public final class MLException extends OpenSearchException {
    MLException(String message) {
        super(message);
    }

    MLException(String message, Throwable cause) {
        super(message, cause);
    }
}
