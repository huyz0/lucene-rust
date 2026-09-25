/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;

import org.opensearch.test.rest.yaml.ClientYamlTestCandidate;
import org.opensearch.test.rest.yaml.OpenSearchClientYamlSuiteTestCase;

/**
 * OpenSearch's own REST YAML test suites ({@code rest-api-spec}, 3.8.0), run against a node with
 * this plugin installed -- {@code scripts/verify-opensearch.sh --yaml}. Which suites run is {@code
 * tests.rest.suite}; the node is {@code tests.rest.cluster}.
 *
 * <p>Nothing here is specific to the plugin: that is the point. The suites pass on a stock node,
 * and must pass unchanged when every eligible query phase runs in Rust.
 */
public class LuceneRustYamlIT extends OpenSearchClientYamlSuiteTestCase {
    public LuceneRustYamlIT(@Name("yaml") ClientYamlTestCandidate testCandidate) {
        super(testCandidate);
    }

    @ParametersFactory
    public static Iterable<Object[]> parameters() throws Exception {
        return createParameters();
    }

    /** The suite's own templates are wiped; the one routing every shape native is ours to keep. */
    @Override
    protected boolean preserveTemplatesUponCompletion() {
        return Boolean.getBoolean("lucene_rust.preserve_templates");
    }
}
