/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */

import java.nio.file.Path;
import java.util.*;
import org.apache.lucene.document.FeatureField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.FSDirectory;
import org.opensearch.neuralsearch.sparse.data.SparseVector;
import org.opensearch.neuralsearch.sparse.quantization.ByteQuantizer;
import org.opensearch.neuralsearch.sparse.query.*;

/**
 * Opens an index with OpenSearch's neural-search on the classpath, prints each segment's codec and
 * how its {@code emb} postings are read to stderr, and for every segment of at least 50 documents
 * prints the top 10 hits of the queries {@code crates/lucene-sparse/tests/seismic_writer.rs} runs
 * ({@code segment query doc score}).
 */
public class Check {
    public static void main(String[] args) throws Exception {
        org.opensearch.neuralsearch.sparse.cache.CircuitBreakerManager.setCircuitBreaker(new org.opensearch.core.common.breaker.NoopCircuitBreaker("x"));
        try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Path.of(args[0])))) {
            for (LeafReaderContext leaf : reader.leaves()) {
                SegmentReader sr = (SegmentReader) leaf.reader();
                Terms terms = sr.terms("emb");
                long postings = 0;
                TermsEnum te = terms.iterator();
                while (te.next() != null) {
                    PostingsEnum pe = te.postings(null, PostingsEnum.FREQS);
                    while (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) postings++;
                }
                System.err.println(sr.getSegmentName() + " codec=" + sr.getSegmentInfo().info.getCodec().getName()
                    + " maxDoc=" + sr.maxDoc() + " terms=" + terms.getClass().getSimpleName() + " postings=" + postings);
                if (sr.maxDoc() < 50) continue;
                IndexSearcher searcher = new IndexSearcher(sr);
                for (int q = 0; q < 20; q++) {
                    Map<Integer, Float> raw = new HashMap<>();
                    List<String> top = new ArrayList<>();
                    BooleanQuery.Builder fb = new BooleanQuery.Builder();
                    for (int i = 4; i >= 0; i--) {
                        int t = (q * 37 + i * 61) % 300;
                        raw.put(t, 0.5f + i);
                        top.add(Integer.toString(t));
                        fb.add(FeatureField.newLinearQuery("emb", Integer.toString(t), 0.5f + i), BooleanClause.Occur.SHOULD);
                    }
                    SparseVectorQuery query = SparseVectorQuery.builder()
                        .fieldName("emb")
                        .queryContext(SparseQueryContext.builder().tokens(top).heapFactor(1.0f).k(10).build())
                        .queryVector(new SparseVector(raw, new ByteQuantizer(16.0f)))
                        .rawQueryTokens(raw)
                        .fallbackQuery(fb.build())
                        .build();
                    TopDocs td = searcher.search(query, 10);
                    for (ScoreDoc sd : td.scoreDocs) {
                        System.out.println(sr.getSegmentName() + " " + q + " " + sd.doc + " " + sd.score);
                    }
                }
            }
        }
    }
}
