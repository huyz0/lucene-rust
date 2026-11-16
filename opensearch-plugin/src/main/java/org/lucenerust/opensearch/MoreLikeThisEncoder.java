/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.search.similarities.TFIDFSimilarity;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.CharsRefBuilder;
import org.opensearch.common.lucene.search.MoreLikeThisQuery;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.StringReader;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * OpenSearch's {@code MoreLikeThisQuery} for the plain highlighter, which rewrites it over a
 * one-document index of the text it highlights ({@code XMoreLikeThis}). In that index every
 * term of the text has document frequency 1 and there is 1 document, so what the rewrite needs
 * of the index is only whether the text holds a term; everything else is the query's own and
 * is collected here as {@code XMoreLikeThis} collects it, in the iteration order of the same
 * {@link HashMap}s, so the native side picks the terms stock picks: the candidates of each
 * field in that order, with their {@code tf * idf}. The native side keeps those the text
 * holds in Lucene's bounded priority queue ({@code NODE_MORE_LIKE_THIS} in {@code queries.rs}).
 */
final class MoreLikeThisEncoder {
    private static final int DEFAULT_MAX_NUM_TOKENS_PARSED = 5000;

    private MoreLikeThisEncoder() {}

    private static final class Int {
        int x = 1;
    }

    private static final class Params {
        Analyzer analyzer;
        String[] moreLikeFields;
        int minTermFreq, minDocFreq, maxDocFreq, maxQueryTerms, minWordLen, maxWordLen;
        Set<?> stopWords;
        Set<Term> skipTerms;
    }

    /** Writes the node, or returns why it cannot be written. */
    static String write(MoreLikeThisQuery q, ByteArrayOutputStream out) {
        try {
            Params p = new Params();
            p.analyzer = q.getAnalyzer();
            p.moreLikeFields = q.getMoreLikeFields();
            p.minTermFreq = q.getMinTermFrequency();
            p.minDocFreq = q.getMinDocFreq();
            p.maxDocFreq = q.getMaxDocFreq();
            p.maxQueryTerms = q.getMaxQueryTerms();
            p.minWordLen = q.getMinWordLen();
            p.maxWordLen = q.getMaxWordLen();
            p.stopWords = q.getStopWords();
            String[] unlikeText = (String[]) read(q, "unlikeText");
            Fields[] unlikeFields = (Fields[]) read(q, "unlikeFields");
            float boostFactor = (Float) read(q, "boostTermsFactor");
            if (unlikeText != null || unlikeFields != null) {
                Set<Term> skip = skipTerms(p, unlikeText, unlikeFields);
                p.skipTerms = skip.isEmpty() ? null : skip;
            }
            Similarity similarity = q.getSimilarity();
            TFIDFSimilarity tfidf = similarity == null ? new ClassicSimilarity() : (TFIDFSimilarity) similarity;
            // the one-document index: docFreq 1 of 1
            float idf = tfidf.idf(1, 1);

            ByteArrayOutputStream body = new ByteArrayOutputStream();
            int parts = 0;
            if (q.getLikeFields() != null) {
                parts++;
                groupsOfFields(p, q.getLikeFields(), idf, body);
            }
            String[] likeText = q.getLikeTexts();
            if (likeText != null) {
                parts++;
                groupOfText(p, likeText, idf, body);
            }
            out.write(QueryEncoder.NODE_MORE_LIKE_THIS_KIND);
            out.write(q.isBoostTerms() ? 1 : 0);
            writeInt(out, Float.floatToRawIntBits(boostFactor));
            writeInt(out, parts);
            out.writeBytes(body.toByteArray());
            return "";
        } catch (IOException | ReflectiveOperationException | RuntimeException e) {
            return "highlight_MoreLikeThisQuery";
        }
    }

    private static Object read(MoreLikeThisQuery q, String name) throws ReflectiveOperationException {
        Field f = MoreLikeThisQuery.class.getDeclaredField(name);
        f.setAccessible(true);
        return f.get(q);
    }

    private static Set<Term> skipTerms(Params p, String[] unlikeText, Fields[] unlikeFields) throws IOException {
        Set<Term> skip = new HashSet<>();
        if (unlikeText != null) {
            for (String text : unlikeText) {
                String fieldName = p.moreLikeFields[0];
                try (TokenStream ts = p.analyzer.tokenStream(fieldName, text)) {
                    CharTermAttribute termAtt = ts.addAttribute(CharTermAttribute.class);
                    ts.reset();
                    while (ts.incrementToken()) {
                        skip.add(new Term(fieldName, termAtt.toString()));
                    }
                    ts.end();
                }
            }
        }
        if (unlikeFields != null) {
            for (Fields fields : unlikeFields) {
                for (String fieldName : fields) {
                    Terms terms = fields.terms(fieldName);
                    TermsEnum termsEnum = terms.iterator();
                    BytesRef text;
                    while ((text = termsEnum.next()) != null) {
                        skip.add(new Term(fieldName, text.utf8ToString()));
                    }
                }
            }
        }
        return skip;
    }

    /** {@code XMoreLikeThis.like(Fields...)}: a group per field, in the field names' set order. */
    private static void groupsOfFields(Params p, Fields[] likeFields, float idf, ByteArrayOutputStream out) throws IOException {
        Set<String> fieldNames = new HashSet<>();
        for (Fields fields : likeFields) {
            for (String fieldName : fields) {
                fieldNames.add(fieldName);
            }
        }
        writeInt(out, fieldNames.size());
        for (String fieldName : fieldNames) {
            Map<String, Int> words = new HashMap<>();
            for (Fields fields : likeFields) {
                Terms vector = fields.terms(fieldName);
                if (vector != null) {
                    addFrequencies(p, words, vector, fieldName);
                }
            }
            group(p, words, fieldName, idf, out);
        }
    }

    /** {@code XMoreLikeThis.like(fieldName, Reader...)}: one group, of the first more-like field. */
    private static void groupOfText(Params p, String[] likeText, float idf, ByteArrayOutputStream out) throws IOException {
        Map<String, Int> words = new HashMap<>();
        String fieldName = p.moreLikeFields[0];
        for (String text : likeText) {
            addFrequencies(p, new StringReader(text), words, fieldName);
        }
        writeInt(out, 1);
        group(p, words, p.moreLikeFields[0], idf, out);
    }

    /** {@code createQueue}'s filters that need no index, over {@code words} in its iteration order. */
    private static void group(Params p, Map<String, Int> words, String topField, float idf, ByteArrayOutputStream out) {
        int limit = Math.min(p.maxQueryTerms, words.size());
        ByteArrayOutputStream candidates = new ByteArrayOutputStream();
        int count = 0;
        for (String word : words.keySet()) {
            int tf = words.get(word).x;
            if (p.minTermFreq > 0 && tf < p.minTermFreq) {
                continue;
            }
            if (p.minDocFreq > 0 && 1 < p.minDocFreq) {
                continue;
            }
            if (1 > p.maxDocFreq) {
                continue;
            }
            float score = tf * idf;
            writeBytes(candidates, word.getBytes(StandardCharsets.UTF_8));
            writeInt(candidates, Float.floatToRawIntBits(score));
            count++;
        }
        writeBytes(out, topField.getBytes(StandardCharsets.UTF_8));
        writeInt(out, Math.max(limit, 0));
        writeInt(out, count);
        out.writeBytes(candidates.toByteArray());
    }

    private static void addFrequencies(Params p, Map<String, Int> map, Terms vector, String fieldName) throws IOException {
        TermsEnum termsEnum = vector.iterator();
        CharsRefBuilder spare = new CharsRefBuilder();
        BytesRef text;
        while ((text = termsEnum.next()) != null) {
            spare.copyUTF8Bytes(text);
            String term = spare.toString();
            if (isNoise(p, term) || isSkip(p, fieldName, term)) {
                continue;
            }
            PostingsEnum docs = termsEnum.postings(null);
            int freq = 0;
            while (docs != null && docs.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                freq += docs.freq();
            }
            Int cnt = map.get(term);
            if (cnt == null) {
                cnt = new Int();
                map.put(term, cnt);
                cnt.x = freq;
            } else {
                cnt.x += freq;
            }
        }
    }

    private static void addFrequencies(Params p, StringReader r, Map<String, Int> map, String fieldName) throws IOException {
        try (TokenStream ts = p.analyzer.tokenStream(fieldName, r)) {
            int tokenCount = 0;
            CharTermAttribute termAtt = ts.addAttribute(CharTermAttribute.class);
            ts.reset();
            while (ts.incrementToken()) {
                String word = termAtt.toString();
                tokenCount++;
                if (tokenCount > DEFAULT_MAX_NUM_TOKENS_PARSED) {
                    break;
                }
                if (isNoise(p, word) || isSkip(p, fieldName, word)) {
                    continue;
                }
                Int cnt = map.get(word);
                if (cnt == null) {
                    map.put(word, new Int());
                } else {
                    cnt.x++;
                }
            }
            ts.end();
        }
    }

    private static boolean isNoise(Params p, String term) {
        int len = term.length();
        if (p.minWordLen > 0 && len < p.minWordLen) {
            return true;
        }
        if (p.maxWordLen > 0 && len > p.maxWordLen) {
            return true;
        }
        return p.stopWords != null && p.stopWords.contains(term);
    }

    private static boolean isSkip(Params p, String field, String value) {
        return field != null && p.skipTerms != null && p.skipTerms.contains(new Term(field, value));
    }

    private static void writeBytes(ByteArrayOutputStream out, byte[] b) {
        writeInt(out, b.length);
        out.writeBytes(b);
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }
}
