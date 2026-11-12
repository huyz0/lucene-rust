/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.substring;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.function.Supplier;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.MultiTermQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.automaton.Automaton;
import org.apache.lucene.util.automaton.CompiledAutomaton;
import org.apache.lucene.util.automaton.Operations;
import org.apache.lucene.util.automaton.RegExp;
import org.opensearch.common.lucene.BytesRefs;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.common.unit.Fuzziness;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.analysis.IndexAnalyzers;
import org.opensearch.index.analysis.NamedAnalyzer;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.SourceValueFetcher;
import org.opensearch.index.mapper.StringFieldType;
import org.opensearch.index.mapper.TextSearchInfo;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.search.lookup.SearchLookup;

/**
 * {@code rust_wildcard}: OpenSearch's {@code wildcard} field, kept as one {@code RustSubstring}
 * BINARY doc value per document instead of a trigram postings field beside a {@code SORTED_SET} copy
 * of the value. Under the Rust engine the value goes into LZ4 blocks of 32 documents indexed by the
 * trigrams they hold ({@code lucene-codecs/src/substring.rs}): about half the disk of a {@code
 * wildcard} field, indexed about 3x faster.
 *
 * <p>Its queries are the {@code wildcard} field's: {@code term}, {@code terms}, {@code prefix},
 * {@code wildcard}, {@code regexp} (and {@code case_insensitive}), {@code match} and {@code exists},
 * built here exactly as {@code WildcardFieldMapper.WildcardFieldType} builds their tests, so a
 * document matches when it would on a {@code wildcard} field holding the same value. Parameters:
 * {@code null_value}, {@code ignore_above}, {@code normalizer} and {@code meta}, as there.
 *
 * <p>What differs: a document takes one value (an array is refused); the field has no ordinals, so
 * sorting, aggregations, scripts and {@code docvalue_fields} on it are refused ({@code fields} and
 * {@code _source} read it); {@code query_string} wildcards use the field's own wildcard test where
 * the {@code wildcard} field runs a {@code WildcardQuery} over its trigram terms; {@code fuzzy} and
 * {@code range} are refused, as there.
 */
public final class RustWildcardFieldMapper extends ParametrizedFieldMapper {
    public static final String CONTENT_TYPE = "rust_wildcard";

    /**
     * The field attribute that routes the field to the {@code RustSubstring} doc-values format;
     * {@code SUBSTRING_ATTRIBUTE} in {@code engine_writer.rs}. Its value is the block shift.
     */
    public static final String SUBSTRING_ATTRIBUTE = "lucene_rust.substring";
    /** {@code DEFAULT_DEFAULT_BLOCK_SHIFT} in {@code substring.rs}: 32 documents per block. */
    public static final int DEFAULT_BLOCK_SHIFT = 5;

    public static final TypeParser PARSER = new TypeParser((n, c) -> new Builder(n, c.getIndexAnalyzers()));

    private static RustWildcardFieldMapper toType(FieldMapper in) {
        return (RustWildcardFieldMapper) in;
    }

    /** The doc-values field type of every {@code rust_wildcard} value. */
    static final FieldType VALUE_TYPE = new FieldType();

    static {
        VALUE_TYPE.setIndexOptions(IndexOptions.NONE);
        VALUE_TYPE.setStored(false);
        VALUE_TYPE.setDocValuesType(DocValuesType.BINARY);
        VALUE_TYPE.putAttribute(SUBSTRING_ATTRIBUTE, Integer.toString(DEFAULT_BLOCK_SHIFT));
        VALUE_TYPE.freeze();
    }

    /** The doc value a {@code rust_wildcard} field indexes for {@code value}. */
    public static Field valueField(String name, String value) {
        return new Field(name, new BytesRef(value), VALUE_TYPE);
    }

    /** The mapping's parameters: the {@code wildcard} field's, less {@code doc_values}. */
    public static final class Builder extends ParametrizedFieldMapper.Builder {
        private final Parameter<String> nullValue = Parameter.stringParam("null_value", false, m -> toType(m).nullValue, null)
            .acceptsNull();
        private final Parameter<Integer> ignoreAbove = Parameter.intParam("ignore_above", true, m -> toType(m).ignoreAbove, Integer.MAX_VALUE);
        private final Parameter<String> normalizer = Parameter.stringParam("normalizer", false, m -> toType(m).normalizerName, "default");
        private final Parameter<Map<String, String>> meta = Parameter.metaParam();
        private final IndexAnalyzers indexAnalyzers;

        public Builder(String name, IndexAnalyzers indexAnalyzers) {
            super(name);
            this.indexAnalyzers = indexAnalyzers;
        }

        public Builder(String name) {
            this(name, null);
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            return Arrays.asList(nullValue, ignoreAbove, normalizer, meta);
        }

        @Override
        public RustWildcardFieldMapper build(BuilderContext context) {
            String normalizerName = normalizer.getValue();
            NamedAnalyzer analyzer = Lucene.KEYWORD_ANALYZER;
            if ("default".equals(normalizerName) == false) {
                analyzer = indexAnalyzers == null ? null : indexAnalyzers.getNormalizer(normalizerName);
                if (analyzer == null) {
                    throw new IllegalArgumentException("normalizer [" + normalizerName + "] not found for field [" + name + "]");
                }
            }
            RustWildcardFieldType ft = new RustWildcardFieldType(
                context.path().pathAsText(name),
                analyzer,
                ignoreAbove.getValue(),
                nullValue.getValue(),
                meta.getValue()
            );
            return new RustWildcardFieldMapper(name, ft, multiFieldsBuilder.build(this, context), copyTo.build(), this);
        }
    }

    private final String nullValue;
    private final int ignoreAbove;
    private final String normalizerName;
    private final IndexAnalyzers indexAnalyzers;

    private RustWildcardFieldMapper(String simpleName, RustWildcardFieldType type, MultiFields multiFields, CopyTo copyTo, Builder builder) {
        super(simpleName, type, multiFields, copyTo);
        this.nullValue = builder.nullValue.getValue();
        this.ignoreAbove = builder.ignoreAbove.getValue();
        this.normalizerName = builder.normalizer.getValue();
        this.indexAnalyzers = builder.indexAnalyzers;
    }

    @Override
    public RustWildcardFieldType fieldType() {
        return (RustWildcardFieldType) super.fieldType();
    }

    @Override
    protected String contentType() {
        return CONTENT_TYPE;
    }

    @Override
    public ParametrizedFieldMapper.Builder getMergeBuilder() {
        return new Builder(simpleName(), indexAnalyzers).init(this);
    }

    /** {@code WildcardFieldMapper.parseCreateField}: the normalized value as the field's one doc value. */
    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        String value;
        if (context.externalValueSet()) {
            value = context.externalValue().toString();
        } else {
            XContentParser parser = context.parser();
            value = parser.currentToken() == XContentParser.Token.VALUE_NULL ? nullValue : parser.textOrNull();
        }
        if (value == null || value.length() > ignoreAbove) {
            return;
        }
        NamedAnalyzer normalizer = fieldType().normalizer();
        if (normalizer != null) {
            value = normalizeValue(normalizer, name(), value);
        }
        if (context.doc().getField(fieldType().name()) != null) {
            throw new IllegalArgumentException(
                "field [" + fieldType().name() + "] of type [" + CONTENT_TYPE + "] takes one value per document"
            );
        }
        context.doc().add(valueField(fieldType().name(), value));
    }

    /** {@code KeywordFieldMapper.normalizeValue}: {@code value} through the normalizer, which must give one token. */
    static String normalizeValue(NamedAnalyzer normalizer, String field, String value) throws IOException {
        try (TokenStream ts = normalizer.tokenStream(field, value)) {
            CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
            ts.reset();
            if (ts.incrementToken() == false) {
                throw new IllegalStateException(
                    "The normalization token stream is expected to produce exactly 1 token, but got 0 for analyzer "
                        + normalizer
                        + " and input \""
                        + value
                        + "\""
                );
            }
            String normalized = term.toString();
            if (ts.incrementToken()) {
                throw new IllegalStateException(
                    "The normalization token stream is expected to produce exactly 1 token, but got 2+ for analyzer "
                        + normalizer
                        + " and input \""
                        + value
                        + "\""
                );
            }
            ts.end();
            return normalized;
        }
    }

    /** The field type: the {@code wildcard} field's queries, as tests of each document's value. */
    public static final class RustWildcardFieldType extends StringFieldType {
        private static final Set<Character> WILDCARD_SPECIAL = Set.of('?', '*', '\\');
        private static final Set<Character> REGEXP_SPECIAL = Set.of(
            '.', '^', '$', '*', '+', '?', '(', ')', '[', ']', '{', '}', '|', '/', '\\'
        );

        private final int ignoreAbove;
        private final String nullValue;

        public RustWildcardFieldType(String name, NamedAnalyzer normalizer, int ignoreAbove, String nullValue, Map<String, String> meta) {
            super(name, true, false, true, TextSearchInfo.SIMPLE_MATCH_ONLY, meta);
            setIndexAnalyzer(normalizer);
            this.ignoreAbove = ignoreAbove;
            this.nullValue = nullValue;
        }

        /** A field of the defaults, for tests. */
        public RustWildcardFieldType(String name) {
            this(name, Lucene.KEYWORD_ANALYZER, Integer.MAX_VALUE, null, Map.of());
        }

        @Override
        public String typeName() {
            return CONTENT_TYPE;
        }

        NamedAnalyzer normalizer() {
            return indexAnalyzer();
        }

        /** {@code WildcardFieldType}'s source fetcher: the source values, normalized, those over {@code ignore_above} left out. */
        @Override
        public ValueFetcher valueFetcher(QueryShardContext context, SearchLookup searchLookup, String format) {
            if (format != null) {
                throw new IllegalArgumentException("Field [" + name() + "] of type [" + typeName() + "] doesn't support formats.");
            }
            NamedAnalyzer normalizer = normalizer();
            return new SourceValueFetcher(name(), context, nullValue) {
                @Override
                protected String parseSourceValue(Object value) {
                    String s = value.toString();
                    if (s.length() > ignoreAbove) {
                        return null;
                    }
                    if (normalizer == null) {
                        return s;
                    }
                    try {
                        return normalizeValue(normalizer, name(), s);
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                }
            };
        }

        @Override
        public IndexFieldData.Builder fielddataBuilder(String fullyQualifiedIndexName, Supplier<SearchLookup> searchLookup) {
            throw new IllegalArgumentException(
                "Field [" + name() + "] of type [" + CONTENT_TYPE + "] has no ordinals: it cannot be sorted on, aggregated or read by scripts"
            );
        }

        @Override
        public Query termQuery(Object value, QueryShardContext context) {
            return wildcardQuery(quoteWildcard(BytesRefs.toString(value)), MultiTermQuery.CONSTANT_SCORE_REWRITE, false, context);
        }

        @Override
        public Query termQueryCaseInsensitive(Object value, QueryShardContext context) {
            return wildcardQuery(quoteWildcard(BytesRefs.toString(value)), MultiTermQuery.CONSTANT_SCORE_REWRITE, true, context);
        }

        /** {@code WildcardFieldType.termsQuery}: a value equal to one of {@code values}, as given. */
        @Override
        public Query termsQuery(List<?> values, QueryShardContext context) {
            Set<String> expected = new LinkedHashSet<>();
            for (Object v : values) {
                expected.add(BytesRefs.toString(v));
            }
            BooleanQuery.Builder b = new BooleanQuery.Builder();
            for (String v : expected) {
                b.add(new RustWildcardQuery(name(), RustWildcardQuery.Kind.EXACT, v, false, true, 0, v::equals), BooleanClause.Occur.SHOULD);
            }
            return new ConstantScoreQuery(b.build());
        }

        @Override
        public Query prefixQuery(String value, MultiTermQuery.RewriteMethod method, boolean caseInsensitive, QueryShardContext context) {
            return wildcardQuery(value + "*", method, caseInsensitive, context);
        }

        /** {@code WildcardFieldType.wildcardQuery}'s test of a value. */
        @Override
        public Query wildcardQuery(String value, MultiTermQuery.RewriteMethod method, boolean caseInsensitive, QueryShardContext context) {
            NamedAnalyzer normalizer = normalizer();
            if (normalizer != null) {
                value = normalizeWildcardPattern(name(), value, normalizer);
            }
            final String finalValue = caseInsensitive ? value.toLowerCase(Locale.ROOT) : value;
            Automaton automaton = WildcardQuery.toAutomaton(new Term(name(), finalValue), Operations.DEFAULT_DETERMINIZE_WORK_LIMIT);
            CompiledAutomaton compiled = new CompiledAutomaton(automaton);
            switch (compiled.type) {
                case ALL:
                    return existsQuery(context);
                case NONE:
                    return new MatchNoDocsQuery("Wildcard expression matches nothing");
                case SINGLE: {
                    String exact = performEscape(finalValue, false);
                    Predicate<String> test = caseInsensitive ? s -> s.toLowerCase(Locale.ROOT).equals(exact) : exact::equals;
                    return new RustWildcardQuery(name(), RustWildcardQuery.Kind.EXACT, exact, caseInsensitive, true, 0, test);
                }
                default: {
                    Predicate<String> test = s -> {
                        String v = caseInsensitive ? s.toLowerCase(Locale.ROOT) : s;
                        BytesRef bytes = new BytesRef(v);
                        return compiled.runAutomaton.run(bytes.bytes, bytes.offset, bytes.length);
                    };
                    return new RustWildcardQuery(name(), RustWildcardQuery.Kind.WILDCARD, value, caseInsensitive, true, 0, test);
                }
            }
        }

        @Override
        public Query normalizedWildcardQuery(String value, MultiTermQuery.RewriteMethod method, QueryShardContext context) {
            return wildcardQuery(value, method, false, context);
        }

        /** {@code WildcardFieldType.regexpQuery}'s test of a value. */
        @Override
        @SuppressWarnings("deprecation")
        public Query regexpQuery(
            String value,
            int syntaxFlags,
            int matchFlags,
            int maxDeterminizedStates,
            MultiTermQuery.RewriteMethod method,
            QueryShardContext context
        ) {
            NamedAnalyzer normalizer = normalizer();
            final String finalValue = normalizer != null ? normalizer.normalize(name(), value).utf8ToString() : value;
            final boolean caseInsensitive = matchFlags == RegExp.ASCII_CASE_INSENSITIVE;
            RegExp regExp = new RegExp(finalValue, syntaxFlags, matchFlags);
            Automaton automaton = Operations.determinize(regExp.toAutomaton(), Operations.DEFAULT_DETERMINIZE_WORK_LIMIT);
            CompiledAutomaton compiled = new CompiledAutomaton(automaton);
            long flags = ((long) syntaxFlags << 32) | (matchFlags & 0xffffffffL);
            // The native side parses RegExp's default syntax with no or ASCII case-insensitive matching.
            boolean nativeSyntax = syntaxFlags == RegExp.ALL && (matchFlags == 0 || caseInsensitive);
            switch (compiled.type) {
                case ALL:
                    return existsQuery(context);
                case NONE:
                    return new MatchNoDocsQuery("Regular expression matches nothing");
                case SINGLE: {
                    String exact = performEscape(finalValue, true);
                    if (caseInsensitive) {
                        // As the wildcard field: the value lower-cased against the pattern as written.
                        boolean nativeOk = exact.equals(exact.toLowerCase(Locale.ROOT));
                        return new RustWildcardQuery(
                            name(), RustWildcardQuery.Kind.EXACT, exact, true, nativeOk, flags,
                            s -> s.toLowerCase(Locale.ROOT).equals(exact)
                        );
                    }
                    return new RustWildcardQuery(name(), RustWildcardQuery.Kind.EXACT, exact, false, true, flags, exact::equals);
                }
                default: {
                    Predicate<String> test = s -> {
                        BytesRef bytes = new BytesRef(s);
                        return compiled.runAutomaton.run(bytes.bytes, bytes.offset, bytes.length);
                    };
                    return new RustWildcardQuery(name(), RustWildcardQuery.Kind.REGEXP, finalValue, caseInsensitive, nativeSyntax, flags, test);
                }
            }
        }

        @Override
        public Query fuzzyQuery(
            Object value,
            Fuzziness fuzziness,
            int prefixLength,
            int maxExpansions,
            boolean transpositions,
            MultiTermQuery.RewriteMethod method,
            QueryShardContext context
        ) {
            throw new IllegalArgumentException(
                "Can only use fuzzy queries on keyword and text fields - not on [" + name() + "] which is of type [" + typeName() + "]"
            );
        }

        @Override
        public Query rangeQuery(Object lowerTerm, Object upperTerm, boolean includeLower, boolean includeUpper, QueryShardContext context) {
            throw new UnsupportedOperationException("TODO");
        }

        /** {@code WildcardFieldType.performEscape}: the escapes of {@code ?}, {@code *} and {@code \} (or a regexp's specials) removed. */
        static String performEscape(String str, boolean regexpMode) {
            StringBuilder sb = new StringBuilder();
            Set<Character> targetChars = regexpMode ? REGEXP_SPECIAL : WILDCARD_SPECIAL;
            for (int i = 0; i < str.length(); i++) {
                if (str.charAt(i) == '\\' && (i + 1) < str.length()) {
                    char c = str.charAt(i + 1);
                    if (targetChars.contains(c)) {
                        i++;
                    }
                }
                sb.append(str.charAt(i));
            }
            return sb.toString();
        }

        /** {@code WildcardFieldType.quoteWildcard}. */
        static String quoteWildcard(String str) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < str.length(); i++) {
                if (WILDCARD_SPECIAL.contains(str.charAt(i))) {
                    sb.append('\\');
                }
                sb.append(str.charAt(i));
            }
            return sb.toString();
        }
    }
}
