/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.hexid;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.AutomatonQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.MultiTermQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.automaton.Automata;
import org.apache.lucene.util.automaton.Automaton;
import org.apache.lucene.util.automaton.Operations;
import org.apache.lucene.util.automaton.UTF32ToUTF8;
import org.opensearch.common.lucene.search.AutomatonQueries;
import org.opensearch.common.unit.Fuzziness;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.DerivedFieldGenerator;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.SortedSetDocValuesFetcher;
import org.opensearch.index.mapper.SourceValueFetcher;
import org.opensearch.index.mapper.StoredFieldFetcher;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.time.ZoneId;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

/**
 * {@code hex_id}: a {@code keyword} for hex identifiers (trace, span and request ids) that indexes
 * and doc-values each lowercase hex value as its bytes, half its characters ({@link HexIdCodec}),
 * and gives the string back everywhere a {@code keyword} would: hits' sort values, aggregation keys,
 * {@code docvalue_fields}, scripts and derived source. Any other value is kept as its exact UTF-8,
 * so {@code term}, {@code terms}, {@code exists} and a {@code case_insensitive} {@code term} match
 * what they match on a {@code keyword} holding the same data. The files are plain Lucene terms and
 * {@code SORTED_SET} doc values.
 *
 * <p>Its type is a {@link KeywordFieldMapper.KeywordFieldType}, so every native path that answers a
 * {@code keyword} (terms, composite and significant terms aggregations, collapse, sorts) answers it,
 * formatting through {@link HexIdDocValueFormat}. Text-pattern queries ({@code prefix}, {@code
 * wildcard}, {@code regexp}, {@code fuzzy}) and {@code range} are refused, as on an {@code ip}: the
 * bytes do not order or match as the characters do once non-canonical values are mixed in.
 */
public final class HexIdFieldMapper extends ParametrizedFieldMapper {
    public static final String CONTENT_TYPE = "hex_id";

    public static final ParametrizedFieldMapper.TypeParser PARSER = new ParametrizedFieldMapper.TypeParser((n, c) -> new Builder(n));

    private static HexIdFieldMapper toType(FieldMapper in) {
        return (HexIdFieldMapper) in;
    }

    /** The mapping's parameters: {@code keyword}'s, less {@code store} and the analysis and scoring ones. */
    public static final class Builder extends ParametrizedFieldMapper.Builder {
        private final Parameter<Boolean> indexed = Parameter.indexParam(m -> toType(m).indexed, true);
        private final Parameter<Boolean> hasDocValues = Parameter.docValuesParam(m -> toType(m).hasDocValues, true);
        private final Parameter<String> nullValue = Parameter.stringParam("null_value", false, m -> toType(m).nullValue, null)
            .acceptsNull();
        private final Parameter<Integer> ignoreAbove = Parameter.intParam("ignore_above", true, m -> toType(m).ignoreAbove, Integer.MAX_VALUE)
            .setValidator(v -> {
                if (v < 0) {
                    throw new IllegalArgumentException("[ignore_above] must be positive, got [" + v + "]");
                }
            });
        private final Parameter<Map<String, String>> meta = Parameter.metaParam();

        public Builder(String name) {
            super(name);
            builder = this;
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            return List.of(indexed, hasDocValues, nullValue, ignoreAbove, meta);
        }

        @Override
        public HexIdFieldMapper build(BuilderContext context) {
            HexIdFieldType ft = new HexIdFieldType(
                buildFullName(context),
                indexed.getValue(),
                hasDocValues.getValue(),
                ignoreAbove.getValue(),
                nullValue.getValue(),
                meta.getValue()
            );
            return new HexIdFieldMapper(name, ft, multiFieldsBuilder.build(this, context), copyTo.build(), this);
        }
    }

    /** The term field: one untokenized term, docs only, no norms, as a {@code keyword}'s. */
    private static final FieldType TERM_TYPE = new FieldType();

    static {
        TERM_TYPE.setTokenized(false);
        TERM_TYPE.setOmitNorms(true);
        TERM_TYPE.setIndexOptions(IndexOptions.DOCS);
        TERM_TYPE.freeze();
    }

    private final boolean indexed;
    private final boolean hasDocValues;
    private final String nullValue;
    private final int ignoreAbove;

    private HexIdFieldMapper(String simpleName, HexIdFieldType type, MultiFields multiFields, CopyTo copyTo, Builder builder) {
        super(simpleName, type, multiFields, copyTo);
        this.indexed = builder.indexed.getValue();
        this.hasDocValues = builder.hasDocValues.getValue();
        this.nullValue = builder.nullValue.getValue();
        this.ignoreAbove = builder.ignoreAbove.getValue();
    }

    @Override
    public HexIdFieldType fieldType() {
        return (HexIdFieldType) super.fieldType();
    }

    @Override
    protected String contentType() {
        return CONTENT_TYPE;
    }

    @Override
    public ParametrizedFieldMapper.Builder getMergeBuilder() {
        return new Builder(simpleName()).init(this);
    }

    /** {@code KeywordFieldMapper.parseCreateField}, the term and doc value encoded. */
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
        BytesRef term = HexIdCodec.encode(value);
        if (indexed) {
            context.doc().add(new Field(fieldType().name(), term, TERM_TYPE));
        }
        if (hasDocValues) {
            context.doc().add(new SortedSetDocValuesField(fieldType().name(), term));
        } else if (indexed) {
            createFieldNamesField(context);
        }
    }

    @Override
    protected void canDeriveSourceInternal() {
        if (ignoreAbove != Integer.MAX_VALUE) {
            throw new UnsupportedOperationException(
                "Unable to derive source for [" + name() + "] with ignore_above set"
            );
        }
        checkDocValuesForDerivedSource();
    }

    @Override
    protected DerivedFieldGenerator derivedFieldGenerator() {
        return new DerivedFieldGenerator(
            mappedFieldType,
            new SortedSetDocValuesFetcher(mappedFieldType, simpleName()),
            new StoredFieldFetcher(mappedFieldType, simpleName())
        );
    }

    /** The field type: a {@code keyword}'s, with the value encoded on the way in and decoded on the way out. */
    public static final class HexIdFieldType extends KeywordFieldMapper.KeywordFieldType {
        private final int ignoreAbove;
        private final String nullValue;

        public HexIdFieldType(
            String name,
            boolean indexed,
            boolean hasDocValues,
            int ignoreAbove,
            String nullValue,
            Map<String, String> meta
        ) {
            super(name, indexed, hasDocValues, false, meta);
            this.ignoreAbove = ignoreAbove;
            this.nullValue = nullValue;
        }

        /** A searchable, doc-valued field of the defaults, for tests. */
        public HexIdFieldType(String name) {
            this(name, true, true, Integer.MAX_VALUE, null, Map.of());
        }

        @Override
        public String typeName() {
            return CONTENT_TYPE;
        }

        @Override
        protected BytesRef indexedValueForSearch(Object value) {
            if (value == null) {
                return null;
            }
            String s = value instanceof BytesRef b ? b.utf8ToString() : value.toString();
            return HexIdCodec.encode(s);
        }

        @Override
        public Object valueForDisplay(Object value) {
            if (value instanceof BytesRef b) {
                return HexIdCodec.decode(b);
            }
            return value;
        }

        @Override
        public DocValueFormat docValueFormat(String format, ZoneId timeZone) {
            if (format != null) {
                throw new IllegalArgumentException("Field [" + name() + "] of type [" + typeName() + "] does not support custom formats");
            }
            if (timeZone != null) {
                throw new IllegalArgumentException("Field [" + name() + "] of type [" + typeName() + "] does not support custom time zones");
            }
            return HexIdDocValueFormat.INSTANCE;
        }

        /** {@code keyword}'s ordinals, with scripts given the strings and a sort's {@code missing} encoded. */
        @Override
        public IndexFieldData.Builder fielddataBuilder(String fullyQualifiedIndexName, Supplier<SearchLookup> searchLookup) {
            failIfNoDocValues();
            return (cache, breakerService) -> new HexIdIndexFieldData(cache, name(), breakerService);
        }

        /** {@code keyword}'s: the source values, those over {@code ignore_above} left out. */
        @Override
        public ValueFetcher valueFetcher(QueryShardContext context, SearchLookup searchLookup, String format) {
            if (format != null) {
                throw new IllegalArgumentException("Field [" + name() + "] of type [" + typeName() + "] doesn't support formats.");
            }
            return new SourceValueFetcher(name(), context, nullValue) {
                @Override
                protected String parseSourceValue(Object value) {
                    String s = value.toString();
                    return s.length() > ignoreAbove ? null : s;
                }
            };
        }

        /**
         * The terms a {@code keyword} would match case-insensitively: the value's lowercase hex
         * bytes when it is hex, and every case variant of it stored as text.
         */
        @Override
        public Query termQueryCaseInsensitive(Object value, QueryShardContext context) {
            failIfNotIndexedAndNoDocValues();
            String s = value instanceof BytesRef b ? b.utf8ToString() : value.toString();
            Automaton text = new UTF32ToUTF8().convert(AutomatonQueries.toCaseInsensitiveString(s));
            Automaton escaped = Operations.concatenate(
                List.of(Automata.makeBinary(new BytesRef(new byte[] { HexIdCodec.MARKER, HexIdCodec.TEXT })), text)
            );
            Query textQuery = new AutomatonQuery(
                new Term(name(), s),
                Operations.determinize(escaped, Operations.DEFAULT_DETERMINIZE_WORK_LIMIT),
                true,
                isSearchable() ? MultiTermQuery.CONSTANT_SCORE_BLENDED_REWRITE : MultiTermQuery.DOC_VALUES_REWRITE
            );
            String lower = s.toLowerCase(Locale.ROOT);
            if (HexIdCodec.isCanonical(lower) == false) {
                return new ConstantScoreQuery(textQuery);
            }
            BytesRef bytes = HexIdCodec.encode(lower);
            BooleanQuery.Builder b = new BooleanQuery.Builder();
            b.add(
                isSearchable() ? new TermQuery(new Term(name(), bytes)) : SortedSetDocValuesField.newSlowExactQuery(name(), bytes),
                BooleanClause.Occur.SHOULD
            );
            b.add(textQuery, BooleanClause.Occur.SHOULD);
            return new ConstantScoreQuery(b.build());
        }

        @Override
        public Query prefixQuery(String value, MultiTermQuery.RewriteMethod method, boolean caseInsensitive, QueryShardContext context) {
            throw unsupported("prefix");
        }

        @Override
        public Query wildcardQuery(String value, MultiTermQuery.RewriteMethod method, boolean caseInsensitive, QueryShardContext context) {
            throw unsupported("wildcard");
        }

        @Override
        public Query normalizedWildcardQuery(String value, MultiTermQuery.RewriteMethod method, QueryShardContext context) {
            throw unsupported("wildcard");
        }

        @Override
        public Query regexpQuery(
            String value,
            int syntaxFlags,
            int matchFlags,
            int maxDeterminizedStates,
            MultiTermQuery.RewriteMethod method,
            QueryShardContext context
        ) {
            throw unsupported("regexp");
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
            throw unsupported("fuzzy");
        }

        @Override
        public Query rangeQuery(Object lowerTerm, Object upperTerm, boolean includeLower, boolean includeUpper, QueryShardContext context) {
            throw unsupported("range");
        }

        private IllegalArgumentException unsupported(String query) {
            return new IllegalArgumentException(
                "Field [" + name() + "] of type [" + CONTENT_TYPE + "] does not support [" + query + "] queries"
            );
        }
    }
}
