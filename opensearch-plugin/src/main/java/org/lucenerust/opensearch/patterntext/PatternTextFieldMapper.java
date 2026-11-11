/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.patterntext;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CachingTokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.util.BytesRef;
import org.opensearch.Version;
import org.opensearch.index.analysis.IndexAnalyzers;
import org.opensearch.index.mapper.DerivedFieldGenerator;
import org.opensearch.index.mapper.FieldValueFetcher;
import org.opensearch.index.mapper.FieldValueType;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.TextFieldMapper;
import org.opensearch.index.mapper.TextSearchInfo;
import org.opensearch.index.query.QueryShardContext;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The {@code pattern_text} field: a log message kept as a shared template plus its variables
 * (after Elasticsearch's {@code pattern_text}), so an index holds one copy of each message shape
 * instead of one per line. On 900k real Loghub lines (derived source) the index is a third smaller
 * than with {@code text}, and phrases take 1-5 ms as there.
 *
 * <p>A value {@code v} of field {@code f} is written as three plain Lucene fields, the layout
 * {@code crates/lucene-pattern-text} writes ({@link PatternTextFormat}):
 *
 * <ul>
 *   <li>{@code f}: the message, indexed docs-only without norms, as {@code match_only_text};
 *   <li>{@code f.template}: the template, a {@code SORTED} doc value;
 *   <li>{@code f.args}: the variables, a stored binary value, absent without variables.
 * </ul>
 *
 * <p>It takes {@code match_only_text}'s parameters; term queries score as there, and phrases ({@code match_phrase}, {@code match_phrase_prefix}, with or without slop)
 * are a {@link PatternPhraseQuery}, scoring 0. A document holds at most one value. The message is
 * rebuilt from the template and variables for an index with {@code index.derived_source.enabled},
 * which is where the savings are: elsewhere {@code _source} still holds every message. As its search
 * info reports positions, a plain {@code match} with a multi-word synonym builds a phrase here too
 * (as on {@code text}), where {@code match_only_text} builds the terms' conjunction.
 */
public final class PatternTextFieldMapper extends TextFieldMapper {
    public static final String CONTENT_TYPE = "pattern_text";
    /** The suffix of the {@code SORTED} doc-values field that holds the template. */
    public static final String TEMPLATE_SUFFIX = ".template";
    /** The suffix of the stored field that holds the variables. */
    public static final String ARGS_SUFFIX = ".args";
    /** {@code IndexWriter.MAX_TERM_LENGTH}: the longest {@code SORTED} value Lucene writes. */
    public static final int MAX_TEMPLATE_BYTES = 32766;

    static final FieldType FIELD_TYPE = new FieldType();

    static {
        FIELD_TYPE.setTokenized(true);
        FIELD_TYPE.setStored(false);
        FIELD_TYPE.setStoreTermVectors(false);
        FIELD_TYPE.setOmitNorms(true);
        FIELD_TYPE.setIndexOptions(IndexOptions.DOCS);
        FIELD_TYPE.freeze();
    }

    public static final TypeParser PARSER = new TypeParser((n, c) -> new Builder(n, c.indexVersionCreated(), c.getIndexAnalyzers()));

    /** The parameters {@code text} has and {@code match_only_text} fixes, or this field leaves out. */
    private static final Set<String> REPLACED = Set.of(
        "index_options",
        "norms",
        "term_vector",
        "index_phrases",
        "index_prefixes"
    );

    private PatternTextFieldMapper(String simpleName, FieldType fieldType, PatternTextFieldType mappedFieldType, MultiFields multiFields, CopyTo copyTo, Builder builder) {
        super(simpleName, fieldType, mappedFieldType, null, null, multiFields, copyTo, builder);
    }

    /** {@code text}'s builder with {@code match_only_text}'s fixed parameters. */
    public static final class Builder extends TextFieldMapper.Builder {
        private static Parameter<Boolean> refusedTrue(String name) {
            return Parameter.boolParam(name, false, m -> false, false).setValidator(v -> {
                if (v) {
                    throw new MapperParsingException("[" + name + "] cannot be enabled on a pattern_text field");
                }
            });
        }

        private final Parameter<String> indexOptions = Parameter.restrictedStringParam("index_options", false, m -> "docs", "docs");
        private final Parameter<Boolean> norms = refusedTrue("norms");
        private final Parameter<String> termVectors = Parameter.restrictedStringParam("term_vector", false, m -> "no", "no");
        private final Parameter<Boolean> indexPhrases = refusedTrue("index_phrases");
        private final Parameter<Object> indexPrefixes = new Parameter<Object>("index_prefixes", false, () -> null, (n, c, o) -> {
            if (o != null) {
                throw new MapperParsingException("[index_prefixes] cannot be set on a pattern_text field");
            }
            return null;
        }, m -> null).acceptsNull();

        public Builder(String name, Version indexCreatedVersion, IndexAnalyzers indexAnalyzers) {
            super(name, indexCreatedVersion, indexAnalyzers);
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            List<Parameter<?>> out = new ArrayList<>();
            for (Parameter<?> p : super.getParameters()) {
                if (REPLACED.contains(p.name) == false) {
                    out.add(p);
                }
            }
            out.addAll(List.of(indexOptions, norms, termVectors, indexPhrases, indexPrefixes));
            return List.copyOf(out);
        }

        @Override
        public PatternTextFieldMapper build(BuilderContext context) {
            FieldType fieldType = new FieldType(FIELD_TYPE);
            fieldType.setStored(store.getValue());
            if (index.getValue() == false) {
                fieldType.setIndexOptions(IndexOptions.NONE);
            }
            fieldType.freeze();
            // OpenSearch's match queries refuse phrases on a field whose search info has no positions
            // (exempting match_only_text by class), and text's builder refuses a
            // position_increment_gap without them; this field's phrases are PatternPhraseQuery, which
            // needs none, so its search info reports them.
            FieldType phrases = new FieldType(fieldType);
            phrases.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
            phrases.freeze();
            TextFieldType base = buildFieldType(phrases, context);
            PatternTextFieldType ft = new PatternTextFieldType(base.name(), base.isSearchable(), base.isStored(), base.getTextSearchInfo(), base.meta());
            ft.setIndexAnalyzer(base.indexAnalyzer());
            ft.setBoost(base.boost());
            ft.setEagerGlobalOrdinals(base.eagerGlobalOrdinals());
            if (base.fielddata()) {
                for (Parameter<?> p : super.getParameters()) {
                    if (p.name.equals("fielddata_frequency_filter")) {
                        ft.setFielddata(true, (FielddataFrequencyFilter) p.getValue());
                    }
                }
            }
            MultiFields multiFields = multiFieldsBuilder.build(this, context);
            for (org.opensearch.index.mapper.Mapper m : multiFields) {
                if (m.name().equals(base.name() + TEMPLATE_SUFFIX) || m.name().equals(base.name() + ARGS_SUFFIX)) {
                    throw new MapperParsingException(
                        "pattern_text field [" + base.name() + "] keeps its own [" + m.name() + "]; name the multi-field otherwise"
                    );
                }
            }
            return new PatternTextFieldMapper(name, fieldType, ft, multiFields, copyTo.build(), this);
        }
    }

    @Override
    public ParametrizedFieldMapper.Builder getMergeBuilder() {
        return new Builder(simpleName(), indexCreatedVersion, indexAnalyzers).init(this);
    }

    @Override
    protected String contentType() {
        return CONTENT_TYPE;
    }

    @Override
    public PatternTextFieldType fieldType() {
        return (PatternTextFieldType) super.fieldType();
    }

    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        String value = getFieldValue(context);
        if (value == null) {
            return;
        }
        String templateField = name() + TEMPLATE_SUFFIX;
        if (context.doc().getField(templateField) != null) {
            throw new IllegalArgumentException("pattern_text field [" + name() + "] takes one value per document");
        }
        PatternTextFormat.Split split = PatternTextFormat.split(value);
        byte[] template = split.template().getBytes(StandardCharsets.UTF_8);
        if (template.length > MAX_TEMPLATE_BYTES) {
            throw new IllegalArgumentException(
                "pattern_text field [" + name() + "]: the message's template is " + template.length + " bytes, over the "
                    + MAX_TEMPLATE_BYTES + "-byte doc-values limit"
            );
        }
        if (fieldType.indexOptions() != IndexOptions.NONE || fieldType.stored()) {
            context.doc().add(new Field(fieldType().name(), value, fieldType));
            createFieldNamesField(context);
        }
        context.doc().add(new SortedDocValuesField(templateField, new BytesRef(template)));
        if (split.args().isEmpty() == false) {
            context.doc().add(new StoredField(name() + ARGS_SUFFIX, PatternTextFormat.encodeArgs(split.args())));
        }
    }

    @Override
    protected void parseCreateFieldForPluggableFormat(ParseContext context) {
        throw new IllegalArgumentException("pattern_text field [" + name() + "] is not supported by pluggable data formats");
    }

    /** The message, rebuilt from the template and the variables. */
    @Override
    protected DerivedFieldGenerator derivedFieldGenerator() {
        return new DerivedFieldGenerator(mappedFieldType, new MessageFetcher(simpleName(), name()), null) {
            @Override
            public FieldValueType getDerivedFieldPreference() {
                return FieldValueType.DOC_VALUES;
            }
        };
    }

    /** One document's message, rebuilt from {@code f.template} and {@code f.args}. */
    public static String message(LeafReader reader, String field, int doc) throws IOException {
        SortedDocValues templates = reader.getSortedDocValues(field + TEMPLATE_SUFFIX);
        if (templates == null || templates.advanceExact(doc) == false) {
            return null;
        }
        String template = templates.lookupOrd(templates.ordValue()).utf8ToString();
        if (template.indexOf(PatternTextFormat.VARIABLE) < 0) {
            return template;
        }
        String argsField = field + ARGS_SUFFIX;
        byte[][] args = new byte[1][];
        reader.storedFields().document(doc, new StoredFieldVisitor() {
            @Override
            public Status needsField(FieldInfo fieldInfo) {
                if (args[0] != null) {
                    return Status.STOP;
                }
                return fieldInfo.name.equals(argsField) ? Status.YES : Status.NO;
            }

            @Override
            public void binaryField(FieldInfo fieldInfo, byte[] value) {
                args[0] = value;
            }
        });
        List<String> values = args[0] == null ? List.of() : PatternTextFormat.decodeArgs(args[0], 0, args[0].length);
        return PatternTextFormat.join(template, values);
    }

    private static final class MessageFetcher extends FieldValueFetcher {
        private final String field;

        MessageFetcher(String simpleName, String field) {
            super(simpleName);
            this.field = field;
        }

        @Override
        public List<Object> fetch(LeafReader reader, int docId) throws IOException {
            String m = message(reader, field, docId);
            return m == null ? List.of() : List.of(m);
        }
    }

    /** The field type: {@code match_only_text}'s term queries, and phrases as {@link PatternPhraseQuery}. */
    public static final class PatternTextFieldType extends TextFieldType {
        private volatile Boolean templateParts;

        public PatternTextFieldType(String name, boolean indexed, boolean stored, TextSearchInfo tsi, Map<String, String> meta) {
            super(name, indexed, stored, tsi, meta);
        }

        @Override
        public String typeName() {
            return CONTENT_TYPE;
        }

        @Override
        public Query termQuery(Object value, QueryShardContext context) {
            return new ConstantScoreQuery(super.termQuery(value, context));
        }

        @Override
        public Query termQueryCaseInsensitive(Object value, QueryShardContext context) {
            return new ConstantScoreQuery(super.termQueryCaseInsensitive(value, context));
        }

        @Override
        public Query phraseQuery(TokenStream stream, int slop, boolean enablePosIncrements, QueryShardContext context) throws IOException {
            failIfNotIndexed();
            return pattern(super.phraseQuery(stream, slop, enablePosIncrements), slop == 0);
        }

        @Override
        public Query multiPhraseQuery(TokenStream stream, int slop, boolean enablePositionIncrements, QueryShardContext context)
            throws IOException {
            failIfNotIndexed();
            return pattern(super.multiPhraseQuery(stream, slop, enablePositionIncrements), slop == 0);
        }

        @Override
        public org.apache.lucene.queries.intervals.IntervalsSource intervals(
            String text,
            int maxGaps,
            org.opensearch.index.query.IntervalMode mode,
            org.opensearch.index.analysis.NamedAnalyzer analyzer,
            boolean prefix
        ) {
            throw new IllegalArgumentException("Can only use interval queries on text fields with positions; [" + name() + "] is a pattern_text field");
        }

        @Override
        public org.apache.lucene.queries.spans.SpanQuery spanPrefixQuery(
            String value,
            org.apache.lucene.queries.spans.SpanMultiTermQueryWrapper.SpanRewriteMethod method,
            QueryShardContext context
        ) {
            throw new IllegalArgumentException("Can only use span queries on text fields with positions; [" + name() + "] is a pattern_text field");
        }

        @Override
        public Query phrasePrefixQuery(TokenStream stream, int slop, int maxExpansions, QueryShardContext context) throws IOException {
            failIfNotIndexed();
            CachingTokenFilter cached = new CachingTokenFilter(stream);
            List<List<BytesRef>> positions = new ArrayList<>();
            TermToBytesRefAttribute term = cached.getAttribute(TermToBytesRefAttribute.class);
            PositionIncrementAttribute inc = cached.getAttribute(PositionIncrementAttribute.class);
            cached.reset();
            while (cached.incrementToken()) {
                if (positions.isEmpty() || inc.getPositionIncrement() != 0) {
                    positions.add(new ArrayList<>());
                }
                positions.get(positions.size() - 1).add(BytesRef.deepCopyOf(term.getBytesRef()));
            }
            cached.end();
            Query positional = super.phrasePrefixQuery(cached, slop, maxExpansions);
            if (positions.isEmpty()) {
                return positional;
            }
            BooleanQuery.Builder approximation = new BooleanQuery.Builder();
            for (int i = 0; i < positions.size(); i++) {
                BooleanQuery.Builder any = new BooleanQuery.Builder();
                for (BytesRef t : positions.get(i)) {
                    Term tt = new Term(name(), t);
                    any.add(i == positions.size() - 1 ? new PrefixQuery(tt) : new TermQuery(tt), Occur.SHOULD);
                }
                approximation.add(any.build(), Occur.FILTER);
            }
            return new PatternPhraseQuery(name(), approximation.build(), positional, indexAnalyzer(), null, null, templateParts());
        }

        private Query pattern(Query positional, boolean exact) {
            return PatternPhraseQuery.of(name(), positional, indexAnalyzer(), exact, templateParts());
        }

        /**
         * Whether a template's static text may decide a phrase: the index analyzer's tokens of any run
         * of space-separated words appear, at the same relative positions, among the whole message's
         * ({@link PatternTextAnalysis#splitsOnSpaces}).
         */
        public boolean templateParts() {
            Boolean b = templateParts;
            if (b == null) {
                Analyzer a = indexAnalyzer();
                b = a != null && PatternTextAnalysis.splitsOnSpaces(a, name());
                templateParts = b;
            }
            return b;
        }
    }
}
