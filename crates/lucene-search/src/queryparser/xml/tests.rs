//! The reader's refusals and the builders' error paths, which the corpus
//! does not reach.

use super::*;
use lucene_analysis::StandardAnalyzer;

fn analyzer() -> Arc<Analyzer> {
    Arc::new(Analyzer::new(StandardAnalyzer::new()))
}

fn parser() -> CoreParser {
    CoreParser::new(Some("body"), analyzer())
}

fn message(xml: &str) -> String {
    parser().parse(xml).unwrap_err().message
}

#[test]
fn the_reader_builds_the_element_tree() {
    let root = dom::parse(
        "<?xml version=\"1.0\"?><!-- c --><?pi x?><r a=\"1\" b=\"&amp;\"> t<![CDATA[ c ]]>&#x41;&lt; <e/></r><!-- after -->\n",
    )
    .unwrap();
    assert_eq!(root.name, "r");
    assert_eq!(root.attribute("a"), "1");
    assert_eq!(root.attribute("b"), "&");
    assert_eq!(root.attribute("missing"), "");
    // CDATA is kept, but not part of the text
    assert_eq!(root.text(), " tA< ");
    assert!(root
        .children
        .iter()
        .any(|c| matches!(c, dom::Node::CData(t) if t == " c ")));
    assert_eq!(root.elements().count(), 1);
}

#[test]
fn a_document_must_have_one_root_and_nothing_but_space_after_it() {
    assert_eq!(dom::parse("").unwrap_err(), "Premature end of file.");
    assert_eq!(
        dom::parse("<a/><b/>").unwrap_err(),
        "The markup in the document following the root element must be well-formed."
    );
    assert_eq!(
        dom::parse("<a/>x").unwrap_err(),
        "content is not allowed in trailing section"
    );
    assert_eq!(
        dom::parse("<a/>&amp;").unwrap_err(),
        "content is not allowed in trailing section"
    );
    assert_eq!(
        dom::parse("<![CDATA[x]]><a/>").unwrap_err(),
        "content is not allowed in trailing section"
    );
    assert!(dom::parse("<a/> \n\t").is_ok());
    assert!(dom::parse("  <a/>").is_ok());
}

#[test]
fn entities_are_expanded_and_undeclared_ones_refused() {
    let root = dom::parse("<a>&#x41;&amp;&lt;&gt;&quot;&apos;</a>").unwrap();
    assert_eq!(root.text(), "A&<>\"'");
    assert_eq!(
        dom::parse("<a>&unknown;</a>").unwrap_err(),
        "The entity \"unknown\" was referenced, but not declared."
    );
    assert!(dom::parse("<a x=\"&bad;\"/>").is_err());
}

#[test]
fn a_document_type_declaration_is_refused() {
    assert_eq!(
        dom::parse("<!DOCTYPE a><a/>").unwrap_err(),
        "DOCTYPE is not supported"
    );
}

#[test]
fn malformed_xml_is_a_syntax_error() {
    assert!(message("<a>").starts_with("Error parsing XML stream"));
    assert!(message("</a>").starts_with("Error parsing XML stream"));
    assert!(message("<a></b>").starts_with("Error parsing XML stream"));
}

#[test]
fn a_user_query_uses_the_shared_parser_when_given_one() {
    let shared = CoreParser::with_parser(analyzer(), QueryParser::new("body", analyzer()));
    let q = shared
        .parse("<UserQuery fieldName=\"title\">quick</UserQuery>")
        .unwrap();
    // the shared parser's own field: `fieldName` is not read
    assert_eq!(q.describe("f"), "Boost(TermQuery(body:quick)^1.0)");
    // a parse error of the shared parser is a ParserException
    let e = shared.parse("<UserQuery>quick (</UserQuery>").unwrap_err();
    assert_eq!(e.class, PARSER_EXCEPTION);
    assert!(e.message.starts_with("Cannot parse 'quick ('"));
}

#[test]
fn a_span_builder_without_its_child_fails_as_java_does() {
    assert_eq!(
        message("<SpanFirst end=\"3\"/>"),
        "Cannot invoke \"org.w3c.dom.Element.getNodeName()\" because \"e\" is null"
    );
    let e = parser().parse("<SpanFirst end=\"3\"/>").unwrap_err();
    assert_eq!(e.class, "java.lang.NullPointerException");
    let e = parser()
        .parse("<SpanPositionRange start=\"0\" end=\"1\"/>")
        .unwrap_err();
    assert_eq!(e.class, "java.lang.NullPointerException");
    assert_eq!(
        message("<SpanNot><Exclude><SpanTerm fieldName=\"body\">a</SpanTerm></Exclude></SpanNot>"),
        "SpanNot missing \"Include\" child element"
    );
}

#[test]
fn span_or_terms_of_no_terms_is_an_empty_or() {
    let q = parser()
        .parse("<SpanOrTerms fieldName=\"body\">!!!</SpanOrTerms>")
        .unwrap();
    assert_eq!(q.to_string("body"), "spanOr([])");
}

#[test]
fn a_negative_minimum_should_match_is_refused_at_parse() {
    let e = parser()
        .parse("<BooleanQuery minimumNumberShouldMatch=\"-1\"/>")
        .unwrap_err();
    assert_eq!(e.class, "java.lang.IllegalArgumentException");
    assert_eq!(e.message, "minimumNumberShouldMatch must be >= 0");
}

#[test]
fn point_bounds_that_do_not_parse_are_parser_exceptions() {
    for ty in ["int", "long", "float", "double"] {
        let xml = format!("<PointRangeQuery fieldName=\"n\" type=\"{ty}\" upperTerm=\"zz\"/>");
        let e = parser().parse(&xml).unwrap_err();
        assert_eq!(e.class, PARSER_EXCEPTION, "{ty}");
        assert_eq!(
            e.message, "Could not parse lowerTerm or upperTerm into a number",
            "{ty}"
        );
    }
}

#[test]
fn a_float_point_range_has_infinite_defaults() {
    let q = parser()
        .parse("<PointRangeQuery fieldName=\"x\" type=\"float\" upperTerm=\"2\"/>")
        .unwrap();
    assert_eq!(q.to_string("f"), "x:[-Infinity TO 2.0]");
}

#[test]
fn java_exception_names_the_classes_the_parser_throws() {
    let mapped =
        |class: &'static str| java_exception(QueryParserError::runtime_class(class, "m")).class;
    assert_eq!(
        mapped("IllegalArgumentException"),
        "java.lang.IllegalArgumentException"
    );
    assert_eq!(
        mapped("NullPointerException"),
        "java.lang.NullPointerException"
    );
    assert_eq!(
        mapped("NumberFormatException"),
        "java.lang.NumberFormatException"
    );
    assert_eq!(
        mapped("TooManyClauses"),
        "org.apache.lucene.search.TooManyClauses"
    );
    assert_eq!(mapped("ParseException"), "ParseException");
}

#[test]
fn xml_1_0_characters_and_attribute_rules_are_enforced() {
    assert_eq!(
        dom::parse("<a>\u{1}</a>").unwrap_err(),
        "An invalid XML character (Unicode: 0x1) was found."
    );
    assert_eq!(
        dom::parse("<a>&#1;</a>").unwrap_err(),
        "An invalid XML character (Unicode: 0x1) was found."
    );
    assert_eq!(
        dom::parse("<a>\u{ffff}</a>").unwrap_err(),
        "An invalid XML character (Unicode: 0xffff) was found."
    );
    assert_eq!(
        dom::parse("<a x=\"a<b\"/>").unwrap_err(),
        "The value of an attribute must not contain the '<' character."
    );
    // a literal line end or tab in an attribute is a space; a reference stays
    let root = dom::parse("<a x=\"1\r\n2\t3\" y=\"&#9;\"/>").unwrap();
    assert_eq!(root.attribute("x"), "1 2 3");
    assert_eq!(root.attribute("y"), "\t");
    // text keeps XML 1.0 line ends: CR LF is LF, U+0085 is not a line end
    let root = dom::parse("<a>1\r\n2\u{85}3</a>").unwrap();
    assert_eq!(root.text(), "1\n2\u{85}3");
}
