//! `org.apache.lucene.queryparser.flexible.standard.parser.StandardSyntaxParserConstants`:
//! the token kinds and `tokenImage`. The output of
//! `crates/lucene-search/tools/gen_flexible.sh`: do not edit it by hand.

#![allow(dead_code)]

pub(crate) const EOF: i32 = 0;
pub(crate) const AND: i32 = 8;
pub(crate) const OR: i32 = 9;
pub(crate) const NOT: i32 = 10;
pub(crate) const FN_PREFIX: i32 = 11;
pub(crate) const PLUS: i32 = 12;
pub(crate) const MINUS: i32 = 13;
pub(crate) const RPAREN: i32 = 14;
pub(crate) const OP_COLON: i32 = 15;
pub(crate) const OP_EQUAL: i32 = 16;
pub(crate) const OP_LESSTHAN: i32 = 17;
pub(crate) const OP_LESSTHANEQ: i32 = 18;
pub(crate) const OP_MORETHAN: i32 = 19;
pub(crate) const OP_MORETHANEQ: i32 = 20;
pub(crate) const CARAT: i32 = 21;
pub(crate) const TILDE: i32 = 22;
pub(crate) const QUOTED: i32 = 23;
pub(crate) const NUMBER: i32 = 24;
pub(crate) const TERM: i32 = 25;
pub(crate) const REGEXPTERM: i32 = 26;
pub(crate) const RANGEIN_START: i32 = 27;
pub(crate) const RANGEEX_START: i32 = 28;
pub(crate) const LPAREN: i32 = 29;
pub(crate) const ATLEAST: i32 = 30;
pub(crate) const AFTER: i32 = 31;
pub(crate) const BEFORE: i32 = 32;
pub(crate) const CONTAINED_BY: i32 = 33;
pub(crate) const CONTAINING: i32 = 34;
pub(crate) const EXTEND: i32 = 35;
pub(crate) const FN_OR: i32 = 36;
pub(crate) const FUZZYTERM: i32 = 37;
pub(crate) const MAXGAPS: i32 = 38;
pub(crate) const MAXWIDTH: i32 = 39;
pub(crate) const NON_OVERLAPPING: i32 = 40;
pub(crate) const NOT_CONTAINED_BY: i32 = 41;
pub(crate) const NOT_CONTAINING: i32 = 42;
pub(crate) const NOT_WITHIN: i32 = 43;
pub(crate) const ORDERED: i32 = 44;
pub(crate) const OVERLAPPING: i32 = 45;
pub(crate) const PHRASE: i32 = 46;
pub(crate) const UNORDERED: i32 = 47;
pub(crate) const UNORDERED_NO_OVERLAPS: i32 = 48;
pub(crate) const WILDCARD: i32 = 49;
pub(crate) const WITHIN: i32 = 50;
pub(crate) const RANGE_TO: i32 = 51;
pub(crate) const RANGEIN_END: i32 = 52;
pub(crate) const RANGEEX_END: i32 = 53;
pub(crate) const RANGE_QUOTED: i32 = 54;
pub(crate) const RANGE_GOOP: i32 = 55;

/// `tokenImage`.
pub(crate) static TOKEN_IMAGE: [&str; 57] = [
    "<EOF>",
    "<_NUM_CHAR>",
    "<_ESCAPED_CHAR>",
    "<_TERM_START_CHAR>",
    "<_TERM_CHAR>",
    "<_WHITESPACE>",
    "<_QUOTED_CHAR>",
    "<token of kind 7>",
    "<AND>",
    "<OR>",
    "<NOT>",
    "\"fn:\"",
    "\"+\"",
    "\"-\"",
    "\")\"",
    "\":\"",
    "\"=\"",
    "\"<\"",
    "\"<=\"",
    "\">\"",
    "\">=\"",
    "\"^\"",
    "\"~\"",
    "<QUOTED>",
    "<NUMBER>",
    "<TERM>",
    "<REGEXPTERM>",
    "\"[\"",
    "\"{\"",
    "\"(\"",
    "<ATLEAST>",
    "\"after\"",
    "\"before\"",
    "<CONTAINED_BY>",
    "\"containing\"",
    "\"extend\"",
    "\"or\"",
    "<FUZZYTERM>",
    "<MAXGAPS>",
    "<MAXWIDTH>",
    "<NON_OVERLAPPING>",
    "<NOT_CONTAINED_BY>",
    "<NOT_CONTAINING>",
    "<NOT_WITHIN>",
    "\"ordered\"",
    "\"overlapping\"",
    "\"phrase\"",
    "\"unordered\"",
    "<UNORDERED_NO_OVERLAPS>",
    "\"wildcard\"",
    "\"within\"",
    "\"TO\"",
    "\"]\"",
    "\"}\"",
    "<RANGE_QUOTED>",
    "<RANGE_GOOP>",
    "\"@\"",
];
