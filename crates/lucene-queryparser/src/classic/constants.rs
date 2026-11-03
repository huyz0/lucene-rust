//! `org.apache.lucene.queryparser.classic.QueryParserConstants`: the token
//! kinds and lexical states of `QueryParser.jj`, and `tokenImage`, each
//! kind's name as a parse error prints it.

pub(crate) const EOF: i32 = 0;
pub(crate) const AND: i32 = 8;
pub(crate) const OR: i32 = 9;
pub(crate) const NOT: i32 = 10;
pub(crate) const PLUS: i32 = 11;
pub(crate) const MINUS: i32 = 12;
pub(crate) const BAREOPER: i32 = 13;
pub(crate) const LPAREN: i32 = 14;
pub(crate) const RPAREN: i32 = 15;
pub(crate) const COLON: i32 = 16;
pub(crate) const STAR: i32 = 17;
pub(crate) const CARAT: i32 = 18;
pub(crate) const QUOTED: i32 = 19;
pub(crate) const TERM: i32 = 20;
pub(crate) const FUZZY_SLOP: i32 = 21;
pub(crate) const PREFIXTERM: i32 = 22;
pub(crate) const WILDTERM: i32 = 23;
pub(crate) const REGEXPTERM: i32 = 24;
pub(crate) const RANGEIN_START: i32 = 25;
pub(crate) const RANGEEX_START: i32 = 26;
pub(crate) const NUMBER: i32 = 27;
pub(crate) const RANGE_TO: i32 = 28;
pub(crate) const RANGEIN_END: i32 = 29;
pub(crate) const RANGEEX_END: i32 = 30;
pub(crate) const RANGE_QUOTED: i32 = 31;
pub(crate) const RANGE_GOOP: i32 = 32;

pub(crate) const BOOST: i32 = 0;
pub(crate) const RANGE: i32 = 1;
pub(crate) const DEFAULT: i32 = 2;

pub(crate) static TOKEN_IMAGE: [&str; 33] = [
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
    "\"+\"",
    "\"-\"",
    "<BAREOPER>",
    "\"(\"",
    "\")\"",
    "\":\"",
    "\"*\"",
    "\"^\"",
    "<QUOTED>",
    "<TERM>",
    "<FUZZY_SLOP>",
    "<PREFIXTERM>",
    "<WILDTERM>",
    "<REGEXPTERM>",
    "\"[\"",
    "\"{\"",
    "<NUMBER>",
    "\"TO\"",
    "\"]\"",
    "\"}\"",
    "<RANGE_QUOTED>",
    "<RANGE_GOOP>",
];
