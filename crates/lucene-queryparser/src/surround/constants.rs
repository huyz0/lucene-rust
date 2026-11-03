//! The surround `QueryParserConstants`: token kinds, lexical states and
//! `tokenImage`.

pub(crate) const EOF: i32 = 0;
pub(crate) const OR: i32 = 8;
pub(crate) const AND: i32 = 9;
pub(crate) const NOT: i32 = 10;
pub(crate) const W: i32 = 11;
pub(crate) const N: i32 = 12;
pub(crate) const LPAREN: i32 = 13;
pub(crate) const RPAREN: i32 = 14;
pub(crate) const COMMA: i32 = 15;
pub(crate) const COLON: i32 = 16;
pub(crate) const CARAT: i32 = 17;
pub(crate) const TRUNCQUOTED: i32 = 18;
pub(crate) const QUOTED: i32 = 19;
pub(crate) const SUFFIXTERM: i32 = 20;
pub(crate) const TRUNCTERM: i32 = 21;
pub(crate) const TERM: i32 = 22;
pub(crate) const NUMBER: i32 = 23;

pub(crate) const BOOST: i32 = 0;
pub(crate) const DEFAULT: i32 = 1;

pub(crate) static TOKEN_IMAGE: [&str; 24] = [
    "<EOF>",
    "<_NUM_CHAR>",
    "<_TERM_CHAR>",
    "<_WHITESPACE>",
    "\"*\"",
    "\"?\"",
    "<_DISTOP_NUM>",
    "<token of kind 7>",
    "<OR>",
    "<AND>",
    "<NOT>",
    "<W>",
    "<N>",
    "\"(\"",
    "\")\"",
    "\",\"",
    "\":\"",
    "\"^\"",
    "<TRUNCQUOTED>",
    "<QUOTED>",
    "<SUFFIXTERM>",
    "<TRUNCTERM>",
    "<TERM>",
    "<NUMBER>",
];
