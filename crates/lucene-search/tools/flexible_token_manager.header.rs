//! `org.apache.lucene.queryparser.flexible.standard.parser.StandardSyntaxParserTokenManager`
//! (the JavaCC 7 lexer generated from `StandardSyntaxParser.jj`), state for
//! state. This file is the output of `crates/lucene-search/tools/gen_flexible.sh`
//! (`javacc_lexer.py` over the generated Java): do not edit it by hand. The
//! three lexical states are `Function` (0, after `fn:`), `Range` (1, inside
//! `[`/`{`) and `DEFAULT` (2). NFA arms of states nothing can enter are left
//! out, which changes nothing. Some kept transitions no query reaches (a
//! keyword DFA level's stop state for a kind that cannot stop there), as in
//! Lucene's generated Java, so the file is not fully covered.

// The generated code's shape is kept: one arm per NFA state, each case's
// tests inside it, the string-literal DFA's returns as written.
#![allow(
    clippy::collapsible_match,
    clippy::collapsible_else_if,
    clippy::collapsible_if,
    clippy::needless_return,
    clippy::match_single_binding,
    clippy::single_match,
    clippy::if_same_then_else
)]

use std::ops::{Deref, DerefMut};

use crate::queryparser::javacc::{CharStream, Dfa, Lexer, NfaCore, Token, TokenMgrError, NO_KIND};
