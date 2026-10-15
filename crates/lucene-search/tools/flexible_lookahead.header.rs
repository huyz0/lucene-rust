//! The syntactic lookaheads of
//! `org.apache.lucene.queryparser.flexible.standard.parser.StandardSyntaxParser`
//! (`jj_3_<n>`, `jj_3R_*`) and its `jj_la1` masks, which scan tokens and build
//! nothing. The output of `crates/lucene-search/tools/gen_flexible.sh`: do not
//! edit it by hand.
//!
//! Not every line can run: a `LOOKAHEAD(2)` that matches both of its tokens
//! succeeds by unwinding (`LookaheadSuccess`, here `Err(Stop)`), so the
//! `Ok(false)` that ends a scan after its last token is dead -- in Lucene's
//! generated Java too.

#![allow(clippy::needless_return)]

use super::constants::*;
use super::token_manager::TokenManager;
use crate::queryparser::javacc::{Runtime, Stop};
