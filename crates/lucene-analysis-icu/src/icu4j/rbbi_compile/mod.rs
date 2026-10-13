//! `com.ibm.icu.text.RBBIRuleBuilder`: ICU's break rule compiler, which
//! `new RuleBasedBreakIterator(rules)` runs and Lucene's
//! `ICUTokenizerFactory` calls for each of its `rulefiles`. The rules' text is
//! scanned into syntax trees (`scanner`), their sets split into character
//! categories (`set_builder`), and the trees made into the state table the
//! break iterator runs (`table_builder`).
//!
//! Compiles what the iterator reads, without the byte-for-byte file ICU4J
//! serialises: the forward table, the category lookup and the rule-status
//! table. Not ported: the safe reverse table (only backward iteration reads
//! it), `optimizeTables` (merging equal columns and states: a smaller table
//! that matches the same), and the `!!reverse` and `!!safe_*` rule sets they
//! belong to (accepted, and ignored).

use crate::icu4j::rbbi_data::{CategoryTrie, RbbiData};
use crate::IcuError;

mod node;
mod parse_table;
mod scanner;
mod set_builder;
mod table_builder;

use node::Arena;
use scanner::ScannerState;
use set_builder::SetBuilderState;
use table_builder::TableBuilderState;

/// `RBBIRuleBuilder`: one compilation's state.
pub(super) struct RuleBuilder {
    /// `fRules`, as UTF-16 units: positions are Java's.
    rules: Vec<u16>,
    arena: Arena,
    /// `fTreeRoots`: the forward, reverse, safe-forward and safe-reverse
    /// rule trees.
    tree_roots: [Option<usize>; 4],
    /// `fDefaultTree`: where the rules after a `!!option` go.
    default_tree: usize,
    chain_rules: bool,
    look_ahead_hard_break: bool,
    /// `fUSetNodes`.
    usets: Vec<usize>,
    /// `fRuleStatusVals`.
    rule_status_vals: Vec<i32>,
    sc: ScannerState,
    sb: SetBuilderState,
    tb: TableBuilderState,
}

impl RuleBuilder {
    fn new(rules: &str) -> Result<RuleBuilder, IcuError> {
        Ok(RuleBuilder {
            rules: rules.encode_utf16().collect(),
            arena: Arena::default(),
            tree_roots: [None; 4],
            default_tree: 0,
            chain_rules: false,
            look_ahead_hard_break: false,
            usets: Vec::new(),
            rule_status_vals: Vec::new(),
            sc: ScannerState::new()?,
            sb: SetBuilderState::default(),
            tb: TableBuilderState::default(),
        })
    }
}

/// `RuleBasedBreakIterator.compileRules`, then
/// `RBBIDataWrapper.get`: the rules' text as break rule data.
///
/// An error is Java's `IllegalArgumentException("Error <code> at line <n>
/// column <n>")`.
pub fn compile_rules(rules: &str) -> Result<RbbiData, IcuError> {
    let mut b = RuleBuilder::new(rules)?;
    b.parse()?;
    b.build_ranges()?;
    let ftable = b.build_forward_table()?;
    Ok(RbbiData {
        cat_count: b.num_char_categories(),
        ftable,
        rtable: None,
        trie: CategoryTrie::Map(b.category_map()),
        rule_source: rules.to_string(),
        status_table: b.rule_status_vals,
    })
}
