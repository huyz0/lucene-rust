//! `RBBIRuleParseTable.gRuleParseStateTable`: the rule scanner's state table,
//! extracted row for row from ICU4J 77.1 (104 rows; the state names are
//! ICU's, as comments on the rows that start a state).
//!
//! A row is `(action, char class, next state, push state, next char)`. The
//! scanner enters a state at its first row and takes the first row whose
//! character class matches the current character: a literal below 127, or
//! `255` any, `254` quoted/escaped, `253` an escaped `P`/`p`, `252` end of
//! input, `128..240` the rule scanner's character sets (digit, name, name
//! start, rule character, white space).

/// One row of the scanner's state table.
#[derive(Clone, Copy)]
pub(super) struct Row {
    pub action: u8,
    pub char_class: u16,
    pub next_state: u16,
    pub push_state: u16,
    pub next_char: bool,
}

const fn r(action: u8, char_class: u16, next_state: u16, push_state: u16, next_char: bool) -> Row {
    Row {
        action,
        char_class,
        next_state,
        push_state,
        next_char,
    }
}

pub(super) const DO_CHECK_VAR_DEF: u8 = 1;
pub(super) const DO_DOT_ANY: u8 = 2;
pub(super) const DO_END_ASSIGN: u8 = 3;
pub(super) const DO_END_OF_RULE: u8 = 4;
pub(super) const DO_END_VARIABLE_NAME: u8 = 5;
pub(super) const DO_EXIT: u8 = 6;
pub(super) const DO_EXPR_CAT_OPERATOR: u8 = 7;
pub(super) const DO_EXPR_FINISHED: u8 = 8;
pub(super) const DO_EXPR_OR_OPERATOR: u8 = 9;
pub(super) const DO_EXPR_R_PAREN: u8 = 10;
pub(super) const DO_EXPR_START: u8 = 11;
pub(super) const DO_L_PAREN: u8 = 12;
pub(super) const DO_NOP: u8 = 13;
pub(super) const DO_NO_CHAIN: u8 = 14;
pub(super) const DO_OPTION_END: u8 = 15;
pub(super) const DO_OPTION_START: u8 = 16;
pub(super) const DO_REVERSE_DIR: u8 = 17;
pub(super) const DO_RULE_CHAR: u8 = 18;
pub(super) const DO_RULE_ERROR: u8 = 19;
pub(super) const DO_RULE_ERROR_ASSIGN_EXPR: u8 = 20;
pub(super) const DO_SCAN_UNICODE_SET: u8 = 21;
pub(super) const DO_SLASH: u8 = 22;
pub(super) const DO_START_ASSIGN: u8 = 23;
pub(super) const DO_START_TAG_VALUE: u8 = 24;
pub(super) const DO_START_VARIABLE_NAME: u8 = 25;
pub(super) const DO_TAG_DIGIT: u8 = 26;
pub(super) const DO_TAG_EXPECTED_ERROR: u8 = 27;
pub(super) const DO_TAG_VALUE: u8 = 28;
pub(super) const DO_UNARY_OP_PLUS: u8 = 29;
pub(super) const DO_UNARY_OP_QUESTION: u8 = 30;
pub(super) const DO_UNARY_OP_STAR: u8 = 31;
pub(super) const DO_VARIABLE_NAME_EXPECTED_ERR: u8 = 32;

pub(super) const TABLE: [Row; 104] = [
    r(13, 0, 0, 0, true),
    r(11, 254, 29, 9, false), // start
    r(13, 132, 1, 0, true),
    r(14, 94, 12, 9, true),
    r(11, 36, 88, 98, false),
    r(13, 33, 19, 0, true),
    r(13, 59, 1, 0, true),
    r(13, 252, 0, 0, false),
    r(11, 255, 29, 9, false),
    r(4, 59, 1, 0, true), // break-rule-end
    r(13, 132, 9, 0, true),
    r(19, 255, 103, 0, false),
    r(11, 254, 29, 0, false), // start-after-caret
    r(13, 132, 12, 0, true),
    r(19, 94, 103, 0, false),
    r(11, 36, 88, 37, false),
    r(19, 59, 103, 0, false),
    r(19, 252, 103, 0, false),
    r(11, 255, 29, 0, false),
    r(13, 33, 21, 0, true), // rev-option
    r(17, 255, 28, 9, false),
    r(16, 130, 23, 0, true), // option-scan1
    r(19, 255, 103, 0, false),
    r(13, 129, 23, 0, true), // option-scan2
    r(15, 255, 25, 0, false),
    r(13, 59, 1, 0, true), // option-scan3
    r(13, 132, 25, 0, true),
    r(19, 255, 103, 0, false),
    r(11, 255, 29, 9, false), // reverse-rule
    r(18, 254, 38, 0, true),  // term
    r(13, 132, 29, 0, true),
    r(18, 131, 38, 0, true),
    r(13, 91, 94, 38, false),
    r(12, 40, 29, 38, true),
    r(13, 36, 88, 37, false),
    r(2, 46, 38, 0, true),
    r(19, 255, 103, 0, false),
    r(1, 255, 38, 0, false), // term-var-ref
    r(13, 132, 38, 0, true), // expr-mod
    r(31, 42, 43, 0, true),
    r(29, 43, 43, 0, true),
    r(30, 63, 43, 0, true),
    r(13, 255, 43, 0, false),
    r(7, 254, 29, 0, false), // expr-cont
    r(13, 132, 43, 0, true),
    r(7, 131, 29, 0, false),
    r(7, 91, 29, 0, false),
    r(7, 40, 29, 0, false),
    r(7, 36, 29, 0, false),
    r(7, 46, 29, 0, false),
    r(7, 47, 55, 0, false),
    r(7, 123, 67, 0, true),
    r(9, 124, 29, 0, true),
    r(10, 41, 255, 0, true),
    r(8, 255, 255, 0, false),
    r(22, 47, 57, 0, true), // look-ahead
    r(13, 255, 103, 0, false),
    r(7, 254, 29, 0, false), // expr-cont-no-slash
    r(13, 132, 43, 0, true),
    r(7, 131, 29, 0, false),
    r(7, 91, 29, 0, false),
    r(7, 40, 29, 0, false),
    r(7, 36, 29, 0, false),
    r(7, 46, 29, 0, false),
    r(9, 124, 29, 0, true),
    r(10, 41, 255, 0, true),
    r(8, 255, 255, 0, false),
    r(13, 132, 67, 0, true), // tag-open
    r(24, 128, 70, 0, false),
    r(27, 255, 103, 0, false),
    r(13, 132, 74, 0, true), // tag-value
    r(13, 125, 74, 0, false),
    r(26, 128, 70, 0, true),
    r(27, 255, 103, 0, false),
    r(13, 132, 74, 0, true), // tag-close
    r(28, 125, 77, 0, true),
    r(27, 255, 103, 0, false),
    r(7, 254, 29, 0, false), // expr-cont-no-tag
    r(13, 132, 77, 0, true),
    r(7, 131, 29, 0, false),
    r(7, 91, 29, 0, false),
    r(7, 40, 29, 0, false),
    r(7, 36, 29, 0, false),
    r(7, 46, 29, 0, false),
    r(7, 47, 55, 0, false),
    r(9, 124, 29, 0, true),
    r(10, 41, 255, 0, true),
    r(8, 255, 255, 0, false),
    r(25, 36, 90, 0, true), // scan-var-name
    r(13, 255, 103, 0, false),
    r(13, 130, 92, 0, true), // scan-var-start
    r(32, 255, 103, 0, false),
    r(13, 129, 92, 0, true), // scan-var-body
    r(5, 255, 255, 0, false),
    r(21, 91, 255, 0, true), // scan-unicode-set
    r(21, 112, 255, 0, true),
    r(21, 80, 255, 0, true),
    r(13, 255, 103, 0, false),
    r(13, 132, 98, 0, true), // assign-or-rule
    r(23, 61, 29, 101, true),
    r(13, 255, 37, 9, false),
    r(3, 59, 1, 0, true), // assign-end
    r(20, 255, 103, 0, false),
    r(6, 255, 103, 0, true), // errorDeath
];
