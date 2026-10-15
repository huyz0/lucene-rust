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

/// `jj_la1_0`.
pub(super) static JJ_LA1_0: [u32; 34] = [
    0x3f803c00, 0x200, 0x100, 0x2400, 0x3400, 0x3400, 0x18000, 0x23800800, 0x3f800800, 0x200000,
    0x0, 0x3800800, 0x3800800, 0x3800800, 0x3800800, 0x3800800, 0x3000000, 0x3800000, 0x1000000,
    0x3000000, 0x3800000, 0x3000000, 0x3800000, 0x1e0000, 0x3800000, 0x3000000, 0x400000,
    0x1f800000, 0x200000, 0x400000, 0x18000000, 0x0, 0x0, 0x0,
];

/// `jj_la1_1`.
pub(super) static JJ_LA1_1: [u32; 34] = [
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1000000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc80000, 0xc80000, 0x300000,
];

/// The number of `jj_2_<n>` lookaheads.
pub(super) const NUM_LOOKAHEADS: usize = 28;

/// `jj_3_<n + 1>()`: `Ok(true)` when the lookahead fails.
pub(super) fn jj_3(rt: &mut Runtime<TokenManager>, n: usize) -> Result<bool, Stop> {
    match n {
        0 => jj_3_1(rt),
        1 => jj_3_2(rt),
        2 => jj_3_3(rt),
        3 => jj_3_4(rt),
        4 => jj_3_5(rt),
        5 => jj_3_6(rt),
        6 => jj_3_7(rt),
        7 => jj_3_8(rt),
        8 => jj_3_9(rt),
        9 => jj_3_10(rt),
        10 => jj_3_11(rt),
        11 => jj_3_12(rt),
        12 => jj_3_13(rt),
        13 => jj_3_14(rt),
        14 => jj_3_15(rt),
        15 => jj_3_16(rt),
        16 => jj_3_17(rt),
        17 => jj_3_18(rt),
        18 => jj_3_19(rt),
        19 => jj_3_20(rt),
        20 => jj_3_21(rt),
        21 => jj_3_22(rt),
        22 => jj_3_23(rt),
        23 => jj_3_24(rt),
        24 => jj_3_25(rt),
        25 => jj_3_26(rt),
        26 => jj_3_27(rt),
        _ => jj_3_28(rt),
    }
}

/// `jj_3_1()`.
fn jj_3_1(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_field_name_344_3_9(rt)? {
        return Ok(true);
    }
    let xsp = rt.jj_scanpos;
    if rt.scan_token(15)? {
        rt.jj_scanpos = xsp;
        if rt.scan_token(16)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `jj_3_2()`.
fn jj_3_2(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_term_798_3_10(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_3()`.
fn jj_3_3(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_field_range_expr_744_3_11(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_4()`.
fn jj_3_4(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_at_least_418_3_12(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_5()`.
fn jj_3_5(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_max_width_430_3_13(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_6()`.
fn jj_3_6(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_max_gaps_442_3_14(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_7()`.
fn jj_3_7(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_ordered_477_3_15(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_8()`.
fn jj_3_8(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_unordered_454_3_16(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_9()`.
fn jj_3_9(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_unordered_no_overlaps_465_3_17(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_10()`.
fn jj_3_10(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_or_489_3_18(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_11()`.
fn jj_3_11(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_wildcard_647_3_19(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_12()`.
fn jj_3_12(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_after_524_3_20(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_13()`.
fn jj_3_13(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_before_513_3_21(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_14()`.
fn jj_3_14(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_phrase_501_3_22(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_15()`.
fn jj_3_15(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_containing_535_3_23(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_16()`.
fn jj_3_16(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_not_containing_546_3_24(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_17()`.
fn jj_3_17(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_contained_by_557_3_25(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_18()`.
fn jj_3_18(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_not_contained_by_568_3_26(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_19()`.
fn jj_3_19(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_within_579_3_27(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_20()`.
fn jj_3_20(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_not_within_611_3_28(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_21()`.
fn jj_3_21(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_overlapping_626_3_29(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_22()`.
fn jj_3_22(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_non_overlapping_636_3_30(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_23()`.
fn jj_3_23(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_extend_595_3_31(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_24()`.
fn jj_3_24(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_fuzzy_term_666_3_32(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_25()`.
fn jj_3_25(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_interval_text_685_5_33(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_26()`.
fn jj_3_26(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(NUMBER)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_27()`.
fn jj_3_27(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(NUMBER)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3_28()`.
fn jj_3_28(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(NUMBER)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_Boost_700_3_44()`.
fn jj_3r_boost_700_3_44(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(CARAT)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_FieldName_344_3_9()`.
fn jj_3r_field_name_344_3_9(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(TERM)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_FieldRangeExpr_744_3_11()`.
fn jj_3r_field_range_expr_744_3_11(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_field_name_344_3_9(rt)? {
        return Ok(true);
    }
    let xsp = rt.jj_scanpos;
    if rt.scan_token(17)? {
        rt.jj_scanpos = xsp;
        if rt.scan_token(18)? {
            rt.jj_scanpos = xsp;
            if rt.scan_token(19)? {
                rt.jj_scanpos = xsp;
                if rt.scan_token(20)? {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// `jj_3R_FuzzyOp_717_3_45()`.
fn jj_3r_fuzzy_op_717_3_45(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(TILDE)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalAfter_524_3_20()`.
fn jj_3r_interval_after_524_3_20(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(AFTER)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalAtLeast_418_3_12()`.
fn jj_3r_interval_at_least_418_3_12(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(ATLEAST)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalBefore_513_3_21()`.
fn jj_3r_interval_before_513_3_21(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(BEFORE)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalContainedBy_557_3_25()`.
fn jj_3r_interval_contained_by_557_3_25(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(CONTAINED_BY)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalContaining_535_3_23()`.
fn jj_3r_interval_containing_535_3_23(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(CONTAINING)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalExtend_595_3_31()`.
fn jj_3r_interval_extend_595_3_31(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(EXTEND)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalFuzzyTerm_666_3_32()`.
fn jj_3r_interval_fuzzy_term_666_3_32(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(FUZZYTERM)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalMaxGaps_442_3_14()`.
fn jj_3r_interval_max_gaps_442_3_14(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(MAXGAPS)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalMaxWidth_430_3_13()`.
fn jj_3r_interval_max_width_430_3_13(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(MAXWIDTH)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalNonOverlapping_636_3_30()`.
fn jj_3r_interval_non_overlapping_636_3_30(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(NON_OVERLAPPING)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalNotContainedBy_568_3_26()`.
fn jj_3r_interval_not_contained_by_568_3_26(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(NOT_CONTAINED_BY)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalNotContaining_546_3_24()`.
fn jj_3r_interval_not_containing_546_3_24(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(NOT_CONTAINING)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalNotWithin_611_3_28()`.
fn jj_3r_interval_not_within_611_3_28(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(NOT_WITHIN)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalOr_489_3_18()`.
fn jj_3r_interval_or_489_3_18(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(FN_OR)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalOrdered_477_3_15()`.
fn jj_3r_interval_ordered_477_3_15(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(ORDERED)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalOverlapping_626_3_29()`.
fn jj_3r_interval_overlapping_626_3_29(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(OVERLAPPING)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalPhrase_501_3_22()`.
fn jj_3r_interval_phrase_501_3_22(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(PHRASE)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalText_685_5_33()`.
fn jj_3r_interval_text_685_5_33(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    let xsp = rt.jj_scanpos;
    if jj_3r_interval_text_685_5_39(rt)? {
        rt.jj_scanpos = xsp;
        if jj_3r_interval_text_686_5_40(rt)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `jj_3R_IntervalText_685_5_39()`.
fn jj_3r_interval_text_685_5_39(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(QUOTED)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalText_686_5_40()`.
fn jj_3r_interval_text_686_5_40(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    let xsp = rt.jj_scanpos;
    if rt.scan_token(25)? {
        rt.jj_scanpos = xsp;
        if rt.scan_token(24)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `jj_3R_IntervalUnorderedNoOverlaps_465_3_17()`.
fn jj_3r_interval_unordered_no_overlaps_465_3_17(
    rt: &mut Runtime<TokenManager>,
) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(UNORDERED_NO_OVERLAPS)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalUnordered_454_3_16()`.
fn jj_3r_interval_unordered_454_3_16(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(UNORDERED)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalWildcard_647_3_19()`.
fn jj_3r_interval_wildcard_647_3_19(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(WILDCARD)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_IntervalWithin_579_3_27()`.
fn jj_3r_interval_within_579_3_27(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(FN_PREFIX)? {
        return Ok(true);
    }
    if rt.scan_token(WITHIN)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_QuotedTerm_829_3_43()`.
fn jj_3r_quoted_term_829_3_43(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(QUOTED)? {
        return Ok(true);
    }
    let xsp = rt.jj_scanpos;
    if jj_3r_quoted_term_834_5_47(rt)? {
        rt.jj_scanpos = xsp;
    }
    Ok(false)
}

/// `jj_3R_QuotedTerm_834_5_47()`.
fn jj_3r_quoted_term_834_5_47(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(TILDE)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_TermRangeExpr_855_3_42()`.
fn jj_3r_term_range_expr_855_3_42(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    let xsp = rt.jj_scanpos;
    if jj_3r_term_range_expr_856_6_46(rt)? {
        rt.jj_scanpos = xsp;
        if rt.scan_token(28)? {
            return Ok(true);
        }
    }
    let xsp = rt.jj_scanpos;
    if rt.scan_token(55)? {
        rt.jj_scanpos = xsp;
        if rt.scan_token(54)? {
            rt.jj_scanpos = xsp;
            if rt.scan_token(51)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// `jj_3R_TermRangeExpr_856_6_46()`.
fn jj_3r_term_range_expr_856_6_46(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(RANGEIN_START)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_Term_798_3_10()`.
fn jj_3r_term_798_3_10(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    let xsp = rt.jj_scanpos;
    if jj_3r_term_799_8_34(rt)? {
        rt.jj_scanpos = xsp;
        if jj_3r_term_804_8_35(rt)? {
            rt.jj_scanpos = xsp;
            if jj_3r_term_807_8_36(rt)? {
                rt.jj_scanpos = xsp;
                if jj_3r_term_808_8_37(rt)? {
                    return Ok(true);
                }
            }
        }
    }
    let xsp = rt.jj_scanpos;
    if jj_3r_term_810_5_38(rt)? {
        rt.jj_scanpos = xsp;
    }
    Ok(false)
}

/// `jj_3R_Term_799_8_34()`.
fn jj_3r_term_799_8_34(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if rt.scan_token(REGEXPTERM)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_Term_804_8_35()`.
fn jj_3r_term_804_8_35(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    let xsp = rt.jj_scanpos;
    if rt.scan_token(25)? {
        rt.jj_scanpos = xsp;
        if rt.scan_token(24)? {
            return Ok(true);
        }
    }
    let xsp = rt.jj_scanpos;
    if jj_3r_term_806_10_41(rt)? {
        rt.jj_scanpos = xsp;
    }
    Ok(false)
}

/// `jj_3R_Term_806_10_41()`.
fn jj_3r_term_806_10_41(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_fuzzy_op_717_3_45(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_Term_807_8_36()`.
fn jj_3r_term_807_8_36(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_term_range_expr_855_3_42(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_Term_808_8_37()`.
fn jj_3r_term_808_8_37(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_quoted_term_829_3_43(rt)? {
        return Ok(true);
    }
    Ok(false)
}

/// `jj_3R_Term_810_5_38()`.
fn jj_3r_term_810_5_38(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {
    if jj_3r_boost_700_3_44(rt)? {
        return Ok(true);
    }
    Ok(false)
}
