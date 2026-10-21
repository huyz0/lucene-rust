//! `English`: numbers spelled out, the text many Lucene tests index
//! (`English.intToEnglish(i)` is the body of document `i`).
//!
//! Output is Java's, separators and trailing spaces included
//! (`intToEnglish(21)` is `"twenty-one "`).

/// `English.longToEnglish(i)`.
pub fn long_to_english(i: i64) -> String {
    let mut result = String::new();
    long_to_english_into(i, &mut result);
    result
}

/// `English.intToEnglish(i)`.
pub fn int_to_english(i: i32) -> String {
    long_to_english(i64::from(i))
}

/// `English.longToEnglish(i, result)`.
///
/// `Long.MIN_VALUE` negates to itself in Java and then matches none of the
/// magnitude tests, giving `"minus "`; the same here.
pub fn long_to_english_into(mut i: i64, result: &mut String) {
    if i == 0 {
        result.push_str("zero");
        return;
    }
    if i < 0 {
        result.push_str("minus ");
        i = i.wrapping_neg();
    }
    for (scale, name) in [
        (1_000_000_000_000_000_000, "quintillion, "),
        (1_000_000_000_000_000, "quadrillion, "),
        (1_000_000_000_000, "trillion, "),
        (1_000_000_000, "billion, "),
        (1_000_000, "million, "),
        (1_000, "thousand, "),
        (100, "hundred "),
    ] {
        if i >= scale {
            long_to_english_into(i / scale, result);
            result.push_str(name);
            i %= scale;
        }
    }
    if i >= 20 {
        result.push_str(match i / 10 {
            9 => "ninety",
            8 => "eighty",
            7 => "seventy",
            6 => "sixty",
            5 => "fifty",
            4 => "forty",
            3 => "thirty",
            _ => "twenty",
        });
        i %= 10;
        result.push(if i == 0 { ' ' } else { '-' });
    }
    const SMALL: [&str; 20] = [
        "",
        "one ",
        "two ",
        "three ",
        "four ",
        "five ",
        "six ",
        "seven ",
        "eight ",
        "nine ",
        "ten ",
        "eleven ",
        "twelve ",
        "thirteen ",
        "fourteen ",
        "fifteen ",
        "sixteen ",
        "seventeen ",
        "eighteen ",
        "nineteen ",
    ];
    // `Long.MIN_VALUE` is still negative here; Java's `switch` matches nothing.
    if let Some(s) = usize::try_from(i).ok().and_then(|i| SMALL.get(i)) {
        result.push_str(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spells_numbers_as_java_does() {
        assert_eq!(int_to_english(0), "zero");
        assert_eq!(int_to_english(7), "seven ");
        assert_eq!(int_to_english(20), "twenty ");
        assert_eq!(int_to_english(21), "twenty-one ");
        assert_eq!(int_to_english(-115), "minus one hundred fifteen ");
        assert_eq!(int_to_english(1_000_001), "one million, one ");
        assert_eq!(
            long_to_english(i64::MAX),
            "nine quintillion, two hundred twenty-three quadrillion, three hundred seventy-two \
             trillion, thirty-six billion, eight hundred fifty-four million, seven hundred \
             seventy-five thousand, eight hundred seven "
        );
        assert_eq!(long_to_english(i64::MIN), "minus ");
        assert_eq!(int_to_english(90), "ninety ");
        for (n, w) in [
            (30, "thirty "),
            (40, "forty "),
            (50, "fifty "),
            (60, "sixty "),
            (70, "seventy "),
            (80, "eighty "),
        ] {
            assert_eq!(int_to_english(n), w);
        }
    }
}
