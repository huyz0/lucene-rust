//! `org.apache.lucene.analysis.util.CSVUtil`: the minimal CSV dialect of
//! Kuromoji/Nori user dictionaries.

/// `CSVUtil.parse(String)`: fields split at commas outside quotes; a quoted
/// field is unquoted and its `""` unescaped -- except the last field, which
/// Lucene adds as read. An odd number of quotes yields no fields.
pub fn parse(line: &str) -> Vec<String> {
    let mut inside_quote = false;
    let mut result = Vec::new();
    let mut quote_count = 0usize;
    let mut sb = String::new();
    for c in line.chars() {
        if c == '"' {
            inside_quote = !inside_quote;
            quote_count += 1;
        }
        if c == ',' && !inside_quote {
            result.push(un_quote_un_escape(&sb));
            sb.clear();
            continue;
        }
        sb.push(c);
    }
    result.push(sb);
    if !quote_count.is_multiple_of(2) {
        return Vec::new();
    }
    result
}

// Java: CSVUtil.unQuoteUnEscape
fn un_quote_un_escape(original: &str) -> String {
    let mut result = original.to_string();
    if result.contains('"') {
        // `^"([^"]+)"$`
        if original.len() >= 3
            && original.starts_with('"')
            && original.ends_with('"')
            && !original[1..original.len() - 1].contains('"')
        {
            result = original[1..original.len() - 1].to_string();
        }
        if result.contains("\"\"") {
            result = result.replace("\"\"", "\"");
        }
    }
    result
}

/// `CSVUtil.quoteEscape(String)`.
pub fn quote_escape(original: &str) -> String {
    let mut result = original.to_string();
    if result.contains('"') {
        result = result.replace('"', "\"\"");
    }
    if result.contains(',') {
        result = format!("\"{result}\"");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values are Lucene 10.5.0's (TestCSVUtil and a few more).
    #[test]
    fn parse_and_escape_like_lucene() {
        assert_eq!(parse("a,b,c"), ["a", "b", "c"]);
        assert_eq!(parse("\"a,b\",c"), ["a,b", "c"]);
        assert_eq!(parse("\"x\"\"y\",z"), ["\"x\"y\"", "z"]);
        assert_eq!(parse("a,\"b,c\""), ["a", "\"b,c\""]);
        assert_eq!(parse("\"a,b"), Vec::<String>::new());
        assert_eq!(parse(""), [""]);
        assert_eq!(parse("\"\",x"), ["\"", "x"]);
        assert_eq!(quote_escape("a,b"), "\"a,b\"");
        assert_eq!(quote_escape("a\"b"), "a\"\"b");
        assert_eq!(quote_escape("a\",b"), "\"a\"\",b\"");
        assert_eq!(quote_escape("plain"), "plain");
        for s in ["a,b", "x\"y", "p"] {
            assert_eq!(parse(&format!("{},z", quote_escape(s)))[0], s);
        }
    }
}
