#![cfg(test)]
use super::*;
use crate::reader::StrReader;

fn strip(f: &mut HTMLStripCharFilter<StrReader>) -> (String, Vec<i32>) {
    let mut out = Vec::new();
    let mut buf = [0u16; 7];
    loop {
        let n = f.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
    }
    let corr = (0..=out.len() as i32)
        .map(|o| f.correct_offset(o))
        .collect();
    (String::from_utf16_lossy(&out), corr)
}

/// `(html, Lucene's output, Lucene's correctOffset(0..=len))`.
const CASES: &[(&str, &str, &[i32])] = &[
    (
        "<p>a &amp; b</p>",
        "\na & b\n",
        &[0, 3, 4, 5, 10, 11, 12, 16],
    ),
    ("x<!-- c -->y<br/>z", "xy\nz", &[0, 11, 12, 17, 18]),
    ("<script>var a = '<b>';</script>t", "\nt", &[0, 31, 32]),
    ("<style>p{}</style>u", "\nu", &[0, 18, 19]),
    (
        "&#65;&#x42;&#55357;&#56832;&#xD83D;&#xDE00;",
        "AB😀😀",
        &[0, 5, 11, 12, 27, 28, 43],
    ),
    (
        "&#55357;&#xDE00;&#xD83D;&#56832;",
        "😀😀",
        &[0, 1, 16, 17, 32],
    ),
    (
        "&#xD800;&#99999999;&COPY;&unknown;&lt",
        "\u{FFFD}&#99999999;©&unknown;<",
        &[
            0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 37,
        ],
    ),
    ("<![CDATA[ x<y ]]>", " x<y ", &[9, 10, 11, 12, 13, 17]),
    ("<!DOCTYPE html><?php echo 1 ?>q", "q", &[30, 31]),
    ("a <b>b</b> < c", "a b ", &[0, 1, 5, 10, 14]),
    ("<b>ab</b>c", "abc", &[3, 4, 9, 10]),
    ("a<!-- unterminated", "a", &[0, 18]),
    ("a&amp", "a&", &[0, 1, 5]),
    ("a&#3", "a\u{3}", &[0, 1, 4]),
    ("a<", "a<", &[0, 1, 2]),
    ("a<b c=\"x", "a<b c=\"x", &[0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ("a<!--#include virtual=\"x.html\" -->b", "ab", &[0, 34, 35]),
    ("<script>/* \"q\" 'r' */ x</script>c", "\nc", &[0, 32, 33]),
    (
        "<script><!-- s=\"a\" t='b' <!--#echo var=\"v\" --> --></script>d",
        "\nd",
        &[0, 59, 60],
    ),
    (
        "<style><!-- p{content:\"x\"} q{content:'y'} <!--#if expr=\"1\" --> --></style>e",
        "\ne",
        &[0, 74, 75],
    ),
    ("<!x[CDATA[ y ]]>f", "f", &[16, 17]),
    ("<!-- a <!--#include file=\"f\" --> b -->g", "g", &[38, 39]),
    ("<!ELEMENT note (#PCDATA)>h", "h", &[25, 26]),
    (
        "&#x110000;&#1114112;&#xFFFFFFF;&#12345678;i",
        "&#x110000;&#1114112;&#xFFFFFFF;&#12345678;i",
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        ],
    ),
    (
        "&#55357;&#56832x&#xD83D;&#xZZ;j",
        "\u{FFFD}&#56832x\u{FFFD}&#xZZ;j",
        &[
            0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    ),
    ("&#56832;&#55357;k", "\u{FFFD}\u{FFFD}k", &[0, 8, 16, 17]),
    ("<p class=\"a\" id='b'>l</p >", "\nl\n", &[0, 20, 21, 26]),
    (
        "</p>m</ div>n< /p>",
        "\nm\nn< /p>",
        &[0, 4, 5, 12, 13, 14, 15, 16, 17, 18],
    ),
    (
        "<a href=x>o</a><unknowntag>p</unknowntag>",
        "o\np\n",
        &[10, 15, 27, 28, 41],
    ),
    (
        "&COPY&gt&nbsp;q",
        "&COPY&gt q",
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15],
    ),
    ("<br><br/><BR>r</br>", "\n\n\nr\n", &[0, 4, 9, 13, 14, 19]),
    ("<?xml version=\"1.0\"?>s", "s", &[21, 22]),
    (
        "x &amp y &; z &#; w &#x;",
        "x &amp y &; z &#; w &#x;",
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24,
        ],
    ),
    (
        "<scrip>t</scrip><styl>u</styl>",
        "\nt\n\nu\n",
        &[0, 7, 8, 16, 22, 23, 30],
    ),
    ("<script>v", "", &[9]),
    ("<style>w", "", &[8]),
    ("<![CDATA[x", "x", &[9, 10]),
    ("<!--y", "", &[5]),
    ("<b c='z", "<b c='z", &[0, 1, 2, 3, 4, 5, 6, 7]),
    ("&eacute", "é", &[0, 7]),
    ("<p", "", &[2]),
    ("</b", "", &[3]),
    ("<!", "<!", &[0, 1, 2]),
    ("<!-", "<!-", &[0, 1, 2, 3]),
];

#[test]
fn strips_as_lucene_does() {
    for (html, out, corr) in CASES {
        let (s, c) = strip(&mut HTMLStripCharFilter::new(StrReader::new(*html)));
        assert_eq!(&s, out, "{html}");
        assert_eq!(&c, corr, "{html}");
    }
}

#[test]
fn entity_and_digit_parsing() {
    let u = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
    assert_eq!(entity_value(&u("amp")), Some(u16::from(b'&')));
    assert_eq!(entity_value(&u("thetasym")), Some(0x03D1));
    assert_eq!(entity_value(&u("nosuchentity")), None);
    assert_eq!(entity_value(&u("é")), None);
    assert_eq!(parse_digits(&u("65"), 10), Some(65));
    assert_eq!(parse_digits(&u("fF"), 16), Some(255));
    assert_eq!(parse_digits(&[], 10), None);
    assert_eq!(parse_digits(&u("1x"), 10), None);
    assert_eq!(parse_digits(&u("99999999999"), 10), None);
}

#[test]
fn escaped_tags_and_buffer_growth() {
    let mut f = HTMLStripCharFilter::with_escaped_tags(
        StrReader::new("<b>x</b><br><i>y</i><script>s</script>"),
        ["b", "BR", "script", "style"],
    );
    let (s, c) = strip(&mut f);
    assert_eq!(s, "<b>x</b><br>y<script></script>");
    assert_eq!(
        c,
        vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38
        ]
    );
    let mut f = HTMLStripCharFilter::with_escaped_tags(
        StrReader::new("<b>x</b><br>y</br><script>s</script><style>t</style>z<B>w</B>"),
        ["b", "BR", "script", "style"],
    );
    assert_eq!(
        strip(&mut f).0,
        "<b>x</b><br>y</br><script></script><style></style>z<B>w</B>"
    );
    for (html, out) in [("<script>s", "<script>"), ("<style>t", "<style>")] {
        let mut f =
            HTMLStripCharFilter::with_escaped_tags(StrReader::new(html), ["script", "style"]);
        assert_eq!(strip(&mut f).0, out);
    }
    // A tag name longer than the 16384-unit buffer grows it.
    let tag = format!("<{}>z", "a".repeat(20_000));
    assert_eq!(
        strip(&mut HTMLStripCharFilter::new(StrReader::new(tag))).0,
        "\nz"
    );
    let long = format!("<!--{}-->z", "x".repeat(40_000));
    assert_eq!(
        strip(&mut HTMLStripCharFilter::new(StrReader::new(long))).0,
        "z"
    );
}
