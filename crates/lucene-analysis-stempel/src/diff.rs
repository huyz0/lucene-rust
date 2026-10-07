//! `org.egothor.stemmer.Diff.apply`: a stemmer command run on a word, from
//! its end (Egothor Software License 1.00, see [`crate::egothor`]).
//!
//! A command is pairs of `(op, param)` over UTF-16 units, applied from the
//! word's last character backwards: `-` skips `param - 'a' + 1` characters,
//! `R` replaces the current one with `param`, `D` deletes `param - 'a' + 1`
//! ending at the current one, `I` inserts `param` after it. Java stops at
//! the first edit that would index outside the word (its `catch` of
//! `StringIndexOutOfBoundsException`), keeping the edits made so far; so
//! does the port.
//!
//! `Diff.exec`, which builds commands from word pairs, belongs to table
//! compilation, which M12 does not port.

/// `Diff.apply(dest, diff)`.
pub fn apply(dest: &mut Vec<u16>, diff: &[u16]) {
    let Some(last) = dest.len().checked_sub(1) else {
        return;
    };
    // Positions are Java `int`s that may go negative; a command is at most
    // 65,535 units, so i64 holds every value reachable from a word's length.
    let Ok(mut pos) = i64::try_from(last) else {
        return;
    };
    for pair in diff.chunks_exact(2) {
        let (cmd, param) = (pair[0], pair[1]);
        // ARITH: param is a u16 and the edits move pos by at most 65,536
        // per pair over at most 32,768 pairs, far inside i64.
        #[allow(clippy::arithmetic_side_effects)]
        {
            let par_num = i64::from(param) - i64::from(b'a') + 1;
            let len = i64::try_from(dest.len()).unwrap_or(i64::MAX);
            match u8::try_from(cmd).unwrap_or(0) {
                b'-' => pos = pos - par_num + 1,
                b'R' => match usize::try_from(pos).ok().filter(|&p| p < dest.len()) {
                    Some(p) => dest[p] = param,
                    None => return,
                },
                b'D' => {
                    let o = pos;
                    pos -= par_num - 1;
                    // StringBuilder.delete(pos, o + 1): the end clamped to
                    // the length, then start <= end required.
                    let end = (o + 1).min(len);
                    if pos < 0 || pos > end {
                        return;
                    }
                    // Both in 0..=len by the checks above.
                    let (s, e) = (pos as usize, end as usize);
                    dest.drain(s..e);
                }
                b'I' => {
                    pos += 1;
                    if pos < 0 || pos > len {
                        return;
                    }
                    dest.insert(pos as usize, param);
                }
                _ => {}
            }
            pos -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(word: &str, diff: &str) -> String {
        let mut w: Vec<u16> = word.encode_utf16().collect();
        apply(&mut w, &diff.encode_utf16().collect::<Vec<_>>());
        String::from_utf16(&w).unwrap()
    }

    #[test]
    fn edits_from_the_end() {
        assert_eq!(run("kotami", "Db"), "kota");
        assert_eq!(run("kotami", "-bRy"), "kotymi");
        assert_eq!(run("kot", "Ia"), "kota");
        assert_eq!(run("psa", "RyIe"), "psey");
        assert_eq!(run("abc", ""), "abc");
        assert_eq!(run("abc", "Q"), "abc");
        assert_eq!(run("abc", "Za"), "abc");
        assert_eq!(run("", "Db"), "");
    }

    #[test]
    fn stops_where_java_throws() {
        // Replacing before the start: nothing further happens.
        assert_eq!(run("ab", "-cRxRy"), "ab");
        // A deletion running past the start.
        assert_eq!(run("ab", "DdRx"), "ab");
        // A deletion whose start passes its end (a negative count).
        assert_eq!(run("ab", "-`D_"), "ab");
        // Skip forward past the end, then delete: the end is clamped.
        assert_eq!(run("abc", "-_Da"), "abc");
        // Insert before the start.
        assert_eq!(run("ab", "-dIx"), "ab");
        // Insert past the end.
        assert_eq!(run("ab", "-_Ix"), "ab");
        // Edits before the failing one stay.
        assert_eq!(run("abc", "RxDdRy"), "abx");
    }
}
