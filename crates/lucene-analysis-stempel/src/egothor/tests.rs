#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk
use super::*;

/// A test-only `DataOutput`.
#[derive(Default)]
pub(crate) struct Out(pub Vec<u8>);

impl Out {
    pub fn boolean(&mut self, b: bool) -> &mut Self {
        self.0.push(u8::from(b));
        self
    }
    pub fn int(&mut self, v: i32) -> &mut Self {
        self.0.extend_from_slice(&v.to_be_bytes());
        self
    }
    pub fn char(&mut self, c: char) -> &mut Self {
        self.0.extend_from_slice(&(c as u16).to_be_bytes());
        self
    }
    pub fn utf(&mut self, s: &str) -> &mut Self {
        let b = s.as_bytes();
        self.0.extend_from_slice(&(b.len() as u16).to_be_bytes());
        self.0.extend_from_slice(b);
        self
    }
    /// A trie: `forward`, `root`, `cmds`, rows of `(char, cmd, ref)`.
    pub fn trie(
        &mut self,
        forward: bool,
        root: i32,
        cmds: &[&str],
        rows: &[&[(char, i32, i32)]],
    ) -> &mut Self {
        self.boolean(forward).int(root).int(cmds.len() as i32);
        for c in cmds {
            self.utf(c);
        }
        self.int(rows.len() as i32);
        for row in rows {
            self.int(row.len() as i32);
            for &(ch, cmd, r) in *row {
                self.char(ch).int(cmd).int(1).int(r).int(0);
            }
        }
        self
    }
}

fn u(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

#[test]
fn reads_data_input() {
    let mut o = Out::default();
    o.boolean(true).int(-2).char('ż').utf("aż\u{0}");
    let mut i = DataInput::new(&o.0);
    assert!(i.read_boolean().unwrap());
    assert_eq!(i.read_int().unwrap(), -2);
    assert_eq!(i.read_char().unwrap(), 'ż' as u16);
    // Rust's UTF-8 for U+0000 is one byte, which modified UTF-8 also reads.
    assert_eq!(i.read_utf().unwrap(), u("aż\u{0}"));
    assert!(i.read_boolean().is_err());
    for bad in [
        &[0u8, 1, 0x80][..],
        &[0, 2, 0xC4, 0x41],
        &[0, 1, 0xF0],
        &[0, 2, 0xE0, 0x80],
        &[0, 5, b'a'],
    ] {
        assert!(DataInput::new(bad).read_utf().is_err(), "{bad:?}");
    }
    assert_eq!(
        DataInput::new(&[0, 3, 0xE2, 0x82, 0xAC])
            .read_utf()
            .unwrap(),
        u("€")
    );
}

#[test]
fn trie_lookups() {
    // Backward trie: "ab" -> row1 via 'b', then 'a' has cmd 1; 'b' alone cmd 0.
    let mut o = Out::default();
    o.trie(
        false,
        0,
        &["Da", "Rx"],
        &[&[('b', 0, 1), ('c', 1, -1)], &[('a', 1, -1)]],
    );
    let t = Trie::read(&mut DataInput::new(&o.0)).unwrap();
    assert!(!t.is_forward());
    assert_eq!(t.root(), 0);
    assert_eq!(t.rows().len(), 2);
    assert_eq!(t.cmds().len(), 2);
    assert_eq!(
        t.get_last_on_path(&u("ab")).unwrap(),
        Some(u("Rx").as_slice())
    );
    assert_eq!(
        t.get_last_on_path(&u("zb")).unwrap(),
        Some(u("Da").as_slice())
    );
    assert_eq!(
        t.get_last_on_path(&u("zc")).unwrap(),
        Some(u("Rx").as_slice())
    );
    assert_eq!(
        t.get_last_on_path(&u("zzc")).unwrap(),
        Some(u("Rx").as_slice())
    );
    assert_eq!(t.get_last_on_path(&u("zzz")).unwrap(), None);
    assert_eq!(
        t.get_last_on_path(&u("b")).unwrap(),
        Some(u("Da").as_slice())
    );
    assert_eq!(t.get_last_on_path(&[]), Err(OutOfBounds));
    assert_eq!(t.rows()[0].at('q' as u16), None);
    // Duplicate characters in a row: the last one wins.
    let mut d = Out::default();
    d.trie(true, 0, &["Da", "Db"], &[&[('a', 0, -1), ('a', 1, -1)]]);
    let t = Trie::read(&mut DataInput::new(&d.0)).unwrap();
    assert_eq!(
        t.get_last_on_path(&u("a")).unwrap(),
        Some(u("Db").as_slice())
    );
}

#[test]
fn corrupt_tries_are_refused() {
    for (root, rows) in [
        (1, &[&[('a', 0, -1)][..]][..]),
        (-1, &[&[('a', 0, -1)][..]][..]),
        (0, &[&[('a', 5, -1)][..]][..]),
        (0, &[&[('a', 0, 3)][..]][..]),
    ] {
        let mut o = Out::default();
        o.trie(true, root, &["Da"], rows);
        assert!(Trie::read(&mut DataInput::new(&o.0)).is_err());
    }
    let mut o = Out::default();
    o.boolean(true).int(0).int(1).utf("x");
    assert!(Trie::read(&mut DataInput::new(&o.0)).is_err());
}

#[test]
fn length_pp_counts_consumed_chars() {
    assert_eq!(length_pp(&u("-cDbRxIy")), Ok(3 + 2 + 1));
    assert_eq!(length_pp(&u("-")), Err(OutOfBounds));
    assert_eq!(length_pp(&u("Ia")), Ok(0));
}

/// A test trie: its commands and its rows of `(char, cmd, ref)`.
type TestTrie<'a> = (&'a [&'a str], &'a [&'a [(char, i32, i32)]]);

fn multi(forward: bool, tries: &[TestTrie]) -> MultiTrie2 {
    let mut o = Out::default();
    o.utf("-0ME2")
        .boolean(forward)
        .int(1)
        .int(tries.len() as i32);
    for (cmds, rows) in tries {
        o.trie(forward, 0, cmds, rows);
    }
    match Table::load(&o.0).unwrap() {
        Table::Multi(m) => m,
        Table::Trie(_) => panic!("a MultiTrie2"),
    }
}

#[test]
fn multi_trie2_chains() {
    // Two tries: the first answers "-b" (skip 2), the second, looked up
    // on the shortened key, "Rx".
    let m = multi(
        false,
        &[
            (&["-b"], &[&[('c', 0, -1)]]),
            (&["Rx", "*"], &[&[('a', 0, -1), ('c', 1, -1)]]),
        ],
    );
    assert_eq!(m.by(), 1);
    assert_eq!(m.tries().len(), 2);
    assert_eq!(m.get_last_on_path(&u("aac")), u("-bRx"));
    // The second trie ends the chain.
    assert_eq!(m.get_last_on_path(&u("ccc")), u("-b"));
    // No command at all.
    assert_eq!(m.get_last_on_path(&u("zzz")), Vec::<u16>::new());
    // Empty key: the first lookup throws in Java; nothing is kept.
    assert_eq!(m.get_last_on_path(&[]), Vec::<u16>::new());
    // The key cannot be shortened by more than it holds.
    assert_eq!(m.get_last_on_path(&u("c")), Vec::<u16>::new());
    // "--" cannot follow "-": the chain stops after the first part.
    let m = multi(
        true,
        &[(&["-a"], &[&[('a', 0, -1)]]), (&["-a"], &[&[('a', 0, -1)]])],
    );
    assert_eq!(m.get_last_on_path(&u("aaaa")), u("-a"));
    // A one-character command has no second-to-last character.
    let m = multi(true, &[(&["R"], &[&[('a', 0, -1)]])]);
    assert_eq!(m.get_last_on_path(&u("a")), Vec::<u16>::new());
    // An empty command.
    let m = multi(true, &[(&[""], &[&[('a', 0, -1)]])]);
    assert_eq!(m.get_last_on_path(&u("a")), Vec::<u16>::new());
    // A previous part whose length cannot be read.
    let m = multi(
        true,
        &[(&["Ia"], &[&[('a', 0, -1)]]), (&["-"], &[&[('a', 0, -1)]])],
    );
    assert_eq!(m.get_last_on_path(&u("aa")), u("Ia"));
    let m = multi(true, &[(&["-a-"], &[&[('a', 0, -1)]])]);
    assert_eq!(m.get_last_on_path(&u("aa")), Vec::<u16>::new());
    let m = multi(
        true,
        &[
            (&["Ix-", "-a"], &[&[('a', 0, -1)]]),
            (&["-a"], &[&[('a', 0, -1)]]),
        ],
    );
    assert_eq!(m.get_last_on_path(&u("aa")), u("Ix-"));
}

#[test]
fn plain_tables_and_errors() {
    let mut o = Out::default();
    o.utf("-0E2").trie(true, 0, &["Rx"], &[&[('a', 0, -1)]]);
    let t = Table::load(&o.0).unwrap();
    assert!(matches!(t, Table::Trie(_)));
    assert_eq!(t.get_last_on_path(&u("a")).unwrap(), Some(u("Rx")));
    assert_eq!(t.get_last_on_path(&[]), Err(OutOfBounds));
    assert!(Table::load(&[0, 1]).is_err());
    let mut lower = Out::default();
    lower.utf("m").boolean(true).int(1).int(0);
    assert!(matches!(Table::load(&lower.0).unwrap(), Table::Multi(_)));
    assert_eq!(
        Table::load(&lower.0)
            .unwrap()
            .get_last_on_path(&u("a"))
            .unwrap(),
        Some(vec![])
    );
}
