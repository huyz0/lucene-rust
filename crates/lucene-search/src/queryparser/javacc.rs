//! The runtime every JavaCC 7 parser Lucene generates shares: the token
//! list, `jj_consume_token`, the syntactic lookahead (`jj_scan_token`,
//! `jj_save`, `jj_rescan_token`), the expected-token bookkeeping
//! (`jj_la1`, `jj_add_error_token`, `generateParseException`) and
//! `ParseException`'s message. A grammar supplies its token images, its
//! `jj_la1_*` masks and its lookahead routine; the productions are its own.
//!
//! Also `FastCharStream`, `TokenMgrError` and `Token`, which the lexers
//! (`QueryParserTokenManager`) and the parsers use alike.

/// A token: its kind, its first UTF-16 column (what `FastCharStream` reports,
/// the line always 1) and its text.
#[derive(Debug, Clone, Default)]
pub(crate) struct Token {
    pub kind: i32,
    pub begin_column: i32,
    pub image: Vec<u16>,
    /// The index of the next token in the parser's token list, once read.
    pub next: Option<usize>,
}

/// `TokenMgrError`: a lexical error, and its message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenMgrError(pub String);

/// A `ParseException` or `TokenMgrError` message: what the productions
/// propagate with `?`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JjError(pub String);

impl From<TokenMgrError> for JjError {
    fn from(e: TokenMgrError) -> Self {
        JjError(e.0)
    }
}

/// How a lookahead scan stops early.
pub(crate) enum Stop {
    /// `LookaheadSuccess`: the lookahead matched.
    Success,
    Lex(TokenMgrError),
}

impl From<TokenMgrError> for Stop {
    fn from(e: TokenMgrError) -> Self {
        Stop::Lex(e)
    }
}

/// `FastCharStream` over the whole query in memory: `bufferStart` is always
/// 0, so a column is an offset into the query's UTF-16 units.
#[derive(Debug, Default)]
pub(crate) struct CharStream {
    buf: Vec<u16>,
    pos: usize,
    token_start: usize,
}

impl CharStream {
    pub(crate) fn new(query: &str) -> Self {
        CharStream {
            buf: query.encode_utf16().collect(),
            pos: 0,
            token_start: 0,
        }
    }

    /// `readChar()`; `None` is the `IOException` at the end.
    pub(crate) fn read_char(&mut self) -> Option<i32> {
        let c = *self.buf.get(self.pos)?;
        self.pos += 1;
        Some(i32::from(c))
    }

    /// `BeginToken()`.
    pub(crate) fn begin_token(&mut self) -> Option<i32> {
        self.token_start = self.pos;
        self.read_char()
    }

    pub(crate) fn backup(&mut self, amount: i32) {
        self.pos -= amount as usize;
    }

    /// `GetImage()`.
    pub(crate) fn image(&self) -> Vec<u16> {
        self.buf[self.token_start..self.pos].to_vec()
    }

    pub(crate) fn begin_column(&self) -> i32 {
        self.token_start as i32
    }

    pub(crate) fn end_column(&self) -> i32 {
        self.pos as i32
    }
}

/// `TokenMgrError.addEscapes` (and `ParseException.add_escapes`).
pub(crate) fn add_escapes(s: &[u16]) -> String {
    let mut out = String::new();
    for &c in s {
        match c {
            0x08 => out.push_str("\\b"),
            0x09 => out.push_str("\\t"),
            0x0a => out.push_str("\\n"),
            0x0c => out.push_str("\\f"),
            0x0d => out.push_str("\\r"),
            0x22 => out.push_str("\\\""),
            0x27 => out.push_str("\\'"),
            0x5c => out.push_str("\\\\"),
            c if !(0x20..=0x7e).contains(&c) => out.push_str(&format!("\\u{c:04x}")),
            c => out.push(char::from(c as u8)),
        }
    }
    out
}

/// `TokenMgrError.LexicalErr`'s message.
pub(crate) fn lexical_error_message(
    eof_seen: bool,
    lex_state: i32,
    error_line: i32,
    error_column: i32,
    error_after: &[u16],
    cur_char: i32,
) -> String {
    // `String.valueOf(curChar)` of an `int`: the character as its decimal code.
    let encountered = if eof_seen {
        "<EOF>".to_string()
    } else {
        format!("'{cur_char}' ({cur_char}),")
    };
    let after = if error_after.is_empty() {
        String::new()
    } else {
        format!(" after prefix \"{}\"", add_escapes(error_after))
    };
    let state = if lex_state == 0 {
        String::new()
    } else {
        format!(" (in lexical state {lex_state})")
    };
    format!(
        "Lexical error at line {error_line}, column {error_column}.  Encountered: {encountered}{after}{state}"
    )
}

/// A generated token manager.
pub(crate) trait Lexer {
    /// `getNextToken()`.
    fn next_token(&mut self) -> Result<Token, TokenMgrError>;
}

/// What a grammar contributes to the runtime.
pub(crate) struct Grammar<L: 'static> {
    /// `tokenImage`.
    pub token_image: &'static [&'static str],
    /// The tokens `generateParseException` loops over (`new boolean[n]`).
    pub num_tokens: usize,
    /// `jj_la1_0`, `jj_la1_1`, ...: one mask per `jj_la1` slot, 32 tokens a
    /// word.
    pub la1_masks: &'static [&'static [u32]],
    /// `jj_3_<n>()`: `Ok(true)` when the lookahead fails.
    pub lookahead: fn(&mut Runtime<L>, usize) -> Result<bool, Stop>,
    /// The number of `jj_2_<n>` lookaheads.
    pub num_lookaheads: usize,
}

/// `JJCalls`: a lookahead to rescan when the parse fails.
#[derive(Debug, Clone, Default)]
struct JjCall {
    gen: i32,
    first: Option<usize>,
    arg: i32,
}

/// One parse's token list and lookahead state.
pub(crate) struct Runtime<L: 'static> {
    grammar: &'static Grammar<L>,
    pub tm: L,
    pub tokens: Vec<Token>,
    pub token: usize,
    jj_ntk: i32,
    pub jj_scanpos: usize,
    jj_lastpos: usize,
    jj_la: i32,
    /// `jj_lookingAhead`: `getToken` counts from `jj_scanpos`.
    pub jj_looking_ahead: bool,
    jj_gen: i32,
    jj_la1: Vec<i32>,
    jj_2_rtns: Vec<Vec<JjCall>>,
    jj_rescan: bool,
    jj_gc: i32,
    jj_expentries: Vec<Vec<i32>>,
    jj_kind: i32,
    jj_lasttokens: [i32; 100],
    jj_endpos: i32,
}

impl<L: Lexer + 'static> Runtime<L> {
    pub(crate) fn new(grammar: &'static Grammar<L>, tm: L) -> Self {
        Runtime {
            grammar,
            tm,
            tokens: vec![Token::default()],
            token: 0,
            jj_ntk: -1,
            jj_scanpos: 0,
            jj_lastpos: 0,
            jj_la: 0,
            jj_looking_ahead: false,
            jj_gen: 0,
            jj_la1: vec![-1; grammar.la1_masks.first().map_or(0, |m| m.len())],
            jj_2_rtns: (0..grammar.num_lookaheads)
                .map(|_| vec![JjCall::default()])
                .collect(),
            jj_rescan: false,
            jj_gc: 0,
            jj_expentries: Vec::new(),
            jj_kind: -1,
            jj_lasttokens: [0; 100],
            jj_endpos: 0,
        }
    }

    /// The token after `t`, read from the lexer the first time.
    pub(crate) fn next_of(&mut self, t: usize) -> Result<usize, TokenMgrError> {
        if let Some(n) = self.tokens[t].next {
            return Ok(n);
        }
        let tok = self.tm.next_token()?;
        self.tokens.push(tok);
        let n = self.tokens.len() - 1;
        self.tokens[t].next = Some(n);
        Ok(n)
    }

    pub(crate) fn kind(&self, t: usize) -> i32 {
        self.tokens[t].kind
    }

    /// `(jj_ntk==-1)?jj_ntk_f():jj_ntk`.
    pub(crate) fn ntk(&mut self) -> Result<i32, JjError> {
        if self.jj_ntk == -1 {
            let n = self.next_of(self.token)?;
            self.jj_ntk = self.kind(n);
        }
        Ok(self.jj_ntk)
    }

    /// `jj_consume_token(kind)`.
    pub(crate) fn consume(&mut self, kind: i32) -> Result<usize, JjError> {
        let old = self.token;
        self.token = self.next_of(self.token)?;
        self.jj_ntk = -1;
        if self.kind(self.token) == kind {
            self.jj_gen += 1;
            self.jj_gc += 1;
            if self.jj_gc > 100 {
                self.jj_gc = 0;
                let gen = self.jj_gen;
                for list in &mut self.jj_2_rtns {
                    for c in list.iter_mut() {
                        if c.gen < gen {
                            c.first = None;
                        }
                    }
                }
            }
            return Ok(self.token);
        }
        self.token = old;
        self.jj_kind = kind;
        Err(self.generate_parse_exception())
    }

    /// `jj_consume_token(-1); throw new ParseException();` -- the second
    /// is never reached: `-1` is no token's kind.
    pub(crate) fn fail(&mut self) -> JjError {
        self.consume(-1)
            .err()
            .unwrap_or_else(|| JjError(String::new()))
    }

    /// A choice none of whose alternatives starts with the next token:
    /// `jj_la1[n] = jj_gen;`, then [`Self::fail`].
    pub(crate) fn no_match(&mut self, n: usize) -> JjError {
        self.jj_la1[n] = self.jj_gen;
        self.fail()
    }

    /// `jj_la1[n] = jj_gen;`: an optional part the next token does not start.
    pub(crate) fn skipped(&mut self, n: usize) {
        self.jj_la1[n] = self.jj_gen;
    }

    /// `jj_scan_token(kind)`: `Ok(true)` when the token is not `kind`.
    pub(crate) fn scan_token(&mut self, kind: i32) -> Result<bool, Stop> {
        if self.jj_scanpos == self.jj_lastpos {
            self.jj_la -= 1;
            let n = self.next_of(self.jj_scanpos)?;
            self.jj_scanpos = n;
            self.jj_lastpos = n;
        } else {
            self.jj_scanpos = self.next_of(self.jj_scanpos)?;
        }
        if self.jj_rescan {
            let mut i = 0;
            let mut tok = Some(self.token);
            while let Some(t) = tok {
                if t == self.jj_scanpos {
                    break;
                }
                i += 1;
                tok = self.tokens[t].next;
            }
            if tok.is_some() {
                self.add_error_token(kind, i);
            }
        }
        if self.kind(self.jj_scanpos) != kind {
            return Ok(true);
        }
        if self.jj_la == 0 && self.jj_scanpos == self.jj_lastpos {
            return Err(Stop::Success);
        }
        Ok(false)
    }

    /// `getToken(index)`.
    pub(crate) fn get_token(&mut self, index: usize) -> Result<usize, TokenMgrError> {
        let mut t = if self.jj_looking_ahead {
            self.jj_scanpos
        } else {
            self.token
        };
        for _ in 0..index {
            t = self.next_of(t)?;
        }
        Ok(t)
    }

    /// `jj_2_<n+1>(xla)`: whether the lookahead matches.
    pub(crate) fn jj_2(&mut self, n: usize, xla: i32) -> Result<bool, JjError> {
        self.jj_la = xla;
        self.jj_scanpos = self.token;
        self.jj_lastpos = self.token;
        let result = match (self.grammar.lookahead)(self, n) {
            Ok(failed) => Ok(!failed),
            Err(Stop::Success) => Ok(true),
            Err(Stop::Lex(e)) => Err(JjError::from(e)),
        };
        self.jj_save(n, xla);
        result
    }

    /// `jj_save(index, xla)`.
    fn jj_save(&mut self, index: usize, xla: i32) {
        let gen = self.jj_gen;
        let list = &mut self.jj_2_rtns[index];
        let mut k = 0;
        while list[k].gen > gen {
            if k + 1 == list.len() {
                list.push(JjCall::default());
                k += 1;
                break;
            }
            k += 1;
        }
        list[k].gen = gen + xla - self.jj_la;
        list[k].first = Some(self.token);
        list[k].arg = xla;
    }

    /// `jj_add_error_token(kind, pos)`.
    fn add_error_token(&mut self, kind: i32, pos: i32) {
        if pos >= 100 {
            return;
        }
        if pos == self.jj_endpos + 1 {
            self.jj_lasttokens[self.jj_endpos as usize] = kind;
            self.jj_endpos += 1;
        } else if self.jj_endpos != 0 {
            let entry: Vec<i32> = self.jj_lasttokens[..self.jj_endpos as usize].to_vec();
            // JavaCC 7 adds the sequence only when an equal one is already
            // listed.
            if self.jj_expentries.contains(&entry) {
                self.jj_expentries.push(entry);
            }
            if pos != 0 {
                self.jj_endpos = pos;
                self.jj_lasttokens[(pos - 1) as usize] = kind;
            }
        }
    }

    /// `jj_rescan_token()`.
    fn rescan_token(&mut self) -> Result<(), TokenMgrError> {
        self.jj_rescan = true;
        for i in 0..self.jj_2_rtns.len() {
            let mut k = 0;
            while k < self.jj_2_rtns[i].len() {
                let p = self.jj_2_rtns[i][k].clone();
                if p.gen > self.jj_gen {
                    self.jj_la = p.arg;
                    let first = p.first.unwrap_or(0);
                    self.jj_scanpos = first;
                    self.jj_lastpos = first;
                    match (self.grammar.lookahead)(self, i) {
                        Ok(_) => {}
                        Err(Stop::Success) => break,
                        Err(Stop::Lex(e)) => {
                            self.jj_rescan = false;
                            return Err(e);
                        }
                    }
                }
                k += 1;
            }
        }
        self.jj_rescan = false;
        Ok(())
    }

    /// `generateParseException()`.
    fn generate_parse_exception(&mut self) -> JjError {
        self.jj_expentries.clear();
        let words = self.grammar.la1_masks.len();
        let mut la1tokens = vec![false; 32 * words];
        if self.jj_kind >= 0 {
            la1tokens[self.jj_kind as usize] = true;
            self.jj_kind = -1;
        }
        for i in 0..self.jj_la1.len() {
            if self.jj_la1[i] == self.jj_gen {
                for (w, masks) in self.grammar.la1_masks.iter().enumerate() {
                    for j in 0..32 {
                        if masks[i] & (1 << j) != 0 {
                            la1tokens[32 * w + j] = true;
                        }
                    }
                }
            }
        }
        for (i, &set) in la1tokens.iter().enumerate().take(self.grammar.num_tokens) {
            if set {
                self.jj_expentries.push(vec![i as i32]);
            }
        }
        self.jj_endpos = 0;
        if let Err(e) = self.rescan_token() {
            return JjError::from(e);
        }
        self.add_error_token(0, 0);
        JjError(self.parse_exception_message())
    }

    /// `ParseException.initialise(token, jj_expentries, tokenImage)`.
    fn parse_exception_message(&self) -> String {
        const EOL: &str = "\n";
        let image = self.grammar.token_image;
        let mut expected = String::new();
        let mut max_size = 0;
        for seq in &self.jj_expentries {
            max_size = max_size.max(seq.len());
            for &k in seq {
                expected.push_str(image[k as usize]);
                expected.push(' ');
            }
            if seq.last().copied().unwrap_or(0) != 0 {
                expected.push_str("...");
            }
            expected.push_str(EOL);
            expected.push_str("    ");
        }
        let mut retval = String::from("Encountered \"");
        let mut tok = self.tokens[self.token].next;
        for i in 0..max_size {
            if i != 0 {
                retval.push(' ');
            }
            let Some(t) = tok else { break };
            let t = &self.tokens[t];
            if t.kind == 0 {
                retval.push_str(image[0]);
                break;
            }
            retval.push(' ');
            retval.push_str(image[t.kind as usize]);
            retval.push_str(" \"");
            retval.push_str(&add_escapes(&t.image));
            retval.push_str(" \"");
            tok = t.next;
        }
        if let Some(next) = self.tokens[self.token].next {
            retval.push_str(&format!(
                "\" at line 1, column {}",
                self.tokens[next].begin_column
            ));
        }
        retval.push('.');
        retval.push_str(EOL);
        match self.jj_expentries.len() {
            0 => {}
            1 => {
                retval.push_str("Was expecting:");
                retval.push_str(EOL);
                retval.push_str("    ");
            }
            _ => {
                retval.push_str("Was expecting one of:");
                retval.push_str(EOL);
                retval.push_str("    ");
            }
        }
        retval.push_str(&expected);
        retval
    }
}

// ------------------------------------------------------------------ lexer

/// `NO_KIND`: `0x7fffffff`, no token matched yet.
pub(crate) const NO_KIND: i32 = 0x7fff_ffff;

/// What every generated `QueryParserTokenManager` carries: the character
/// stream, the NFA's state set and round counters, and the match so far.
pub(crate) struct NfaCore {
    pub input: CharStream,
    pub cur_char: i32,
    pub cur_lex_state: i32,
    jjnew_state_cnt: usize,
    jjround: i32,
    pub jjmatched_pos: i32,
    pub jjmatched_kind: i32,
    jjrounds: Vec<i32>,
    jjstate_set: Vec<i32>,
    /// `jjnextStates`.
    next_states: &'static [i32],
    num_states: usize,
}

impl NfaCore {
    /// A token manager over `input` whose NFAs have at most `num_states`
    /// states, in `default_lex_state`.
    pub(crate) fn new(
        input: CharStream,
        num_states: usize,
        next_states: &'static [i32],
        default_lex_state: i32,
    ) -> Self {
        let mut core = NfaCore {
            input,
            cur_char: 0,
            cur_lex_state: default_lex_state,
            jjnew_state_cnt: 0,
            jjround: 0,
            jjmatched_pos: 0,
            jjmatched_kind: 0,
            jjrounds: vec![0; num_states],
            jjstate_set: vec![0; 2 * num_states],
            next_states,
            num_states,
        };
        core.re_init_rounds();
        core
    }

    fn re_init_rounds(&mut self) {
        self.jjround = 0x8000_0001_u32 as i32;
        self.jjrounds = vec![i32::MIN; self.num_states];
    }

    /// `readChar()` into `curChar`; `false` at the end.
    pub(crate) fn read(&mut self) -> bool {
        match self.input.read_char() {
            Some(c) => {
                self.cur_char = c;
                true
            }
            None => false,
        }
    }

    /// `jjstateSet[jjnewStateCnt++] = state`.
    pub(crate) fn push(&mut self, state: i32) {
        self.jjstate_set[self.jjnew_state_cnt] = state;
        self.jjnew_state_cnt += 1;
    }

    /// `jjCheckNAdd`.
    pub(crate) fn check_n_add(&mut self, state: i32) {
        if self.jjrounds[state as usize] != self.jjround {
            self.push(state);
            self.jjrounds[state as usize] = self.jjround;
        }
    }

    /// `jjAddStates`.
    pub(crate) fn add_states(&mut self, start: usize, end: usize) {
        for k in start..=end {
            let state = self.next_states[k];
            self.push(state);
        }
    }

    /// `jjCheckNAddTwoStates`.
    pub(crate) fn two(&mut self, s1: i32, s2: i32) {
        self.check_n_add(s1);
        self.check_n_add(s2);
    }

    /// `jjCheckNAddStates`.
    pub(crate) fn states(&mut self, start: usize, end: usize) {
        for k in start..=end {
            let state = self.next_states[k];
            self.check_n_add(state);
        }
    }

    /// `jjStopAtPos`.
    pub(crate) fn stop_at_pos(&mut self, pos: i32, kind: i32) -> i32 {
        self.jjmatched_kind = kind;
        self.jjmatched_pos = pos;
        pos + 1
    }

    /// The start of a `jjMoveNfa_*`: a state set of `start_state` alone, in
    /// an NFA of `size` states.
    pub(crate) fn nfa_start(&mut self, start_state: i32, size: usize) {
        self.jjnew_state_cnt = size;
        self.jjstate_set[0] = start_state;
    }

    /// `jjstateSet[i]`.
    pub(crate) fn state_at(&self, i: usize) -> i32 {
        self.jjstate_set[i]
    }

    /// The round bookkeeping that ends each step of every `jjMoveNfa_*`:
    /// records `kind`, advances, and swaps the two halves of the state set.
    /// `Err(cur_pos)` returns from the NFA.
    pub(crate) fn nfa_step(
        &mut self,
        kind: &mut i32,
        cur_pos: &mut i32,
        i: &mut usize,
        starts_at: &mut usize,
        size: usize,
    ) -> Result<(), i32> {
        if *kind != NO_KIND {
            self.jjmatched_kind = *kind;
            self.jjmatched_pos = *cur_pos;
            *kind = NO_KIND;
        }
        *cur_pos += 1;
        *i = self.jjnew_state_cnt;
        let old = *starts_at;
        self.jjnew_state_cnt = old;
        *starts_at = size - old;
        if *i == *starts_at {
            return Err(*cur_pos);
        }
        if !self.read() {
            return Err(*cur_pos);
        }
        Ok(())
    }

    /// `++jjround == 0x7fffffff` -> `ReInitRounds()`.
    pub(crate) fn next_round(&mut self) {
        self.jjround = self.jjround.wrapping_add(1);
        if self.jjround == 0x7fff_ffff {
            self.re_init_rounds();
        }
    }

    /// `jjFillToken()`.
    fn fill_token(&self, literal: Option<&'static str>) -> Token {
        let image = match literal {
            Some(s) => s.encode_utf16().collect(),
            None => self.input.image(),
        };
        Token {
            kind: self.jjmatched_kind,
            begin_column: self.input.begin_column(),
            image,
            next: None,
        }
    }

    /// The `TokenMgrError` `getNextToken` throws when no token matches.
    fn lexical_error(&mut self, cur_pos: i32) -> TokenMgrError {
        let mut error_line = 1;
        let mut error_column = self.input.end_column();
        let eof_seen;
        let error_after;
        if self.input.read_char().is_some() {
            self.input.backup(1);
            eof_seen = false;
            self.input.backup(1);
            error_after = if cur_pos <= 1 {
                Vec::new()
            } else {
                self.input.image()
            };
        } else {
            eof_seen = true;
            error_after = if cur_pos <= 1 {
                Vec::new()
            } else {
                self.input.image()
            };
            if self.cur_char == 0x0a || self.cur_char == 0x0d {
                error_line += 1;
                error_column = 0;
            } else {
                error_column += 1;
            }
        }
        TokenMgrError(lexical_error_message(
            eof_seen,
            self.cur_lex_state,
            error_line,
            error_column,
            &error_after,
            self.cur_char,
        ))
    }
}

/// A generated token manager: its lexical states' DFA/NFA entry points, on
/// top of [`NfaCore`].
pub(crate) trait Dfa: std::ops::DerefMut<Target = NfaCore> + Sized {
    /// The `jjMoveStringLiteralDfa0_<state>` of the current lexical state,
    /// from `cur_char`: the position after the longest match.
    fn dispatch(&mut self) -> i32;
    /// `jjnewLexState[kind]`: `-1` leaves the state alone.
    // SENTINEL: `-1` = the kind does not change the lexical state.
    fn new_lex_state(kind: i32) -> i32;
    /// `jjtoToken`: whether `kind` is a token (not a skipped one).
    fn is_token(kind: i32) -> bool;
    /// `jjstrLiteralImages[kind]`.
    fn literal_image(kind: i32) -> Option<&'static str>;

    /// `getNextToken()`.
    fn next_token_loop(&mut self) -> Result<Token, TokenMgrError> {
        loop {
            match self.input.begin_token() {
                Some(c) => self.cur_char = c,
                None => {
                    self.jjmatched_kind = 0;
                    self.jjmatched_pos = -1;
                    return Ok(self.fill_token(Self::literal_image(0)));
                }
            }
            self.jjmatched_kind = NO_KIND;
            self.jjmatched_pos = 0;
            let cur_pos = self.dispatch();
            if self.jjmatched_kind != NO_KIND {
                if self.jjmatched_pos + 1 < cur_pos {
                    let amount = cur_pos - self.jjmatched_pos - 1;
                    self.input.backup(amount);
                }
                let kind = self.jjmatched_kind;
                let new_state = Self::new_lex_state(kind);
                if Self::is_token(kind) {
                    let token = self.fill_token(Self::literal_image(kind));
                    if new_state != -1 {
                        self.cur_lex_state = new_state;
                    }
                    return Ok(token);
                }
                if new_state != -1 {
                    self.cur_lex_state = new_state;
                }
                continue;
            }
            return Err(self.lexical_error(cur_pos));
        }
    }
}
