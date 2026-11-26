//! The parser: `PainlessParser.g4` as recursive descent, with precedence
//! climbing for the binary operators and bounded backtracking where the
//! grammar needs lookahead (a declaration against an expression, a cast
//! against a parenthesised expression, a lambda or method reference against
//! an expression).

use super::ast::*;
use super::lex::{lex, Tok, Token};
use super::ParseError;

/// What a script may be before it is refused, whatever it means.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Source length in bytes.
    pub max_source: usize,
    /// Nesting of expressions, statements and blocks.
    pub max_depth: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_source: 65_535,
            max_depth: 100,
        }
    }
}

type R<T> = Result<T, ParseError>;

/// Parses `source` into a syntax tree.
pub fn parse(source: &str, limits: &Limits) -> R<Script> {
    if source.len() > limits.max_source {
        return Err(ParseError {
            offset: 0,
            message: format!(
                "the script is {} bytes, over the limit of {}",
                source.len(),
                limits.max_source
            ),
        });
    }
    let toks = lex(source)?;
    let mut p = P {
        toks,
        i: 0,
        depth: 0,
        max_depth: limits.max_depth,
    };
    p.script()
}

struct P {
    toks: Vec<Token>,
    i: usize,
    depth: usize,
    max_depth: usize,
}

fn binary_op(t: &Tok) -> Option<(u8, BinOp)> {
    Some(match t {
        Tok::Mul => (12, BinOp::Mul),
        Tok::Div => (12, BinOp::Div),
        Tok::Rem => (12, BinOp::Rem),
        Tok::Add => (11, BinOp::Add),
        Tok::Sub => (11, BinOp::Sub),
        Tok::Find => (10, BinOp::Find),
        Tok::Match => (10, BinOp::Match),
        Tok::Lsh => (9, BinOp::Lsh),
        Tok::Rsh => (9, BinOp::Rsh),
        Tok::Ush => (9, BinOp::Ush),
        Tok::Lt => (8, BinOp::Lt),
        Tok::Lte => (8, BinOp::Lte),
        Tok::Gt => (8, BinOp::Gt),
        Tok::Gte => (8, BinOp::Gte),
        Tok::Eq => (6, BinOp::Eq),
        Tok::EqRef => (6, BinOp::EqRef),
        Tok::Ne => (6, BinOp::Ne),
        Tok::NeRef => (6, BinOp::NeRef),
        Tok::BwAnd => (5, BinOp::BitAnd),
        Tok::Xor => (4, BinOp::Xor),
        Tok::BwOr => (3, BinOp::BitOr),
        Tok::BoolAnd => (2, BinOp::And),
        Tok::BoolOr => (1, BinOp::Or),
        Tok::Elvis => (0, BinOp::Elvis),
        _ => return None,
    })
}

const INSTANCEOF_PREC: u8 = 7;

fn assign_op(t: &Tok) -> Option<AssignOp> {
    Some(match t {
        Tok::Assign => AssignOp::Set,
        Tok::AAdd => AssignOp::With(BinOp::Add),
        Tok::ASub => AssignOp::With(BinOp::Sub),
        Tok::AMul => AssignOp::With(BinOp::Mul),
        Tok::ADiv => AssignOp::With(BinOp::Div),
        Tok::ARem => AssignOp::With(BinOp::Rem),
        Tok::AAnd => AssignOp::With(BinOp::BitAnd),
        Tok::AXor => AssignOp::With(BinOp::Xor),
        Tok::AOr => AssignOp::With(BinOp::BitOr),
        Tok::ALsh => AssignOp::With(BinOp::Lsh),
        Tok::ARsh => AssignOp::With(BinOp::Rsh),
        Tok::AUsh => AssignOp::With(BinOp::Ush),
        _ => return None,
    })
}

impl P {
    fn peek(&self) -> &Tok {
        &self.toks[self.i].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        &self.toks[(self.i + n).min(self.toks.len() - 1)].tok
    }

    fn pos(&self) -> usize {
        self.toks[self.i].pos
    }

    fn at(&self, t: &Tok) -> bool {
        self.peek() == t
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.i].tok.clone();
        if self.i < self.toks.len() - 1 {
            self.i += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.at(t) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn error<T>(&self, expected: &str) -> R<T> {
        Err(ParseError {
            offset: self.pos(),
            message: format!("expected {expected}, found {}", describe(self.peek())),
        })
    }

    fn expect(&mut self, t: &Tok, what: &str) -> R<()> {
        if self.eat(t) {
            Ok(())
        } else {
            self.error(what)
        }
    }

    fn ident(&mut self) -> R<String> {
        match self.peek().clone() {
            Tok::Id(s) => {
                self.bump();
                Ok(s)
            }
            _ => self.error("an identifier"),
        }
    }

    fn enter(&mut self) -> R<()> {
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(ParseError {
                offset: self.pos(),
                message: format!("the script nests deeper than {}", self.max_depth),
            });
        }
        Ok(())
    }

    /// Runs `f`; on failure puts the position back and answers `None`.
    fn attempt<T>(&mut self, f: impl FnOnce(&mut Self) -> R<T>) -> Option<T> {
        let (i, d) = (self.i, self.depth);
        match f(self) {
            Ok(v) => Some(v),
            Err(_) => {
                self.i = i;
                self.depth = d;
                None
            }
        }
    }

    // ---- script, functions, types ----

    fn script(&mut self) -> R<Script> {
        let mut functions = Vec::new();
        while let Some(f) = self.function()? {
            functions.push(f);
        }
        let mut body = Vec::new();
        while !self.at(&Tok::Eof) {
            body.push(self.statement(false)?);
        }
        Ok(Script { functions, body })
    }

    /// A function if the next tokens are `type name (`; the header decides,
    /// so an error inside the function is its own.
    fn function(&mut self) -> R<Option<Function>> {
        if !matches!(self.peek(), Tok::Def | Tok::Primitive(_) | Tok::Id(_)) {
            return Ok(None);
        }
        let start = self.i;
        let pos = self.pos();
        let header = self.attempt(|p| {
            let ret = p.decltype()?;
            let name = p.ident()?;
            if !p.at(&Tok::Lp) {
                return p.error("(");
            }
            Ok((ret, name))
        });
        let Some((ret, name)) = header else {
            self.i = start;
            return Ok(None);
        };
        self.bump();
        let mut params = Vec::new();
        if !self.at(&Tok::Rp) {
            loop {
                let ty = self.decltype()?;
                params.push((ty, self.ident()?));
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        self.expect(&Tok::Rp, ")")?;
        let body = self.block()?;
        Ok(Some(Function {
            pos: pos as Pos,
            ret,
            name,
            params,
            body,
        }))
    }

    /// `def`, a primitive, or a dotted name.
    fn type_name(&mut self) -> R<Type> {
        let pos = self.pos() as Pos;
        let name = match self.peek().clone() {
            Tok::Def => "def".to_string(),
            Tok::Primitive(s) => s,
            Tok::Id(mut s) => {
                while self.peek_at(1) == &Tok::Dot && matches!(self.peek_at(2), Tok::DotId(_)) {
                    self.bump();
                    self.bump();
                    if let Tok::DotId(n) = &self.toks[self.i].tok {
                        s.push('.');
                        s.push_str(n);
                    }
                }
                s
            }
            _ => return self.error("a type"),
        };
        self.bump();
        Ok(Type { pos, name, dims: 0 })
    }

    fn dims(&mut self) -> usize {
        let mut n = 0;
        while self.at(&Tok::LBrace) && self.peek_at(1) == &Tok::RBrace {
            self.bump();
            self.bump();
            n += 1;
        }
        n
    }

    fn decltype(&mut self) -> R<Type> {
        let mut t = self.type_name()?;
        t.dims = self.dims();
        Ok(t)
    }

    /// Whether a declaration starts here: a type, then a name.
    fn at_declaration(&mut self) -> bool {
        if !matches!(self.peek(), Tok::Def | Tok::Primitive(_) | Tok::Id(_)) {
            return false;
        }
        let i = self.i;
        let ok = self.decltype().is_ok() && matches!(self.peek(), Tok::Id(_));
        self.i = i;
        ok
    }

    // ---- statements ----

    fn block(&mut self) -> R<Block> {
        self.enter()?;
        let pos = self.pos() as Pos;
        self.expect(&Tok::LBrack, "{")?;
        let mut stmts = Vec::new();
        while !self.at(&Tok::RBrack) {
            if self.at(&Tok::Eof) {
                return self.error("}");
            }
            stmts.push(self.statement(true)?);
        }
        self.bump();
        self.depth -= 1;
        Ok(Block { pos, stmts })
    }

    /// A statement. A plain statement ends at `;` or the end of the script;
    /// in a block's own list it may also end at the block's `}`.
    fn statement(&mut self, in_block: bool) -> R<Stmt> {
        self.enter()?;
        let pos = self.pos() as Pos;
        let kind = match self.peek() {
            Tok::If => self.if_stmt()?,
            Tok::While => self.while_stmt()?,
            Tok::For => self.for_stmt()?,
            Tok::Try => self.try_stmt()?,
            _ => {
                let kind = self.dstatement()?;
                if !(self.eat(&Tok::Semicolon)
                    || self.at(&Tok::Eof)
                    || (in_block && self.at(&Tok::RBrack)))
                {
                    return self.error(";");
                }
                kind
            }
        };
        self.depth -= 1;
        Ok(Stmt { pos, kind })
    }

    /// A block, or one statement (the body of an `if`, `for`, ...).
    fn trailer(&mut self) -> R<Box<Stmt>> {
        if self.at(&Tok::LBrack) {
            let b = self.block()?;
            return Ok(Box::new(Stmt {
                pos: b.pos,
                kind: StmtKind::Block(b),
            }));
        }
        Ok(Box::new(self.statement(false)?))
    }

    /// A loop body: a trailer, or an empty `;`.
    fn loop_body(&mut self) -> R<Option<Box<Stmt>>> {
        if self.eat(&Tok::Semicolon) {
            return Ok(None);
        }
        Ok(Some(self.trailer()?))
    }

    fn paren_expr(&mut self) -> R<Expr> {
        self.expect(&Tok::Lp, "(")?;
        let e = self.expression()?;
        self.expect(&Tok::Rp, ")")?;
        Ok(e)
    }

    fn if_stmt(&mut self) -> R<StmtKind> {
        self.bump();
        let cond = self.paren_expr()?;
        let then = self.trailer()?;
        let otherwise = if self.eat(&Tok::Else) {
            Some(self.trailer()?)
        } else {
            None
        };
        Ok(StmtKind::If {
            cond,
            then,
            otherwise,
        })
    }

    fn while_stmt(&mut self) -> R<StmtKind> {
        self.bump();
        let cond = self.paren_expr()?;
        let body = self.loop_body()?;
        Ok(StmtKind::While { cond, body })
    }

    fn for_stmt(&mut self) -> R<StmtKind> {
        self.bump();
        self.expect(&Tok::Lp, "(")?;
        let each = self.attempt(|p| {
            if p.peek_at(1) == &Tok::In && matches!(p.peek(), Tok::Id(_)) {
                let name = p.ident()?;
                p.bump();
                return Ok((None, name));
            }
            let ty = p.decltype()?;
            let name = p.ident()?;
            p.expect(&Tok::Colon, ":")?;
            Ok((Some(ty), name))
        });
        if let Some((ty, name)) = each {
            let iterable = self.expression()?;
            self.expect(&Tok::Rp, ")")?;
            let body = self.trailer()?;
            return Ok(StmtKind::Each {
                ty,
                name,
                iterable,
                body,
            });
        }
        let init = if self.at(&Tok::Semicolon) {
            None
        } else if self.at_declaration() {
            Some(ForInit::Decl(self.declaration()?))
        } else {
            Some(ForInit::Expr(self.expression()?))
        };
        self.expect(&Tok::Semicolon, ";")?;
        let cond = if self.at(&Tok::Semicolon) {
            None
        } else {
            Some(self.expression()?)
        };
        self.expect(&Tok::Semicolon, ";")?;
        let after = if self.at(&Tok::Rp) {
            None
        } else {
            Some(self.expression()?)
        };
        self.expect(&Tok::Rp, ")")?;
        let body = self.loop_body()?;
        Ok(StmtKind::For {
            init,
            cond,
            after,
            body,
        })
    }

    fn try_stmt(&mut self) -> R<StmtKind> {
        self.bump();
        let body = self.block()?;
        let mut catches = Vec::new();
        while self.at(&Tok::Catch) {
            let pos = self.pos() as Pos;
            self.bump();
            self.expect(&Tok::Lp, "(")?;
            let ty = self.type_name()?;
            let name = self.ident()?;
            self.expect(&Tok::Rp, ")")?;
            let body = self.block()?;
            catches.push(Catch {
                pos,
                ty,
                name,
                body,
            });
        }
        if catches.is_empty() {
            return self.error("catch");
        }
        Ok(StmtKind::Try { body, catches })
    }

    fn dstatement(&mut self) -> R<StmtKind> {
        match self.peek().clone() {
            Tok::Do => {
                self.bump();
                let body = self.block()?;
                self.expect(&Tok::While, "while")?;
                let cond = self.paren_expr()?;
                Ok(StmtKind::Do { body, cond })
            }
            Tok::Continue => {
                self.bump();
                Ok(StmtKind::Continue)
            }
            Tok::Break => {
                self.bump();
                Ok(StmtKind::Break)
            }
            Tok::Return => {
                self.bump();
                if matches!(self.peek(), Tok::Semicolon | Tok::RBrack | Tok::Eof) {
                    Ok(StmtKind::Return(None))
                } else {
                    Ok(StmtKind::Return(Some(self.expression()?)))
                }
            }
            Tok::Throw => {
                self.bump();
                Ok(StmtKind::Throw(self.expression()?))
            }
            _ if self.at_declaration() => Ok(StmtKind::Decl(self.declaration()?)),
            _ => Ok(StmtKind::Expr(self.expression()?)),
        }
    }

    fn declaration(&mut self) -> R<Vec<Declaration>> {
        let ty = self.decltype()?;
        let mut out = Vec::new();
        loop {
            let pos = self.pos() as Pos;
            let name = self.ident()?;
            let init = if self.eat(&Tok::Assign) {
                Some(self.expression()?)
            } else {
                None
            };
            out.push(Declaration {
                pos,
                ty: ty.clone(),
                name,
                init,
            });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(out)
    }

    // ---- expressions ----

    fn expression(&mut self) -> R<Expr> {
        self.enter()?;
        let pos = self.pos() as Pos;
        let lhs = self.noncond(0)?;
        let kind = if self.eat(&Tok::Cond) {
            let a = self.expression()?;
            self.expect(&Tok::Colon, ":")?;
            let b = self.expression()?;
            ExprKind::Cond(Box::new(lhs), Box::new(a), Box::new(b))
        } else if let Some(op) = assign_op(self.peek()) {
            self.bump();
            let rhs = self.expression()?;
            ExprKind::Assign(op, Box::new(lhs), Box::new(rhs))
        } else {
            self.depth -= 1;
            return Ok(lhs);
        };
        self.depth -= 1;
        Ok(Expr { pos, kind })
    }

    fn noncond(&mut self, min: u8) -> R<Expr> {
        let mut lhs = self.unary()?;
        loop {
            let pos = lhs.pos;
            if self.at(&Tok::InstanceOf) {
                if INSTANCEOF_PREC < min {
                    break;
                }
                self.bump();
                let ty = self.decltype()?;
                lhs = Expr {
                    pos,
                    kind: ExprKind::InstanceOf(Box::new(lhs), ty),
                };
                continue;
            }
            let Some((prec, op)) = binary_op(self.peek()) else {
                break;
            };
            if prec < min {
                break;
            }
            self.bump();
            // Elvis is right associative; every other operator is left.
            let rhs = self.noncond(if op == BinOp::Elvis { prec } else { prec + 1 })?;
            lhs = Expr {
                pos,
                kind: ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)),
            };
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> R<Expr> {
        self.enter()?;
        let pos = self.pos() as Pos;
        let kind = match self.peek() {
            Tok::Incr | Tok::Decr => {
                let inc = self.bump() == Tok::Incr;
                let target = self.chain()?;
                ExprKind::IncDec {
                    prefix: true,
                    inc,
                    target: Box::new(target),
                }
            }
            Tok::Add | Tok::Sub => {
                let op = if self.bump() == Tok::Add {
                    UnOp::Plus
                } else {
                    UnOp::Minus
                };
                ExprKind::Unary(op, Box::new(self.unary()?))
            }
            _ => {
                self.depth -= 1;
                return self.unary_not_add_sub();
            }
        };
        self.depth -= 1;
        Ok(Expr { pos, kind })
    }

    fn unary_not_add_sub(&mut self) -> R<Expr> {
        let pos = self.pos() as Pos;
        match self.peek() {
            Tok::BoolNot | Tok::BwNot => {
                let op = if self.bump() == Tok::BoolNot {
                    UnOp::Not
                } else {
                    UnOp::BitNot
                };
                let e = self.unary()?;
                Ok(Expr {
                    pos,
                    kind: ExprKind::Unary(op, Box::new(e)),
                })
            }
            Tok::Lp => {
                if let Some(c) = self.attempt(|p| p.cast()) {
                    return Ok(c);
                }
                self.postfix_incdec()
            }
            _ => self.postfix_incdec(),
        }
    }

    fn postfix_incdec(&mut self) -> R<Expr> {
        let pos = self.pos() as Pos;
        let e = self.chain()?;
        if matches!(self.peek(), Tok::Incr | Tok::Decr) {
            let inc = self.bump() == Tok::Incr;
            return Ok(Expr {
                pos,
                kind: ExprKind::IncDec {
                    prefix: false,
                    inc,
                    target: Box::new(e),
                },
            });
        }
        Ok(e)
    }

    fn cast(&mut self) -> R<Expr> {
        let pos = self.pos() as Pos;
        self.expect(&Tok::Lp, "(")?;
        let mut ty = self.type_name()?;
        ty.dims = self.dims();
        self.expect(&Tok::Rp, ")")?;
        let value_is_unary = ty.dims == 0 && matches!(ty.name.as_str(), "def")
            || ty.dims == 0 && is_primitive(&ty.name);
        let e = if value_is_unary {
            self.unary()?
        } else {
            self.unary_not_add_sub()?
        };
        Ok(Expr {
            pos,
            kind: ExprKind::Cast(ty, Box::new(e)),
        })
    }

    /// A primary with its postfixes, or an array creation.
    fn chain(&mut self) -> R<Expr> {
        let pos = self.pos() as Pos;
        let base = match self.peek().clone() {
            Tok::New => return self.new_expr(),
            Tok::Lp => {
                self.bump();
                let e = self.expression()?;
                self.expect(&Tok::Rp, ")")?;
                ExprKind::Paren(Box::new(e))
            }
            Tok::Octal(s) => {
                self.bump();
                ExprKind::Num(NumKind::Octal, s)
            }
            Tok::Hex(s) => {
                self.bump();
                ExprKind::Num(NumKind::Hex, s)
            }
            Tok::Integer(s) => {
                self.bump();
                ExprKind::Num(NumKind::Integer, s)
            }
            Tok::Decimal(s) => {
                self.bump();
                ExprKind::Num(NumKind::Decimal, s)
            }
            Tok::True => {
                self.bump();
                ExprKind::Bool(true)
            }
            Tok::False => {
                self.bump();
                ExprKind::Bool(false)
            }
            Tok::Null => {
                self.bump();
                ExprKind::Null
            }
            Tok::Str(s) => {
                self.bump();
                ExprKind::Str(s)
            }
            Tok::Regex(p, f) => {
                self.bump();
                ExprKind::Regex(p, f)
            }
            Tok::LBrace => self.list_or_map()?,
            Tok::Id(name) => {
                self.bump();
                if self.at(&Tok::Lp) {
                    ExprKind::CallLocal(name, self.arguments()?)
                } else {
                    ExprKind::Var(name)
                }
            }
            _ => return self.error("an expression"),
        };
        self.postfixes(Expr { pos, kind: base })
    }

    fn postfixes(&mut self, mut e: Expr) -> R<Expr> {
        loop {
            let pos = e.pos;
            let null_safe = match self.peek() {
                Tok::Dot => false,
                Tok::NsDot => true,
                Tok::LBrace => {
                    self.bump();
                    let index = self.expression()?;
                    self.expect(&Tok::RBrace, "]")?;
                    e = Expr {
                        pos,
                        kind: ExprKind::Index(Box::new(e), Box::new(index)),
                    };
                    continue;
                }
                _ => return Ok(e),
            };
            self.bump();
            let kind = match self.peek().clone() {
                Tok::DotId(name) => {
                    self.bump();
                    if self.at(&Tok::Lp) {
                        ExprKind::Call {
                            recv: Box::new(e),
                            name,
                            args: self.arguments()?,
                            null_safe,
                        }
                    } else {
                        ExprKind::Field {
                            recv: Box::new(e),
                            name,
                            null_safe,
                        }
                    }
                }
                Tok::DotInteger(name) => {
                    self.bump();
                    ExprKind::Field {
                        recv: Box::new(e),
                        name,
                        null_safe,
                    }
                }
                _ => return self.error("a name after '.'"),
            };
            e = Expr { pos, kind };
        }
    }

    fn new_expr(&mut self) -> R<Expr> {
        let pos = self.pos() as Pos;
        self.bump();
        let ty = self.type_name()?;
        if self.at(&Tok::Lp) {
            let args = self.arguments()?;
            return self.postfixes(Expr {
                pos,
                kind: ExprKind::NewObject(ty, args),
            });
        }
        if !self.at(&Tok::LBrace) {
            return self.error("( or [");
        }
        if self.peek_at(1) == &Tok::RBrace {
            // new T[] { a, b }: `ty` is the array's own type.
            let mut ty = ty;
            ty.dims = self.dims();
            self.expect(&Tok::LBrack, "{")?;
            let mut elems = Vec::new();
            if !self.at(&Tok::RBrack) {
                loop {
                    elems.push(self.expression()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
            }
            self.expect(&Tok::RBrack, "}")?;
            return self.postfixes(Expr {
                pos,
                kind: ExprKind::NewArrayInit(ty, elems),
            });
        }
        let mut dims = Vec::new();
        while self.at(&Tok::LBrace) {
            self.bump();
            dims.push(self.expression()?);
            self.expect(&Tok::RBrace, "]")?;
        }
        let e = Expr {
            pos,
            kind: ExprKind::NewArray(ty, dims),
        };
        if matches!(self.peek(), Tok::Dot | Tok::NsDot) {
            return self.postfixes(e);
        }
        Ok(e)
    }

    fn list_or_map(&mut self) -> R<ExprKind> {
        self.expect(&Tok::LBrace, "[")?;
        if self.eat(&Tok::RBrace) {
            return Ok(ExprKind::List(Vec::new()));
        }
        if self.at(&Tok::Colon) && self.peek_at(1) == &Tok::RBrace {
            self.bump();
            self.bump();
            return Ok(ExprKind::Map(Vec::new()));
        }
        let first = self.expression()?;
        if self.eat(&Tok::Colon) {
            let mut pairs = vec![(first, self.expression()?)];
            while self.eat(&Tok::Comma) {
                let k = self.expression()?;
                self.expect(&Tok::Colon, ":")?;
                pairs.push((k, self.expression()?));
            }
            self.expect(&Tok::RBrace, "]")?;
            return Ok(ExprKind::Map(pairs));
        }
        let mut items = vec![first];
        while self.eat(&Tok::Comma) {
            items.push(self.expression()?);
        }
        self.expect(&Tok::RBrace, "]")?;
        Ok(ExprKind::List(items))
    }

    // ---- arguments, lambdas, method references ----

    fn arguments(&mut self) -> R<Vec<Arg>> {
        self.expect(&Tok::Lp, "(")?;
        let mut args = Vec::new();
        if !self.at(&Tok::Rp) {
            loop {
                args.push(self.argument()?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        self.expect(&Tok::Rp, ")")?;
        Ok(args)
    }

    fn argument(&mut self) -> R<Arg> {
        let pos = self.pos() as Pos;
        if let Some(params) = self.attempt(|p| p.lambda_header()) {
            let body = if self.at(&Tok::LBrack) {
                LambdaBody::Block(self.block()?)
            } else {
                LambdaBody::Expr(Box::new(self.expression()?))
            };
            return Ok(Arg::Lambda(Lambda { pos, params, body }));
        }
        if let Some(r) = self.attempt(|p| p.funcref()) {
            return Ok(Arg::FuncRef(r));
        }
        Ok(Arg::Expr(self.expression()?))
    }

    fn lamtype(&mut self) -> R<(Option<Type>, String)> {
        let i = self.i;
        if let Ok(ty) = self.decltype() {
            if let Tok::Id(name) = self.peek().clone() {
                self.bump();
                return Ok((Some(ty), name));
            }
        }
        self.i = i;
        Ok((None, self.ident()?))
    }

    /// Up to and including the `->`.
    fn lambda_header(&mut self) -> R<Vec<(Option<Type>, String)>> {
        let mut params = Vec::new();
        if self.eat(&Tok::Lp) {
            if !self.at(&Tok::Rp) {
                loop {
                    params.push(self.lamtype()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
            }
            self.expect(&Tok::Rp, ")")?;
        } else {
            params.push(self.lamtype()?);
        }
        self.expect(&Tok::Arrow, "->")?;
        Ok(params)
    }

    fn funcref(&mut self) -> R<FuncRef> {
        let pos = self.pos() as Pos;
        if self.eat(&Tok::This) {
            self.expect(&Tok::Ref, "::")?;
            let name = self.ident()?;
            return Ok(FuncRef {
                pos,
                ty: None,
                name: Some(name),
            });
        }
        let ty = self.decltype()?;
        self.expect(&Tok::Ref, "::")?;
        let name = if self.eat(&Tok::New) {
            None
        } else {
            Some(self.ident()?)
        };
        Ok(FuncRef {
            pos,
            ty: Some(ty),
            name,
        })
    }
}

fn is_primitive(name: &str) -> bool {
    matches!(
        name,
        "boolean" | "byte" | "short" | "char" | "int" | "long" | "float" | "double"
    )
}

fn describe(t: &Tok) -> String {
    match t {
        Tok::Eof => "the end of the script".to_string(),
        Tok::Id(s) | Tok::DotId(s) | Tok::Primitive(s) => format!("[{s}]"),
        Tok::Str(_) => "a string".to_string(),
        other => format!("{other:?}"),
    }
}
