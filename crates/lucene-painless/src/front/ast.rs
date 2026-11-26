//! The syntax tree. Every node carries the byte offset it starts at.

/// A byte offset into the source.
pub type Pos = u32;

/// A whole script: its functions, then its statements.
#[derive(Debug, Clone, PartialEq)]
pub struct Script {
    pub functions: Vec<Function>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub pos: Pos,
    pub ret: Type,
    pub name: String,
    pub params: Vec<(Type, String)>,
    pub body: Block,
}

/// A written type: `def`, a primitive, or a dotted name, with array dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    pub pos: Pos,
    pub name: String,
    pub dims: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub pos: Pos,
    pub stmts: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub pos: Pos,
    pub kind: StmtKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    If {
        cond: Expr,
        then: Box<Stmt>,
        otherwise: Option<Box<Stmt>>,
    },
    While {
        cond: Expr,
        body: Option<Box<Stmt>>,
    },
    Do {
        body: Block,
        cond: Expr,
    },
    For {
        init: Option<ForInit>,
        cond: Option<Expr>,
        after: Option<Expr>,
        body: Option<Box<Stmt>>,
    },
    /// `for (T x : e)` and `for (x in e)` (`ty` is `None`).
    Each {
        ty: Option<Type>,
        name: String,
        iterable: Expr,
        body: Box<Stmt>,
    },
    Try {
        body: Block,
        catches: Vec<Catch>,
    },
    Decl(Vec<Declaration>),
    Continue,
    Break,
    Return(Option<Expr>),
    Throw(Expr),
    Expr(Expr),
    Block(Block),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ForInit {
    Decl(Vec<Declaration>),
    Expr(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub pos: Pos,
    pub ty: Type,
    pub name: String,
    pub init: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Catch {
    pub pos: Pos,
    pub ty: Type,
    pub name: String,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub pos: Pos,
    pub kind: ExprKind,
}

/// The base of a numeric literal as written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumKind {
    Octal,
    Hex,
    Integer,
    Decimal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    /// `=~`
    Find,
    /// `==~`
    Match,
    Lsh,
    Rsh,
    Ush,
    Lt,
    Lte,
    Gt,
    Gte,
    Eq,
    /// `===`
    EqRef,
    Ne,
    /// `!==`
    NeRef,
    BitAnd,
    Xor,
    BitOr,
    And,
    Or,
    /// `?:`
    Elvis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Not,
    BitNot,
    Plus,
    Minus,
}

/// `=` or a compound assignment; the operator it applies, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Set,
    With(BinOp),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    /// A numeric literal: its base, and its text as written (suffix included).
    Num(NumKind, String),
    Bool(bool),
    Null,
    /// A string literal, escapes resolved.
    Str(String),
    /// `/pattern/flags`
    Regex(String, String),
    List(Vec<Expr>),
    Map(Vec<(Expr, Expr)>),
    Var(String),
    /// `name(args)`: a call of a script function or an imported one.
    CallLocal(String, Vec<Arg>),
    NewObject(Type, Vec<Arg>),
    /// `new T[a][b]`: `dims` expressions, `ty` the element type without them.
    NewArray(Type, Vec<Expr>),
    /// `new T[] {a, b}`: `ty` is the array's own type (`T[]`).
    NewArrayInit(Type, Vec<Expr>),
    Call {
        recv: Box<Expr>,
        name: String,
        args: Vec<Arg>,
        null_safe: bool,
    },
    Field {
        recv: Box<Expr>,
        name: String,
        null_safe: bool,
    },
    Index(Box<Expr>, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    IncDec {
        prefix: bool,
        inc: bool,
        target: Box<Expr>,
    },
    Cast(Type, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    InstanceOf(Box<Expr>, Type),
    Cond(Box<Expr>, Box<Expr>, Box<Expr>),
    Assign(AssignOp, Box<Expr>, Box<Expr>),
    /// A parenthesised expression (kept: `(a) = 1` and casts care).
    Paren(Box<Expr>),
}

/// An argument of a call.
#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    Expr(Expr),
    Lambda(Lambda),
    FuncRef(FuncRef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lambda {
    pub pos: Pos,
    pub params: Vec<(Option<Type>, String)>,
    pub body: LambdaBody,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LambdaBody {
    Block(Block),
    Expr(Box<Expr>),
}

/// `T::name`, `T::new`, or `this::name`.
#[derive(Debug, Clone, PartialEq)]
pub struct FuncRef {
    pub pos: Pos,
    /// `None` for `this`.
    pub ty: Option<Type>,
    /// `None` for `new`.
    pub name: Option<String>,
}
