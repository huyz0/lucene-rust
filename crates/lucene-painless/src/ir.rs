//! The `PIR1` layout: a Painless script's IR tree as the plugin writes it
//! (`org.lucenerust.opensearch.painless.PainlessIr`), after Painless's own
//! front end resolved every type, cast, method and `def` site.
//!
//! Little-endian; a string is an `i32` byte length (`-1` for null) and
//! UTF-8; a type is its JVM binary name (`int`, `java.lang.String`,
//! `[Ljava.lang.Object;`, `org.opensearch.painless.lookup.def`); a node is
//! its tag byte, its fields, then its children (`0` for an absent child).

use crate::{Error, Result};

/// A method an IR node calls: Painless's `PainlessMethod`.
#[derive(Debug, Clone, PartialEq)]
pub struct Method {
    /// The Painless class the method belongs to (`targetClass`).
    pub target: String,
    pub name: String,
    /// The Java class that declares it: an augmentation class for an
    /// augmented method.
    pub declaring: String,
    pub is_static: bool,
    pub params: Vec<String>,
    pub ret: String,
}

/// A constructor: its class and parameter types.
#[derive(Debug, Clone, PartialEq)]
pub struct Constructor {
    pub class: String,
    pub params: Vec<String>,
}

/// `PainlessCast`.
#[derive(Debug, Clone, PartialEq)]
pub struct Cast {
    pub from: String,
    pub to: String,
    pub explicit: bool,
    pub unbox_from: Option<String>,
    pub unbox_to: Option<String>,
    pub box_from: Option<String>,
    pub box_to: Option<String>,
}

/// A field of a class (`PainlessField`).
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub class: String,
    pub name: String,
    pub is_static: bool,
    pub ty: String,
}

/// A constant value.
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Null,
    Bool(bool),
    Byte(i8),
    Short(i16),
    Char(u16),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Str(String),
    /// A compiled regex: its pattern and `java.util.regex.Pattern` flags.
    Pattern(String, i32),
    /// A script parameter's list (an `ArrayList`).
    List(Vec<Constant>),
    /// A script parameter's map (a `HashMap`), in the order Java iterated it.
    Map(Vec<(Constant, Constant)>),
}

/// The target of an `InvokeCallMemberNode`.
#[derive(Debug, Clone, PartialEq)]
pub enum Member {
    /// A function of the script: one the user wrote, a lambda, or (when
    /// `internal`) one of the script class's own methods (`get_score`,
    /// `getDoc`, `getParams`).
    Local {
        name: String,
        internal: bool,
        is_static: bool,
        params: Vec<String>,
        ret: String,
    },
    /// A static method imported into the script (`saturation`).
    Imported(Method),
    /// A class binding (`decayDateGauss`): the class is constructed from
    /// the first `ctor_arity` arguments (after `offset` script-instance
    /// arguments) once per script instance, then `method` takes the rest.
    ClassBinding {
        class: String,
        method: String,
        ctor_arity: i32,
        offset: i32,
        params: Vec<String>,
        ret: String,
    },
    /// An instance binding: a method of an object the context provides.
    InstanceBinding {
        class: String,
        method: String,
        params: Vec<String>,
        ret: String,
    },
}

/// A method reference or lambda (`FunctionRef`).
#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    pub interface_method: String,
    pub delegate_class: String,
    pub delegate_interface: bool,
    pub delegate_augmented: bool,
    pub invoke_type: i32,
    pub delegate_method: String,
    pub delegate_params: Vec<String>,
    pub delegate_ret: String,
    pub interface_params: Vec<String>,
    pub interface_ret: String,
    pub factory_arity: i32,
}

/// An IR node. Expression variants carry their Painless type.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Block(Vec<Node>),
    DeclarationBlock(Vec<Node>),
    Declaration {
        name: String,
        ty: String,
        init: Option<Box<Node>>,
    },
    StatementExpression(Box<Node>),
    Return(Option<Box<Node>>),
    If {
        cond: Box<Node>,
        then: Box<Node>,
    },
    IfElse {
        cond: Box<Node>,
        then: Box<Node>,
        otherwise: Box<Node>,
    },
    While {
        continuous: bool,
        cond: Option<Box<Node>>,
        body: Option<Box<Node>>,
    },
    DoWhile {
        continuous: bool,
        cond: Option<Box<Node>>,
        body: Option<Box<Node>>,
    },
    For {
        continuous: bool,
        init: Option<Box<Node>>,
        cond: Option<Box<Node>>,
        after: Option<Box<Node>>,
        body: Option<Box<Node>>,
    },
    ForEachArray {
        var_ty: String,
        var: String,
        cast: Option<Cast>,
        array_ty: String,
        indexed_ty: String,
        iterable: Box<Node>,
        body: Option<Box<Node>>,
    },
    ForEachIterable {
        var_ty: String,
        var: String,
        cast: Option<Cast>,
        method: Option<Method>,
        iterable: Box<Node>,
        body: Option<Box<Node>>,
    },
    Break,
    Continue,
    Throw(Box<Node>),
    Try {
        body: Box<Node>,
        catches: Vec<Node>,
    },
    Catch {
        exception: String,
        symbol: String,
        body: Option<Box<Node>>,
    },

    BinaryImpl {
        ty: String,
        left: Box<Node>,
        right: Box<Node>,
    },
    BinaryMath {
        ty: String,
        op: String,
        binary_ty: String,
        shift_ty: Option<String>,
        flags: i32,
        regex_limit: i32,
        left: Box<Node>,
        right: Box<Node>,
    },
    Boolean {
        ty: String,
        op: String,
        left: Box<Node>,
        right: Box<Node>,
    },
    Comparison {
        ty: String,
        op: String,
        cmp_ty: String,
        left: Box<Node>,
        right: Box<Node>,
    },
    Conditional {
        ty: String,
        cond: Box<Node>,
        left: Box<Node>,
        right: Box<Node>,
    },
    Elvis {
        ty: String,
        left: Box<Node>,
        right: Box<Node>,
    },
    Cast {
        ty: String,
        cast: Option<Cast>,
        child: Box<Node>,
    },
    Constant {
        ty: String,
        value: Constant,
    },
    Null {
        ty: String,
    },
    LoadVariable {
        ty: String,
        name: String,
    },
    StoreVariable {
        ty: String,
        store_ty: String,
        name: String,
        child: Box<Node>,
    },
    UnaryMath {
        ty: String,
        op: String,
        unary_ty: Option<String>,
        cat: bool,
        explicit: bool,
        child: Box<Node>,
    },
    Instanceof {
        ty: String,
        instance_ty: String,
        child: Box<Node>,
    },
    Dup {
        ty: String,
        size: i32,
        depth: i32,
        child: Box<Node>,
    },
    FlipArrayIndex {
        ty: String,
        child: Box<Node>,
    },
    FlipCollectionIndex {
        ty: String,
        child: Box<Node>,
    },
    FlipDefIndex {
        ty: String,
        child: Box<Node>,
    },
    InvokeCall {
        ty: String,
        method: Option<Method>,
        boxed: Option<String>,
        args: Vec<Node>,
    },
    InvokeCallDef {
        ty: String,
        name: String,
        args: Vec<Node>,
    },
    InvokeCallMember {
        ty: String,
        member: Member,
        args: Vec<Node>,
    },
    ListInit {
        ty: String,
        ctor: Option<Constructor>,
        method: Option<Method>,
        args: Vec<Node>,
    },
    MapInit {
        ty: String,
        ctor: Option<Constructor>,
        method: Option<Method>,
        keys: Vec<Node>,
        values: Vec<Node>,
    },
    NewArray {
        ty: String,
        initialize: bool,
        args: Vec<Node>,
    },
    NewObject {
        ty: String,
        ctor: Option<Constructor>,
        read: bool,
        args: Vec<Node>,
    },
    LoadBrace {
        ty: String,
    },
    LoadBraceDef {
        ty: String,
        index_ty: String,
    },
    LoadDotArrayLength {
        ty: String,
    },
    LoadDotDef {
        ty: String,
        name: String,
    },
    LoadDot {
        ty: String,
        field: Field,
    },
    LoadDotShortcut {
        ty: String,
        getter: Option<Method>,
    },
    LoadListShortcut {
        ty: String,
        getter: Option<Method>,
    },
    LoadMapShortcut {
        ty: String,
        getter: Option<Method>,
    },
    StoreBrace {
        ty: String,
        store_ty: String,
        child: Box<Node>,
    },
    StoreBraceDef {
        ty: String,
        store_ty: String,
        index_ty: String,
        child: Box<Node>,
    },
    StoreDotDef {
        ty: String,
        store_ty: String,
        name: String,
        child: Box<Node>,
    },
    StoreDot {
        ty: String,
        store_ty: String,
        field: Field,
        child: Box<Node>,
    },
    StoreDotShortcut {
        ty: String,
        store_ty: String,
        setter: Option<Method>,
        child: Box<Node>,
    },
    StoreListShortcut {
        ty: String,
        store_ty: String,
        setter: Option<Method>,
        child: Box<Node>,
    },
    StoreMapShortcut {
        ty: String,
        store_ty: String,
        setter: Option<Method>,
        child: Box<Node>,
    },
    StringConcat {
        ty: String,
        args: Vec<Node>,
    },
    NullSafeSub {
        ty: String,
        child: Box<Node>,
    },
    Static {
        ty: String,
    },
    LoadFieldMember {
        ty: String,
        name: String,
        is_static: bool,
    },
    StoreFieldMember {
        ty: String,
        store_ty: String,
        name: String,
        is_static: bool,
        child: Box<Node>,
    },
    TypedInterfaceReference {
        ty: String,
        reference: Reference,
        captures: Vec<String>,
    },
    DefInterfaceReference {
        ty: String,
        encoding: String,
        captures: Vec<String>,
    },
    TypedCaptureReference {
        ty: String,
        method: String,
        captures: Vec<String>,
    },
}

/// A function of the script: `execute`, one the user declared, or a lambda.
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub ret: String,
    pub param_types: Vec<String>,
    pub param_names: Vec<String>,
    pub is_static: bool,
    pub synthetic: bool,
    pub max_loop_counter: i32,
    pub body: Node,
}

/// A decoded script.
#[derive(Debug, Clone, PartialEq)]
pub struct Script {
    pub functions: Vec<Function>,
}

/// Deepest nesting the decoder follows; Painless's own parser refuses far
/// shallower trees, so this only stops a corrupt blob.
const MAX_DEPTH: usize = 512;

pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| Error::Ir("truncated".into()))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub(crate) fn at_end(&self) -> bool {
        self.pos == self.buf.len()
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn bool(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(Error::Ir(format!("bool {v}"))),
        }
    }

    pub(crate) fn i32(&mut self) -> Result<i32> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(crate) fn i64(&mut self) -> Result<i64> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(i64::from_le_bytes(a))
    }

    pub(crate) fn len(&mut self) -> Result<usize> {
        let n = self.i32()?;
        let n = usize::try_from(n).map_err(|_| Error::Ir(format!("length {n}")))?;
        if n > self.buf.len() - self.pos {
            return Err(Error::Ir("truncated".into()));
        }
        Ok(n)
    }

    pub(crate) fn opt_str(&mut self) -> Result<Option<String>> {
        let n = self.i32()?;
        if n == -1 {
            return Ok(None);
        }
        let n = usize::try_from(n).map_err(|_| Error::Ir(format!("length {n}")))?;
        let b = self.take(n)?;
        String::from_utf8(b.to_vec())
            .map(Some)
            .map_err(|_| Error::Ir("utf-8".into()))
    }

    pub(crate) fn str(&mut self) -> Result<String> {
        self.opt_str()?
            .ok_or_else(|| Error::Ir("null string".into()))
    }

    fn types(&mut self) -> Result<Vec<String>> {
        let n = self.len()?;
        (0..n).map(|_| self.str()).collect()
    }

    fn method(&mut self) -> Result<Option<Method>> {
        if !self.bool()? {
            return Ok(None);
        }
        Ok(Some(Method {
            target: self.str()?,
            name: self.str()?,
            declaring: self.str()?,
            is_static: self.bool()?,
            params: self.types()?,
            ret: self.str()?,
        }))
    }

    fn constructor(&mut self) -> Result<Option<Constructor>> {
        if !self.bool()? {
            return Ok(None);
        }
        Ok(Some(Constructor {
            class: self.str()?,
            params: self.types()?,
        }))
    }

    fn cast(&mut self) -> Result<Option<Cast>> {
        if !self.bool()? {
            return Ok(None);
        }
        Ok(Some(Cast {
            // Null for an unbox-then-box cast, which names only those types.
            from: self.opt_str()?.unwrap_or_default(),
            to: self.opt_str()?.unwrap_or_default(),
            explicit: self.bool()?,
            unbox_from: self.opt_str()?,
            unbox_to: self.opt_str()?,
            box_from: self.opt_str()?,
            box_to: self.opt_str()?,
        }))
    }

    fn field(&mut self) -> Result<Field> {
        Ok(Field {
            class: self.str()?,
            name: self.str()?,
            is_static: self.bool()?,
            ty: self.str()?,
        })
    }

    pub(crate) fn constant(&mut self) -> Result<Constant> {
        Ok(match self.u8()? {
            0 => Constant::Null,
            1 => Constant::Bool(self.bool()?),
            2 => Constant::Byte(self.i32()? as i8),
            3 => Constant::Short(self.i32()? as i16),
            4 => Constant::Char(self.i32()? as u16),
            5 => Constant::Int(self.i32()?),
            6 => Constant::Long(self.i64()?),
            7 => Constant::Float(f32::from_bits(self.i32()? as u32)),
            8 => Constant::Double(f64::from_bits(self.i64()? as u64)),
            9 => Constant::Str(self.str()?),
            10 => Constant::Pattern(self.str()?, self.i32()?),
            11 => {
                let n = self.len()?;
                let mut items = Vec::with_capacity(n.min(self.remaining()));
                for _ in 0..n {
                    items.push(self.constant()?);
                }
                Constant::List(items)
            }
            12 => {
                let n = self.len()?;
                let mut entries = Vec::with_capacity(n.min(self.remaining()));
                for _ in 0..n {
                    let k = self.constant()?;
                    entries.push((k, self.constant()?));
                }
                Constant::Map(entries)
            }
            t => return Err(Error::Ir(format!("constant tag {t}"))),
        })
    }

    fn names(&mut self) -> Result<Vec<String>> {
        let n = self.len()?;
        (0..n).map(|_| self.str()).collect()
    }

    fn nodes(&mut self, depth: usize) -> Result<Vec<Node>> {
        let n = self.len()?;
        let mut v = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            v.push(
                self.node(depth)?
                    .ok_or_else(|| Error::Ir("absent list node".into()))?,
            );
        }
        Ok(v)
    }

    fn child(&mut self, depth: usize) -> Result<Box<Node>> {
        self.node(depth)?
            .map(Box::new)
            .ok_or_else(|| Error::Ir("absent child".into()))
    }

    fn opt_child(&mut self, depth: usize) -> Result<Option<Box<Node>>> {
        Ok(self.node(depth)?.map(Box::new))
    }

    fn node(&mut self, depth: usize) -> Result<Option<Node>> {
        if depth > MAX_DEPTH {
            return Err(Error::Ir("too deep".into()));
        }
        let d = depth + 1;
        let tag = self.u8()?;
        Ok(Some(match tag {
            0 => return Ok(None),
            1 => Node::Block(self.nodes(d)?),
            2 => Node::DeclarationBlock(self.nodes(d)?),
            3 => Node::Declaration {
                name: self.str()?,
                ty: self.str()?,
                init: self.opt_child(d)?,
            },
            4 => Node::StatementExpression(self.child(d)?),
            5 => Node::Return(self.opt_child(d)?),
            6 => Node::If {
                cond: self.child(d)?,
                then: self.child(d)?,
            },
            7 => Node::IfElse {
                cond: self.child(d)?,
                then: self.child(d)?,
                otherwise: self.child(d)?,
            },
            8 => Node::While {
                continuous: self.bool()?,
                cond: self.opt_child(d)?,
                body: self.opt_child(d)?,
            },
            9 => Node::DoWhile {
                continuous: self.bool()?,
                cond: self.opt_child(d)?,
                body: self.opt_child(d)?,
            },
            10 => Node::For {
                continuous: self.bool()?,
                init: self.opt_child(d)?,
                cond: self.opt_child(d)?,
                after: self.opt_child(d)?,
                body: self.opt_child(d)?,
            },
            11 => Node::ForEachArray {
                var_ty: self.str()?,
                var: self.str()?,
                cast: self.cast()?,
                array_ty: self.str()?,
                indexed_ty: self.str()?,
                iterable: self.child(d)?,
                body: self.opt_child(d)?,
            },
            12 => Node::ForEachIterable {
                var_ty: self.str()?,
                var: self.str()?,
                cast: self.cast()?,
                method: self.method()?,
                iterable: self.child(d)?,
                body: self.opt_child(d)?,
            },
            13 => Node::Break,
            14 => Node::Continue,
            15 => Node::Throw(self.child(d)?),
            16 => Node::Try {
                body: self.child(d)?,
                catches: self.nodes(d)?,
            },
            17 => Node::Catch {
                exception: self.str()?,
                symbol: self.str()?,
                body: self.opt_child(d)?,
            },
            32 => Node::BinaryImpl {
                ty: self.str()?,
                left: self.child(d)?,
                right: self.child(d)?,
            },
            33 => Node::BinaryMath {
                ty: self.str()?,
                op: self.str()?,
                binary_ty: self.str()?,
                shift_ty: self.opt_str()?,
                flags: self.i32()?,
                regex_limit: self.i32()?,
                left: self.child(d)?,
                right: self.child(d)?,
            },
            34 => Node::Boolean {
                ty: self.str()?,
                op: self.str()?,
                left: self.child(d)?,
                right: self.child(d)?,
            },
            35 => Node::Comparison {
                ty: self.str()?,
                op: self.str()?,
                cmp_ty: self.str()?,
                left: self.child(d)?,
                right: self.child(d)?,
            },
            36 => Node::Conditional {
                ty: self.str()?,
                cond: self.child(d)?,
                left: self.child(d)?,
                right: self.child(d)?,
            },
            37 => Node::Elvis {
                ty: self.str()?,
                left: self.child(d)?,
                right: self.child(d)?,
            },
            38 => Node::Cast {
                ty: self.str()?,
                cast: self.cast()?,
                child: self.child(d)?,
            },
            39 => Node::Constant {
                ty: self.str()?,
                value: self.constant()?,
            },
            40 => Node::Null { ty: self.str()? },
            41 => Node::LoadVariable {
                ty: self.str()?,
                name: self.str()?,
            },
            42 => Node::StoreVariable {
                ty: self.str()?,
                store_ty: self.str()?,
                name: self.str()?,
                child: self.child(d)?,
            },
            43 => Node::UnaryMath {
                ty: self.str()?,
                op: self.str()?,
                unary_ty: self.opt_str()?,
                cat: self.bool()?,
                explicit: self.bool()?,
                child: self.child(d)?,
            },
            44 => Node::Instanceof {
                ty: self.str()?,
                instance_ty: self.str()?,
                child: self.child(d)?,
            },
            45 => Node::Dup {
                ty: self.str()?,
                size: self.i32()?,
                depth: self.i32()?,
                child: self.child(d)?,
            },
            46 => Node::FlipArrayIndex {
                ty: self.str()?,
                child: self.child(d)?,
            },
            47 => Node::FlipCollectionIndex {
                ty: self.str()?,
                child: self.child(d)?,
            },
            48 => Node::FlipDefIndex {
                ty: self.str()?,
                child: self.child(d)?,
            },
            49 => Node::InvokeCall {
                ty: self.str()?,
                method: self.method()?,
                boxed: self.opt_str()?,
                args: self.nodes(d)?,
            },
            50 => Node::InvokeCallDef {
                ty: self.str()?,
                name: self.str()?,
                args: self.nodes(d)?,
            },
            51 => {
                let ty = self.str()?;
                let member = match self.u8()? {
                    1 => Member::Local {
                        name: self.str()?,
                        internal: self.bool()?,
                        is_static: self.bool()?,
                        params: self.types()?,
                        ret: self.str()?,
                    },
                    2 => Member::Imported(
                        self.method()?
                            .ok_or_else(|| Error::Ir("imported method".into()))?,
                    ),
                    3 => Member::ClassBinding {
                        class: self.str()?,
                        method: self.str()?,
                        ctor_arity: self.i32()?,
                        offset: self.i32()?,
                        params: self.types()?,
                        ret: self.str()?,
                    },
                    4 => Member::InstanceBinding {
                        class: self.str()?,
                        method: self.str()?,
                        params: self.types()?,
                        ret: self.str()?,
                    },
                    k => return Err(Error::Ir(format!("member kind {k}"))),
                };
                Node::InvokeCallMember {
                    ty,
                    member,
                    args: self.nodes(d)?,
                }
            }
            52 => Node::ListInit {
                ty: self.str()?,
                ctor: self.constructor()?,
                method: self.method()?,
                args: self.nodes(d)?,
            },
            53 => Node::MapInit {
                ty: self.str()?,
                ctor: self.constructor()?,
                method: self.method()?,
                keys: self.nodes(d)?,
                values: self.nodes(d)?,
            },
            54 => Node::NewArray {
                ty: self.str()?,
                initialize: self.bool()?,
                args: self.nodes(d)?,
            },
            55 => Node::NewObject {
                ty: self.str()?,
                ctor: self.constructor()?,
                read: self.bool()?,
                args: self.nodes(d)?,
            },
            56 => Node::LoadBrace { ty: self.str()? },
            57 => Node::LoadBraceDef {
                ty: self.str()?,
                index_ty: self.str()?,
            },
            58 => Node::LoadDotArrayLength { ty: self.str()? },
            59 => Node::LoadDotDef {
                ty: self.str()?,
                name: self.str()?,
            },
            60 => Node::LoadDot {
                ty: self.str()?,
                field: self.field()?,
            },
            61 => Node::LoadDotShortcut {
                ty: self.str()?,
                getter: self.method()?,
            },
            62 => Node::LoadListShortcut {
                ty: self.str()?,
                getter: self.method()?,
            },
            63 => Node::LoadMapShortcut {
                ty: self.str()?,
                getter: self.method()?,
            },
            64 => Node::StoreBrace {
                ty: self.str()?,
                store_ty: self.str()?,
                child: self.child(d)?,
            },
            65 => Node::StoreBraceDef {
                ty: self.str()?,
                store_ty: self.str()?,
                index_ty: self.str()?,
                child: self.child(d)?,
            },
            66 => Node::StoreDotDef {
                ty: self.str()?,
                store_ty: self.str()?,
                name: self.str()?,
                child: self.child(d)?,
            },
            67 => Node::StoreDot {
                ty: self.str()?,
                store_ty: self.str()?,
                field: self.field()?,
                child: self.child(d)?,
            },
            68 => Node::StoreDotShortcut {
                ty: self.str()?,
                store_ty: self.str()?,
                setter: self.method()?,
                child: self.child(d)?,
            },
            69 => Node::StoreListShortcut {
                ty: self.str()?,
                store_ty: self.str()?,
                setter: self.method()?,
                child: self.child(d)?,
            },
            70 => Node::StoreMapShortcut {
                ty: self.str()?,
                store_ty: self.str()?,
                setter: self.method()?,
                child: self.child(d)?,
            },
            71 => Node::StringConcat {
                ty: self.str()?,
                args: self.nodes(d)?,
            },
            72 => Node::NullSafeSub {
                ty: self.str()?,
                child: self.child(d)?,
            },
            73 => Node::Static { ty: self.str()? },
            74 => Node::LoadFieldMember {
                ty: self.str()?,
                name: self.str()?,
                is_static: self.bool()?,
            },
            75 => Node::StoreFieldMember {
                ty: self.str()?,
                store_ty: self.str()?,
                name: self.str()?,
                is_static: self.bool()?,
                child: self.child(d)?,
            },
            76 => {
                let ty = self.str()?;
                let reference = Reference {
                    interface_method: self.str()?,
                    delegate_class: self.str()?,
                    delegate_interface: self.bool()?,
                    delegate_augmented: self.bool()?,
                    invoke_type: self.i32()?,
                    delegate_method: self.str()?,
                    delegate_params: self.types()?,
                    delegate_ret: self.str()?,
                    interface_params: self.types()?,
                    interface_ret: self.str()?,
                    factory_arity: self.i32()?,
                };
                Node::TypedInterfaceReference {
                    ty,
                    reference,
                    captures: self.names()?,
                }
            }
            77 => Node::DefInterfaceReference {
                ty: self.str()?,
                encoding: self.str()?,
                captures: self.names()?,
            },
            78 => Node::TypedCaptureReference {
                ty: self.str()?,
                method: self.str()?,
                captures: self.names()?,
            },
            t => return Err(Error::Ir(format!("node tag {t}"))),
        }))
    }
}

/// Decodes a constant on its own: a script's parameters, as the plugin
/// writes them (a map constant).
///
/// # Errors
/// [`Error::Ir`] on a malformed blob.
pub fn decode_constant(blob: &[u8]) -> Result<Constant> {
    let mut r = Reader::new(blob);
    let c = r.constant()?;
    if r.remaining() != 0 {
        return Err(Error::Ir("trailing bytes after a constant".into()));
    }
    Ok(c)
}

/// Decodes a `PIR1` blob.
///
/// # Errors
/// [`Error::Ir`] on a malformed blob.
pub fn decode(blob: &[u8]) -> Result<Script> {
    let mut r = Reader::new(blob);
    if r.take(4)? != b"PIR1" {
        return Err(Error::Ir("magic".into()));
    }
    let n = r.len()?;
    let mut functions = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        let name = r.str()?;
        let ret = r.str()?;
        let param_types = r.types()?;
        let param_names = r.names()?;
        let is_static = r.bool()?;
        let synthetic = r.bool()?;
        let max_loop_counter = r.i32()?;
        let body = r
            .node(0)?
            .ok_or_else(|| Error::Ir("function without a body".into()))?;
        functions.push(Function {
            name,
            ret,
            param_types,
            param_names,
            is_static,
            synthetic,
            max_loop_counter,
            body,
        });
    }
    if !r.at_end() {
        return Err(Error::Ir("trailing bytes".into()));
    }
    Ok(Script { functions })
}
