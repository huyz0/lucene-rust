//! Native code for an expression, generated with Cranelift.
//!
//! The [`Node`] tree is translated with the same typing the closure
//! evaluator has (`double`, `long`, `int`, converted with the JVM's `d2i`,
//! `d2l`, `l2i`) into two functions:
//!
//! * `scalar(reads) -> f64`, one document, each variable read through
//!   [`read_var`] (so a read is still lazy, positioned and may fail, as the
//!   closure evaluator's is);
//! * `batch(cols, n, out)`, every document of a batch, each variable loaded
//!   from its column.
//!
//! Default functions Cranelift has an instruction for (`abs`, `sqrt`,
//! `floor`, `ceil`, `max`, `min`) are inlined; the others are called through
//! their `fn` pointer, a custom function through its closure. A panic in a
//! variable read or a custom function is caught at the callback, carried
//! over the generated frames and resumed when the call returns.

use std::any::Any;
use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    types, AbiParam, Block, InstBuilder, MemFlagsData, Signature, StackSlotData, StackSlotKind,
    Type, Value,
};
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Module};

use super::{
    chain_kind, java_max, java_min, Bin, ChainKind, CustomFunction, FunctionBody, Node, Reads,
};

/// `scalar`'s signature.
type ScalarFn = unsafe extern "C" fn(*mut Reads<'_, '_>) -> f64;
/// `batch`'s signature.
type BatchFn = unsafe extern "C" fn(*const f64, usize, *mut f64);

/// An expression's generated code, and what it calls.
pub(crate) struct Code {
    module: Option<JITModule>,
    scalar: ScalarFn,
    batch: BatchFn,
    /// The custom functions the code calls, by the address of their box
    /// (boxed so the address stays put as the vector grows).
    #[allow(clippy::vec_box)]
    _custom: Vec<Box<CustomFunction>>,
}

// The module is only read once finalized, and its code calls only `Send +
// Sync` functions.
unsafe impl Send for Code {}
unsafe impl Sync for Code {}

impl Drop for Code {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            // SAFETY: the code is dropped with the last expression holding it
            unsafe { m.free_memory() };
        }
    }
}

thread_local! {
    /// A panic caught at a callback, resumed once the generated code returns.
    static PANIC: RefCell<Option<Box<dyn Any + Send>>> = const { RefCell::new(None) };
}

fn resume_panic() {
    if let Some(p) = PANIC.with(|p| p.borrow_mut().take()) {
        std::panic::resume_unwind(p);
    }
}

fn catch(f: impl FnOnce() -> f64) -> f64 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(p) => {
            PANIC.with(|slot| {
                slot.borrow_mut().get_or_insert(p);
            });
            0.0
        }
    }
}

impl Code {
    /// The value at the document `reads` is on.
    #[inline]
    pub(crate) fn scalar(&self, reads: &mut Reads<'_, '_>) -> f64 {
        // SAFETY: generated for this signature; `reads` outlives the call
        let v = unsafe { (self.scalar)(reads) };
        resume_panic();
        v
    }

    /// The values over a batch: `cols[i * n + j]` variable `i` at the `j`th.
    #[inline]
    pub(crate) fn batch(&self, cols: &[f64], out: &mut [f64]) {
        // SAFETY: the caller checked `cols` holds every variable's column
        unsafe { (self.batch)(cols.as_ptr(), out.len(), out.as_mut_ptr()) };
        resume_panic();
    }
}

unsafe extern "C" fn read_var(reads: *mut Reads<'_, '_>, i: usize) -> f64 {
    // SAFETY: `reads` is the pointer `Code::scalar` passed in
    catch(|| unsafe { (*reads).read(i) })
}

unsafe extern "C" fn call_f1(f: usize, a: f64) -> f64 {
    // SAFETY: `f` is a `fn(f64) -> f64`'s address
    let f: fn(f64) -> f64 = unsafe { std::mem::transmute(f) };
    f(a)
}

unsafe extern "C" fn call_f2(f: usize, a: f64, b: f64) -> f64 {
    // SAFETY: `f` is a `fn(f64, f64) -> f64`'s address
    let f: fn(f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
    f(a, b)
}

unsafe extern "C" fn call_f4(f: usize, a: f64, b: f64, c: f64, d: f64) -> f64 {
    // SAFETY: `f` is a `fn(f64, f64, f64, f64) -> f64`'s address
    let f: fn(f64, f64, f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
    f(a, b, c, d)
}

unsafe extern "C" fn call_any(f: *const CustomFunction, args: *const f64, n: usize) -> f64 {
    // SAFETY: `f` is a box `Code` keeps; `args` holds `n` values
    catch(|| unsafe { (*f)(std::slice::from_raw_parts(args, n)) })
}

extern "C" fn drem(a: f64, b: f64) -> f64 {
    a % b
}

/// The host's ISA, built once.
fn isa() -> Option<OwnedTargetIsa> {
    static ISA: OnceLock<Option<OwnedTargetIsa>> = OnceLock::new();
    ISA.get_or_init(|| {
        let mut flags = settings::builder();
        flags.set("opt_level", "speed").ok()?;
        flags.set("use_colocated_libcalls", "false").ok()?;
        flags.set("is_pic", "false").ok()?;
        cranelift_native::builder()
            .ok()?
            .finish(settings::Flags::new(flags))
            .ok()
    })
    .clone()
}

/// `root` as native code, or `None` where this host has no Cranelift
/// backend or code generation fails (the closure evaluator is used then).
pub(crate) fn compile(root: &Node) -> Option<Arc<Code>> {
    let isa = isa()?;
    let ptr = isa.pointer_type();
    let call_conv = isa.default_call_conv();
    let mut module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));
    let mut custom = Vec::new();
    let frontend = module.target_config();
    let mut fctx = FunctionBuilderContext::new();
    let mut ctx = module.make_context();

    // scalar(reads) -> f64
    ctx.func.signature.params.push(AbiParam::new(ptr));
    ctx.func.signature.returns.push(AbiParam::new(types::F64));
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let reads = b.block_params(entry)[0];
        let mut g = Gen {
            b,
            ptr,
            call_conv,
            vars: Vars::Read(reads),
            custom: &mut custom,
        };
        let v = g.double(root);
        g.b.ins().return_(&[v]);
        g.b.finalize(frontend);
    }
    let scalar_id = module
        .declare_anonymous_function(&ctx.func.signature)
        .ok()?;
    module.define_function(scalar_id, &mut ctx).ok()?;
    module.clear_context(&mut ctx);

    // batch(cols, n, out)
    for _ in 0..3 {
        ctx.func.signature.params.push(AbiParam::new(ptr));
    }
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let entry = b.create_block();
        let header = b.create_block();
        let body = b.create_block();
        let exit = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let [cols, n, out] = [0, 1, 2].map(|i| b.block_params(entry)[i]);
        let zero = b.ins().iconst(ptr, 0);
        b.ins().jump(header, &[zero.into()]);

        b.append_block_param(header, ptr);
        b.switch_to_block(header);
        let j = b.block_params(header)[0];
        let more = b.ins().icmp(IntCC::UnsignedLessThan, j, n);
        b.ins().brif(more, body, &[], exit, &[]);

        b.switch_to_block(body);
        b.seal_block(body);
        let mut g = Gen {
            b,
            ptr,
            call_conv,
            vars: Vars::Columns { cols, n, j },
            custom: &mut custom,
        };
        let v = g.double(root);
        let offset = g.b.ins().ishl_imm_s(j, 3);
        let at = g.b.ins().iadd(out, offset);
        g.b.ins().store(MemFlagsData::trusted(), v, at, 0);
        let next = g.b.ins().iadd_imm_s(j, 1);
        g.b.ins().jump(header, &[next.into()]);
        g.b.seal_block(header);

        g.b.switch_to_block(exit);
        g.b.seal_block(exit);
        g.b.ins().return_(&[]);
        g.b.finalize(frontend);
    }
    let batch_id = module
        .declare_anonymous_function(&ctx.func.signature)
        .ok()?;
    module.define_function(batch_id, &mut ctx).ok()?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().ok()?;

    let scalar = module.get_finalized_function(scalar_id);
    let batch = module.get_finalized_function(batch_id);
    // SAFETY: both were generated for these signatures
    let (scalar, batch) = unsafe {
        (
            std::mem::transmute::<*const u8, ScalarFn>(scalar),
            std::mem::transmute::<*const u8, BatchFn>(batch),
        )
    };
    Some(Arc::new(Code {
        module: Some(module),
        scalar,
        batch,
        _custom: custom,
    }))
}

/// Where a variable's value comes from.
#[derive(Clone, Copy)]
enum Vars {
    /// `read_var(reads, i)`.
    Read(Value),
    /// `cols[i * n + j]`.
    Columns { cols: Value, n: Value, j: Value },
}

struct Gen<'a, 'f> {
    b: FunctionBuilder<'f>,
    ptr: Type,
    call_conv: cranelift_codegen::isa::CallConv,
    vars: Vars,
    #[allow(clippy::vec_box)]
    custom: &'a mut Vec<Box<CustomFunction>>,
}

/// The address of a function, to compare a default function against.
fn addr1(f: fn(f64) -> f64) -> usize {
    f as usize
}

fn addr2(f: fn(f64, f64) -> f64) -> usize {
    f as usize
}

impl Gen<'_, '_> {
    fn sig(&self, params: &[Type], ret: Option<Type>) -> Signature {
        let mut s = Signature::new(self.call_conv);
        s.params.extend(params.iter().map(|&t| AbiParam::new(t)));
        s.returns.extend(ret.map(AbiParam::new));
        s
    }

    /// A call of the host function at `addr`.
    fn call(&mut self, addr: usize, params: &[Type], args: &[Value]) -> Value {
        let sig = self.sig(params, Some(types::F64));
        let sig = self.b.import_signature(sig);
        let callee = self.b.ins().iconst(self.ptr, addr as i64);
        let call = self.b.ins().call_indirect(sig, callee, args);
        self.b.inst_results(call)[0]
    }

    fn read(&mut self, i: usize) -> Value {
        match self.vars {
            Vars::Read(reads) => {
                let i = self.b.ins().iconst(self.ptr, i as i64);
                self.call(
                    read_var as *const () as usize,
                    &[self.ptr, self.ptr],
                    &[reads, i],
                )
            }
            Vars::Columns { cols, n, j } => {
                let at = self.b.ins().imul_imm_s(n, i as i64);
                let at = self.b.ins().iadd(at, j);
                let at = self.b.ins().ishl_imm_s(at, 3);
                let at = self.b.ins().iadd(cols, at);
                self.b
                    .ins()
                    .load(types::F64, MemFlagsData::trusted(), at, 0)
            }
        }
    }

    fn fconst(&mut self, v: f64) -> Value {
        self.b.ins().f64const(v)
    }

    fn d2i(&mut self, v: Value) -> Value {
        self.b.ins().fcvt_to_sint_sat(types::I32, v)
    }

    fn d2l(&mut self, v: Value) -> Value {
        self.b.ins().fcvt_to_sint_sat(types::I64, v)
    }

    fn i2d(&mut self, v: Value) -> Value {
        self.b.ins().fcvt_from_sint(types::F64, v)
    }

    fn bool_int(&mut self, flag: Value) -> Value {
        self.b.ins().uextend(types::I32, flag)
    }

    fn arith(&mut self, op: Bin, a: Value, b: Value) -> Value {
        match op {
            Bin::Mul => self.b.ins().fmul(a, b),
            Bin::Div => self.b.ins().fdiv(a, b),
            Bin::Rem => self.call(
                drem as *const () as usize,
                &[types::F64, types::F64],
                &[a, b],
            ),
            Bin::Add => self.b.ins().fadd(a, b),
            _ => self.b.ins().fsub(a, b),
        }
    }

    fn compare(&mut self, op: Bin, a: Value, b: Value) -> Value {
        let cc = match op {
            Bin::Lt => FloatCC::LessThan,
            Bin::Lte => FloatCC::LessThanOrEqual,
            Bin::Gt => FloatCC::GreaterThan,
            Bin::Gte => FloatCC::GreaterThanOrEqual,
            Bin::Eq => FloatCC::Equal,
            _ => FloatCC::NotEqual,
        };
        let flag = self.b.ins().fcmp(cc, a, b);
        self.bool_int(flag)
    }

    fn shift(&mut self, op: Bin, v: Value, n: Value) -> Value {
        // Java masks a long shift's count to its low six bits
        let n = self.b.ins().band_imm_s(n, 63);
        let n = self.b.ins().sextend(types::I64, n);
        match op {
            Bin::Lsh => self.b.ins().ishl(v, n),
            Bin::Rsh => self.b.ins().sshr(v, n),
            _ => self.b.ins().ushr(v, n),
        }
    }

    fn bitwise(&mut self, op: Bin, a: Value, b: Value) -> Value {
        match op {
            Bin::BwAnd => self.b.ins().band(a, b),
            Bin::BwXor => self.b.ins().bxor(a, b),
            _ => self.b.ins().bor(a, b),
        }
    }

    /// `if c != 0 { then() } else { other() }`, of type `ty`.
    fn branch(
        &mut self,
        c: Value,
        ty: Type,
        then: impl FnOnce(&mut Self) -> Value,
        other: impl FnOnce(&mut Self) -> Value,
    ) -> Value {
        let (t, e, merge) = (
            self.b.create_block(),
            self.b.create_block(),
            self.b.create_block(),
        );
        self.b.append_block_param(merge, ty);
        self.b.ins().brif(c, t, &[], e, &[]);
        self.b.seal_block(t);
        self.b.seal_block(e);
        self.b.switch_to_block(t);
        let v = then(self);
        self.b.ins().jump(merge, &[v.into()]);
        self.b.switch_to_block(e);
        let v = other(self);
        self.b.ins().jump(merge, &[v.into()]);
        self.b.seal_block(merge);
        self.b.switch_to_block(merge);
        self.b.block_params(merge)[0]
    }

    /// `operands` short-circuited: `0` at the first zero for `&&`, `1` at the
    /// first non-zero for `||`.
    fn logic(&mut self, operands: &[&Node], stop_at_zero: bool) -> Value {
        let merge: Block = self.b.create_block();
        self.b.append_block_param(merge, types::I32);
        let stopped = self.b.ins().iconst(types::I32, i64::from(!stop_at_zero));
        for a in operands {
            let v = self.int(a);
            let next = self.b.create_block();
            if stop_at_zero {
                self.b.ins().brif(v, next, &[], merge, &[stopped.into()]);
            } else {
                self.b.ins().brif(v, merge, &[stopped.into()], next, &[]);
            }
            self.b.seal_block(next);
            self.b.switch_to_block(next);
        }
        let done = self.b.ins().iconst(types::I32, i64::from(stop_at_zero));
        self.b.ins().jump(merge, &[done.into()]);
        self.b.seal_block(merge);
        self.b.switch_to_block(merge);
        self.b.block_params(merge)[0]
    }

    fn call_node(&mut self, body: &FunctionBody, args: &[Node]) -> Value {
        let vals: Vec<Value> = args.iter().map(|a| self.double(a)).collect();
        let (f64t, ptr) = (types::F64, self.ptr);
        match body {
            FunctionBody::F1(f) => {
                let a = vals[0];
                let addr = addr1(*f);
                if addr == addr1(f64::abs) {
                    self.b.ins().fabs(a)
                } else if addr == addr1(f64::sqrt) {
                    self.b.ins().sqrt(a)
                } else if addr == addr1(f64::floor) {
                    self.b.ins().floor(a)
                } else if addr == addr1(f64::ceil) {
                    self.b.ins().ceil(a)
                } else {
                    let fv = self.b.ins().iconst(ptr, addr as i64);
                    self.call(call_f1 as *const () as usize, &[ptr, f64t], &[fv, a])
                }
            }
            FunctionBody::F2(f) => {
                let (a, b) = (vals[0], vals[1]);
                let addr = addr2(*f);
                if addr == addr2(java_max) {
                    self.b.ins().fmax(a, b)
                } else if addr == addr2(java_min) {
                    self.b.ins().fmin(a, b)
                } else {
                    let fv = self.b.ins().iconst(ptr, addr as i64);
                    self.call(
                        call_f2 as *const () as usize,
                        &[ptr, f64t, f64t],
                        &[fv, a, b],
                    )
                }
            }
            FunctionBody::F4(f) => {
                let fv = self.b.ins().iconst(ptr, *f as usize as i64);
                let args = [fv, vals[0], vals[1], vals[2], vals[3]];
                self.call(
                    call_f4 as *const () as usize,
                    &[ptr, f64t, f64t, f64t, f64t],
                    &args,
                )
            }
            FunctionBody::Any(f) => {
                let boxed = Box::new(f.clone());
                let fp: *const CustomFunction = &*boxed;
                self.custom.push(boxed);
                let size = u32::try_from(vals.len().max(1) * 8).expect("arity fits");
                let slot = self.b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    size,
                    3,
                ));
                for (k, v) in vals.iter().enumerate() {
                    self.b.ins().stack_store(ptr, *v, slot, (k * 8) as i32);
                }
                let args_ptr = self.b.ins().stack_addr(ptr, slot, 0);
                let fv = self.b.ins().iconst(ptr, fp as usize as i64);
                let n = self.b.ins().iconst(ptr, vals.len() as i64);
                self.call(
                    call_any as *const () as usize,
                    &[ptr, ptr, ptr],
                    &[fv, args_ptr, n],
                )
            }
        }
    }

    fn double(&mut self, node: &Node) -> Value {
        match node {
            Node::Integer(i) => self.fconst(*i as f64),
            Node::Decimal(d) => self.fconst(*d),
            Node::Variable(i) => self.read(*i),
            Node::Plus(e) => self.double(e),
            Node::Call(f, args) => self.call_node(&f.call, args),
            Node::Neg(e) => {
                let v = self.double(e);
                self.b.ins().fneg(v)
            }
            Node::Arith(op, l, r) => {
                let a = self.double(l);
                let b = self.double(r);
                self.arith(*op, a, b)
            }
            Node::BwNot(_) | Node::Shift(..) | Node::Bitwise(..) => {
                let v = self.long(node);
                self.i2d(v)
            }
            Node::Not(_) | Node::Compare(..) | Node::And(..) | Node::Or(..) => {
                let v = self.int(node);
                self.i2d(v)
            }
            Node::Conditional(c, a, e) => {
                let c = self.int(c);
                self.branch(c, types::F64, |g| g.double(a), |g| g.double(e))
            }
            Node::Chain(first, steps) => match chain_kind(steps) {
                ChainKind::Arith => self.arith_chain(first, steps),
                ChainKind::Shift | ChainKind::Bitwise => {
                    let v = self.long_chain(first, steps);
                    self.i2d(v)
                }
                _ => {
                    let v = self.int_chain(first, steps);
                    self.i2d(v)
                }
            },
        }
    }

    fn long(&mut self, node: &Node) -> Value {
        match node {
            Node::Integer(i) => self.b.ins().iconst(types::I64, *i),
            Node::Decimal(d) => self.b.ins().iconst(types::I64, *d as i64),
            Node::Variable(_) | Node::Call(..) | Node::Neg(_) | Node::Arith(..) => {
                let v = self.double(node);
                self.d2l(v)
            }
            Node::Plus(e) => self.long(e),
            Node::BwNot(e) => {
                let v = self.long(e);
                self.b.ins().bnot(v)
            }
            Node::Shift(op, l, n) => {
                let v = self.long(l);
                let n = self.int(n);
                self.shift(*op, v, n)
            }
            Node::Bitwise(op, l, r) => {
                let a = self.long(l);
                let b = self.long(r);
                self.bitwise(*op, a, b)
            }
            Node::Not(_) | Node::Compare(..) | Node::And(..) | Node::Or(..) => {
                let v = self.int(node);
                self.b.ins().sextend(types::I64, v)
            }
            Node::Conditional(c, a, e) => {
                let c = self.int(c);
                self.branch(c, types::I64, |g| g.long(a), |g| g.long(e))
            }
            Node::Chain(first, steps) => match chain_kind(steps) {
                ChainKind::Arith => {
                    let v = self.arith_chain(first, steps);
                    self.d2l(v)
                }
                ChainKind::Shift | ChainKind::Bitwise => self.long_chain(first, steps),
                _ => {
                    let v = self.int_chain(first, steps);
                    self.b.ins().sextend(types::I64, v)
                }
            },
        }
    }

    fn int(&mut self, node: &Node) -> Value {
        match node {
            Node::Integer(i) => self.b.ins().iconst(types::I32, i64::from(*i as i32)),
            Node::Decimal(d) => self.b.ins().iconst(types::I32, i64::from(*d as i32)),
            Node::Variable(_) | Node::Call(..) | Node::Neg(_) | Node::Arith(..) => {
                let v = self.double(node);
                self.d2i(v)
            }
            Node::Plus(e) => self.int(e),
            Node::BwNot(_) | Node::Shift(..) | Node::Bitwise(..) => {
                let v = self.long(node);
                self.b.ins().ireduce(types::I32, v)
            }
            Node::Not(e) => {
                let v = self.int(e);
                let flag = self.b.ins().icmp_imm_s(IntCC::Equal, v, 0);
                self.bool_int(flag)
            }
            Node::Compare(op, l, r) => {
                let a = self.double(l);
                let b = self.double(r);
                self.compare(*op, a, b)
            }
            Node::And(l, r) => self.logic(&[l, r], true),
            Node::Or(l, r) => self.logic(&[l, r], false),
            Node::Conditional(c, a, e) => {
                let c = self.int(c);
                self.branch(c, types::I32, |g| g.int(a), |g| g.int(e))
            }
            Node::Chain(first, steps) => match chain_kind(steps) {
                ChainKind::Arith => {
                    let v = self.arith_chain(first, steps);
                    self.d2i(v)
                }
                ChainKind::Shift | ChainKind::Bitwise => {
                    let v = self.long_chain(first, steps);
                    self.b.ins().ireduce(types::I32, v)
                }
                _ => self.int_chain(first, steps),
            },
        }
    }

    fn arith_chain(&mut self, first: &Node, steps: &[(Bin, Node)]) -> Value {
        let mut v = self.double(first);
        for (op, a) in steps {
            let b = self.double(a);
            v = self.arith(*op, v, b);
        }
        v
    }

    fn long_chain(&mut self, first: &Node, steps: &[(Bin, Node)]) -> Value {
        let mut v = self.long(first);
        let shift = chain_kind(steps) == ChainKind::Shift;
        for (op, a) in steps {
            v = if shift {
                let n = self.int(a);
                self.shift(*op, v, n)
            } else {
                let b = self.long(a);
                self.bitwise(*op, v, b)
            };
        }
        v
    }

    fn int_chain(&mut self, first: &Node, steps: &[(Bin, Node)]) -> Value {
        match chain_kind(steps) {
            ChainKind::Compare => {
                let mut v = self.double(first);
                let mut result = None;
                for (op, a) in steps {
                    let b = self.double(a);
                    let r = self.compare(*op, v, b);
                    v = self.i2d(r);
                    result = Some(r);
                }
                result.expect("a chain has steps")
            }
            kind => {
                let operands: Vec<&Node> = std::iter::once(first)
                    .chain(steps.iter().map(|(_, a)| a))
                    .collect();
                self.logic(&operands, kind == ChainKind::And)
            }
        }
    }
}
