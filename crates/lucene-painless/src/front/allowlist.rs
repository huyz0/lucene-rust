//! Painless's allowlist: what a script may name, read from
//! `allowlist.tsv`, which `scripts/gen-painless-allowlist.py` writes from
//! OpenSearch's `spi/*.txt` and the JDK's own class hierarchy.
//!
//! The analyzer types every call, field and constructor against this table
//! and never against the runtime library, so a member the allowlist does not
//! list is refused even when the runtime could serve it.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

const TABLE: &str = include_str!("allowlist.tsv");

/// A method, as the allowlist lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodSig {
    pub ret: String,
    pub params: Vec<String>,
    pub is_static: bool,
    /// `random`, `currentTimeMillis`, ...: a script that calls it is not a
    /// pure function of its inputs.
    pub nondeterministic: bool,
    /// The class whose static method this is, for a method Painless adds to
    /// the receiver's class (`Augmentation`).
    pub augment: Option<String>,
    /// The allowlist file that lists it, which names the contexts that serve it.
    pub file: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSig {
    pub ty: String,
    pub is_static: bool,
    pub file: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CtorSig {
    pub params: Vec<String>,
    pub file: String,
}

/// A static import (`saturation(double, double)`): a function a script
/// calls by its bare name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Import {
    pub name: String,
    pub ret: String,
    pub params: Vec<String>,
    /// `from_class` (a static method) or `bound_to` (a class binding).
    pub kind: String,
    pub target: String,
    pub file: String,
}

#[derive(Debug, Clone, Default)]
pub struct Class {
    pub file: String,
    /// Not reachable by its short name (`ScoreScript`, `int`).
    pub no_import: bool,
    /// Every allowlisted class this one is assignable to.
    pub parents: Vec<String>,
    pub ctors: Vec<CtorSig>,
    pub methods: BTreeMap<String, Vec<MethodSig>>,
    pub fields: BTreeMap<String, FieldSig>,
}

#[derive(Debug, Default)]
pub struct Allowlist {
    classes: BTreeMap<String, Class>,
    imports: Vec<Import>,
    /// Canonical short name (`Map.Entry`) to class.
    shorts: HashMap<String, String>,
}

fn class_mut<'a>(a: &'a mut Allowlist, name: &str, n: usize) -> Result<&'a mut Class, String> {
    a.classes
        .get_mut(name)
        .ok_or_else(|| format!("line {}: unknown class {name}", n + 1))
}

fn params(s: &str) -> Vec<String> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(',').map(str::to_string).collect()
    }
}

impl Allowlist {
    /// The table the build embeds.
    pub fn get() -> &'static Allowlist {
        static TABLE_ONCE: OnceLock<Allowlist> = OnceLock::new();
        TABLE_ONCE.get_or_init(|| Allowlist::parse(TABLE).expect("allowlist.tsv is well formed"))
    }

    fn parse(text: &str) -> Result<Allowlist, String> {
        let mut a = Allowlist::default();
        for (n, line) in text.lines().enumerate() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let bad = || format!("line {}: {line}", n + 1);
            match f.first().copied() {
                Some("class") if f.len() == 4 => {
                    let no_import = f[3] == "no_import";
                    if !no_import {
                        let short = f[1].rsplit('.').next().unwrap_or(f[1]).replace('$', ".");
                        a.shorts.insert(short, f[1].to_string());
                    }
                    a.classes.insert(
                        f[1].to_string(),
                        Class {
                            file: f[2].to_string(),
                            no_import,
                            ..Class::default()
                        },
                    );
                }
                Some("parent") if f.len() == 3 => {
                    class_mut(&mut a, f[1], n)?.parents.push(f[2].to_string())
                }
                Some("ctor") if f.len() == 4 => class_mut(&mut a, f[1], n)?.ctors.push(CtorSig {
                    params: params(f[2]),
                    file: f[3].to_string(),
                }),
                Some("method") if f.len() == 7 => {
                    let flags: Vec<&str> = f[5].split(',').collect();
                    let sig = MethodSig {
                        ret: f[3].to_string(),
                        params: params(f[4]),
                        is_static: flags.contains(&"static"),
                        nondeterministic: flags.contains(&"nondeterministic"),
                        augment: flags
                            .iter()
                            .find_map(|x| x.strip_prefix("augment="))
                            .map(str::to_string),
                        file: f[6].to_string(),
                    };
                    class_mut(&mut a, f[1], n)?
                        .methods
                        .entry(f[2].to_string())
                        .or_default()
                        .push(sig);
                }
                Some("field") if f.len() == 6 => {
                    let sig = FieldSig {
                        ty: f[3].to_string(),
                        is_static: f[4].contains("static"),
                        file: f[5].to_string(),
                    };
                    class_mut(&mut a, f[1], n)?
                        .fields
                        .insert(f[2].to_string(), sig);
                }
                Some("import") if f.len() == 7 => a.imports.push(Import {
                    name: f[1].to_string(),
                    ret: f[2].to_string(),
                    params: params(f[3]),
                    kind: f[4].to_string(),
                    target: f[5].to_string(),
                    file: f[6].to_string(),
                }),
                _ => return Err(bad()),
            }
        }
        Ok(a)
    }

    pub fn class(&self, name: &str) -> Option<&Class> {
        self.classes.get(name)
    }

    /// The class a script's type name means: a canonical short name of an
    /// importable class (`String`, `Map.Entry`), or a full name.
    pub fn resolve(&self, written: &str) -> Option<&str> {
        if let Some(full) = self.shorts.get(written) {
            return Some(full);
        }
        self.classes.get_key_value(written).map(|(k, _)| k.as_str())
    }

    pub fn imports(&self) -> &[Import] {
        &self.imports
    }

    pub fn classes(&self) -> impl Iterator<Item = (&str, &Class)> {
        self.classes.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// The methods named `name` that a value of static type `class` has: its
    /// own, then those it inherits from every allowlisted class it is
    /// assignable to, a subclass's overload hiding a parent's of the same
    /// parameters.
    pub fn methods(&self, class: &str, name: &str) -> Vec<&MethodSig> {
        let Some(c) = self.classes.get(class) else {
            return Vec::new();
        };
        let mut out: Vec<&MethodSig> = Vec::new();
        let chain = std::iter::once(class).chain(c.parents.iter().map(String::as_str));
        for owner in chain {
            if let Some(list) = self.classes.get(owner).and_then(|c| c.methods.get(name)) {
                for m in list {
                    if !out.iter().any(|seen| seen.params == m.params) {
                        out.push(m);
                    }
                }
            }
        }
        out
    }

    /// The field `name` of `class`, or one it inherits.
    pub fn field(&self, class: &str, name: &str) -> Option<&FieldSig> {
        let c = self.classes.get(class)?;
        std::iter::once(class)
            .chain(c.parents.iter().map(String::as_str))
            .find_map(|owner| self.classes.get(owner).and_then(|c| c.fields.get(name)))
    }

    /// The constructors of `class` (never inherited).
    pub fn ctors(&self, class: &str) -> &[CtorSig] {
        self.classes.get(class).map_or(&[], |c| c.ctors.as_slice())
    }

    /// Whether a value of class `from` may be used as a `to`: the same class
    /// or one of its allowlisted ancestors.
    pub fn is_assignable(&self, from: &str, to: &str) -> bool {
        from == to
            || self
                .classes
                .get(from)
                .is_some_and(|c| c.parents.iter().any(|p| p == to))
    }
}
