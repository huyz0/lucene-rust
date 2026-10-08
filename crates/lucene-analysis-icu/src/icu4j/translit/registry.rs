//! `com.ibm.icu.text.TransliteratorRegistry` and `Transliterator`'s
//! static set-up: the IDs ICU ships (`translit/root.res`'s
//! `RuleBasedTransliteratorIDs`: rule files, internal rule files and
//! aliases, read and parsed on first use), the built-in transliterators,
//! the `Any-<script>` transliterators registered for every script target,
//! and `getInstance`'s ID resolution (compound IDs, filters, aliases,
//! script-name fallback of a source or target).
//!
//! Not ported: specs that are locales (`el-Latin`, which Java resolves
//! through `translit/el.res`) and script fallbacks a locale implies
//! (`UScript.getCode("ja")`), `registerInstance`/`registerFactory` for
//! callers, and display names.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::icu4j::coll::res::{pack_file, ResReader};
use crate::icu4j::normalizer2::{Mode, Normalizer2};
use crate::icu4j::translit::id::{self, id_to_stv, register_special_inverse, stv_to_id, SingleId};
use crate::icu4j::translit::parser::{self, Parsed};
use crate::icu4j::translit::rules::Data;
use crate::icu4j::translit::{
    any_transliterator, CaseKind, Kind, Transliterator, FORWARD, REVERSE,
};
use crate::icu4j::unicode_set::UnicodeSet;
use crate::icu4j::uprops;
use crate::{IcuError, IcuErrorKind};

/// A registry entry (Java's `Object[] { entry }`).
#[derive(Debug, Clone)]
enum Entry {
    /// `ResourceEntry`: rules not yet parsed.
    Rules { rules: Arc<str>, dir: i32 },
    /// `AliasEntry`.
    Alias(String),
    /// `RuleBasedTransliterator.Data`.
    Data(Arc<Data>),
    /// `CompoundRBTEntry`.
    Compound {
        id_blocks: Vec<String>,
        data: Vec<Arc<Data>>,
        filter: Option<UnicodeSet>,
    },
    /// A factory (`Transliterator.Factory`) or a class (`registerClass`).
    Factory(fn(&str) -> Result<Transliterator, IcuError>),
    /// A registered instance (`AnyTransliterator`, `BreakTransliterator`).
    Instance(Arc<Transliterator>),
}

/// A target and its variants.
type Variants = (String, Vec<String>);

/// `specDAG`: source -> target -> variants, case-insensitive keys keeping
/// the first spelling.
#[derive(Debug, Default)]
struct SpecDag {
    sources: Vec<(String, Vec<Variants>)>,
}

impl SpecDag {
    fn register(&mut self, source: &str, target: &str, variant: &str) {
        let si = match self
            .sources
            .iter()
            .position(|(s, _)| s.eq_ignore_ascii_case(source))
        {
            Some(i) => i,
            None => {
                self.sources.push((source.to_string(), Vec::new()));
                self.sources.len().saturating_sub(1)
            }
        };
        let targets = &mut self.sources[si].1;
        let ti = match targets
            .iter()
            .position(|(t, _)| t.eq_ignore_ascii_case(target))
        {
            Some(i) => i,
            None => {
                targets.push((target.to_string(), Vec::new()));
                targets.len().saturating_sub(1)
            }
        };
        let variants = &mut targets[ti].1;
        if !variants.iter().any(|v| v.eq_ignore_ascii_case(variant)) {
            if variant.is_empty() {
                variants.insert(0, String::new());
            } else {
                variants.push(variant.to_string());
            }
        }
    }

    fn remove(&mut self, source: &str, target: &str, variant: &str) {
        let Some(si) = self
            .sources
            .iter()
            .position(|(s, _)| s.eq_ignore_ascii_case(source))
        else {
            return;
        };
        let targets = &mut self.sources[si].1;
        let Some(ti) = targets
            .iter()
            .position(|(t, _)| t.eq_ignore_ascii_case(target))
        else {
            return;
        };
        targets[ti].1.retain(|v| !v.eq_ignore_ascii_case(variant));
        if targets[ti].1.is_empty() {
            targets.remove(ti);
            if targets.is_empty() {
                self.sources.remove(si);
            }
        }
    }
}

/// `TransliteratorRegistry`.
#[derive(Debug, Default)]
struct Registry {
    entries: HashMap<String, Entry>,
    dag: SpecDag,
}

impl Registry {
    /// `registerEntry(ID, entry, visible)`.
    fn put(&mut self, id: &str, entry: Entry, visible: bool) {
        let (s, t, v, _) = id_to_stv(id);
        let id = stv_to_id(&s, &t, &v);
        self.entries.insert(id.to_lowercase(), entry);
        if visible {
            self.dag.register(&s, &t, &v);
        } else {
            self.dag.remove(&s, &t, &v);
        }
    }

    fn find(&self, id: &str) -> Option<(String, Entry)> {
        self.entries
            .get(&id.to_lowercase())
            .map(|e| (id.to_string(), e.clone()))
    }
}

fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(build_registry()))
}

fn lock() -> std::sync::MutexGuard<'static, Registry> {
    registry().lock().unwrap_or_else(|e| e.into_inner())
}

fn unsupported_factory(id: &str) -> Result<Transliterator, IcuError> {
    Err(IcuError::with_kind(
        IcuErrorKind::UnsupportedOperation,
        format!("transliterator {id} is not ported"),
    ))
}

fn normalization(id: &str, name: &str, mode: Mode) -> Result<Transliterator, IcuError> {
    let _ = id;
    Ok(Transliterator::new(
        name,
        Kind::Normalization(Normalizer2::get_instance(
            if name.contains('K') { "nfkc" } else { "nfc" },
            mode,
        )?),
    ))
}

/// `Transliterator`'s static initializer.
fn build_registry() -> Registry {
    let mut reg = Registry::default();
    // RuleBasedTransliteratorIDs, in the bundle's order.
    if let Some(file) = pack_file("translit/root.res") {
        if let Ok(r) = ResReader::new(file) {
            if let Some(ids) = r.table_get(r.root(), "RuleBasedTransliteratorIDs") {
                for (id, row) in r.table_entries(ids) {
                    if id.contains("-t-") {
                        continue;
                    }
                    let Some((ty, res)) = r.table_entries(row).into_iter().next() else {
                        continue;
                    };
                    match ty.as_str() {
                        "file" | "internal" => {
                            let rules = r
                                .table_get(res, "resource")
                                .and_then(|x| r.string(x))
                                .unwrap_or_default();
                            let dir = match r
                                .table_get(res, "direction")
                                .and_then(|x| r.string(x))
                                .as_deref()
                            {
                                Some(d) if d.starts_with('R') => REVERSE,
                                _ => FORWARD,
                            };
                            reg.put(
                                &id,
                                Entry::Rules {
                                    rules: rules.into(),
                                    dir,
                                },
                                ty != "internal",
                            );
                        }
                        "alias" => {
                            let alias = r.string(res).unwrap_or_default();
                            reg.put(&id, Entry::Alias(alias), true);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    register_special_inverse("Null", "Null", false);
    reg.put(
        "Any-Null",
        Entry::Factory(|_| Ok(Transliterator::new("Any-Null", Kind::Null))),
        true,
    );
    reg.put(
        "Any-Remove",
        Entry::Factory(|_| Ok(Transliterator::new("Any-Remove", Kind::Remove))),
        true,
    );
    register_special_inverse("Remove", "Null", false);
    // EscapeTransliterator, UnescapeTransliterator: not ported.
    for v in ["Unicode", "Java", "C", "XML", "XML10", "Perl", "Plain"] {
        reg.put(
            &format!("Any-Hex/{v}"),
            Entry::Factory(unsupported_factory),
            true,
        );
        reg.put(
            &format!("Hex-Any/{v}"),
            Entry::Factory(unsupported_factory),
            true,
        );
    }
    reg.put("Any-Hex", Entry::Factory(unsupported_factory), true);
    reg.put("Hex-Any", Entry::Factory(unsupported_factory), true);
    register_special_inverse("Hex", "Any", true);
    reg.put(
        "Any-Lower",
        Entry::Factory(|_| {
            Ok(Transliterator::new(
                "Any-Lower",
                Kind::Case(CaseKind::Lower),
            ))
        }),
        true,
    );
    register_special_inverse("Lower", "Upper", true);
    reg.put(
        "Any-Upper",
        Entry::Factory(|_| {
            Ok(Transliterator::new(
                "Any-Upper",
                Kind::Case(CaseKind::Upper),
            ))
        }),
        true,
    );
    register_special_inverse("Upper", "Lower", true);
    reg.put(
        "Any-Title",
        Entry::Factory(|_| {
            Ok(Transliterator::new(
                "Any-Title",
                Kind::Case(CaseKind::Title),
            ))
        }),
        true,
    );
    register_special_inverse("Title", "Lower", false);
    reg.put(
        "Any-CaseFold",
        Entry::Factory(|_| {
            Ok(Transliterator::new(
                "Any-CaseFold",
                Kind::Case(CaseKind::Fold),
            ))
        }),
        true,
    );
    register_special_inverse("CaseFold", "Upper", false);
    // UnicodeNameTransliterator, NameUnicodeTransliterator: not ported.
    reg.put("Any-Name", Entry::Factory(unsupported_factory), true);
    reg.put("Name-Any", Entry::Factory(unsupported_factory), true);
    register_special_inverse("Name", "Any", true);
    reg.put(
        "Any-NFC",
        Entry::Factory(|id| normalization(id, "NFC", Mode::Compose)),
        true,
    );
    reg.put(
        "Any-NFD",
        Entry::Factory(|id| normalization(id, "NFD", Mode::Decompose)),
        true,
    );
    reg.put(
        "Any-NFKC",
        Entry::Factory(|id| normalization(id, "NFKC", Mode::Compose)),
        true,
    );
    reg.put(
        "Any-NFKD",
        Entry::Factory(|id| normalization(id, "NFKD", Mode::Decompose)),
        true,
    );
    reg.put(
        "Any-FCD",
        Entry::Factory(|id| normalization(id, "FCD", Mode::Fcd)),
        true,
    );
    reg.put(
        "Any-FCC",
        Entry::Factory(|id| normalization(id, "FCC", Mode::ComposeContiguous)),
        true,
    );
    register_special_inverse("NFC", "NFD", true);
    register_special_inverse("NFKC", "NFKD", true);
    register_special_inverse("FCC", "NFD", false);
    register_special_inverse("FCD", "FCD", false);
    reg.put(
        "Any-BreakInternal",
        Entry::Instance(Arc::new(Transliterator::new(
            "Any-BreakInternal",
            Kind::Break,
        ))),
        false,
    );
    // AnyTransliterator.register(): an Any-<target> for every script target.
    let mut seen: HashMap<String, Vec<String>> = HashMap::new();
    let mut to_register: Vec<(String, String, String, i32)> = Vec::new();
    for (source, targets) in &reg.dag.sources {
        if source.eq_ignore_ascii_case("Any") {
            continue;
        }
        for (target, variants) in targets {
            let target_script = script_name_to_code(target);
            if target_script < 0 {
                continue;
            }
            let seen_variants = seen.entry(target.clone()).or_default();
            for variant in variants {
                if seen_variants.contains(variant) {
                    continue;
                }
                seen_variants.push(variant.clone());
                let id = stv_to_id("Any", target, variant);
                to_register.push((id, target.clone(), variant.clone(), target_script));
            }
        }
    }
    for (id, target, variant, script) in to_register {
        let t = any_transliterator(&id, &target, &variant, script);
        reg.put(&id, Entry::Instance(Arc::new(t)), true);
        register_special_inverse(&target, "Null", false);
    }
    reg
}

/// `AnyTransliterator.scriptNameToCode(name)` for script names (a locale
/// name, which Java maps through likely subtags, reads as no script).
// SENTINEL: `-1` = `UScript.INVALID_CODE`, not a script name.
fn script_name_to_code(name: &str) -> i32 {
    if name.contains('_') || name.contains('-') {
        return -1;
    }
    let u = uprops::uprops();
    u.property(uprops::SCRIPT)
        .and_then(|p| u.value_by_alias(p, name))
        .unwrap_or(-1)
}

/// Whether an ID that resolves to nothing names a locale as its source or
/// target (`el-Latin`, `Any-am_FONIPA`, and `Any-Any`, whose target Java
/// reads as the locale `any`): Java resolves those through
/// `translit/<locale>.res`, likely subtags and `UScript.getCode`, which are
/// not ported, so they are refused as unsupported rather than as unknown.
fn names_locale(basic_id: &str) -> bool {
    let (source, target, _, _) = id_to_stv(basic_id);
    target == "Any"
        || [source, target].iter().any(|spec| {
            let language = spec.split('_').next().unwrap_or("");
            (2..=3).contains(&language.len())
                && language.bytes().all(|b| b.is_ascii_lowercase())
                && script_name_to_code(spec) < 0
        })
}

/// `Spec`: a source or target and its script-name fallback (`Latn` ->
/// `Latin`).
struct Spec {
    top: String,
    script_name: Option<String>,
}

impl Spec {
    fn new(top: &str) -> Spec {
        let code = script_name_to_code(top);
        let script_name = if code >= 0 {
            uprops::uprops()
                .script_name(code)
                .map(|(_, long)| long.to_string())
                .filter(|n| !n.eq_ignore_ascii_case(top))
        } else {
            None
        };
        Spec {
            top: top.to_string(),
            script_name,
        }
    }

    /// The spec and its fallbacks, in order.
    fn chain(&self) -> Vec<String> {
        let mut v = vec![self.top.clone()];
        if let Some(s) = &self.script_name {
            v.push(s.clone());
        }
        v
    }
}

/// `find(source, target, variant)`: the entry and the ID it was found
/// under.
fn find(id: &str) -> Option<(String, Entry)> {
    let (source, target, variant, _) = id_to_stv(id);
    let src = Spec::new(&source);
    let trg = Spec::new(&target);
    let reg = lock();
    if !variant.is_empty() {
        if let Some(e) = reg.find(&stv_to_id(&src.top, &trg.top, &variant)) {
            return Some(e);
        }
    }
    for t in trg.chain() {
        for s in src.chain() {
            if let Some(e) = reg.find(&stv_to_id(&s, &t, "")) {
                return Some(e);
            }
        }
    }
    None
}

/// `registry.get(id, aliasReturn)` and `instantiateEntry`: the
/// transliterator, or the alias to resolve instead.
fn get(id: &str) -> Result<Option<Result<Transliterator, String>>, IcuError> {
    let Some((found_id, entry)) = find(id) else {
        return Ok(None);
    };
    let mut entry = entry;
    loop {
        match entry {
            Entry::Data(data) => return Ok(Some(Ok(rule_based(id, data)))),
            Entry::Factory(f) => return f(id).map(|t| Some(Ok(t))),
            Entry::Alias(a) => return Ok(Some(Err(a))),
            Entry::Compound {
                id_blocks,
                data,
                filter,
            } => {
                return compound_rbt(id, &id_blocks, &data, filter.as_ref()).map(|t| Some(Ok(t)));
            }
            Entry::Instance(t) => {
                let mut t = (*t).clone();
                if let Kind::Any(_) = t.kind() {
                    // AnyTransliterator.safeClone shares the cache.
                }
                t.set_id(t.id().to_string().as_str());
                return Ok(Some(Ok(t)));
            }
            Entry::Rules { rules, dir } => {
                let parsed = parser::parse(&rules, dir)?;
                let next = entry_from_parsed(parsed);
                lock().entries.insert(stv_key(&found_id), next.clone());
                entry = next;
            }
        }
    }
}

fn stv_key(id: &str) -> String {
    let (s, t, v, _) = id_to_stv(id);
    stv_to_id(&s, &t, &v).to_lowercase()
}

/// The entry a parse leaves in the registry (`instantiateEntry`).
fn entry_from_parsed(p: Parsed) -> Entry {
    if p.id_block_vector.is_empty() && p.data_vector.is_empty() {
        Entry::Alias("Any-Null".to_string())
    } else if p.id_block_vector.is_empty() && p.data_vector.len() == 1 {
        Entry::Data(p.data_vector[0].clone())
    } else if p.id_block_vector.len() == 1 && p.data_vector.is_empty() {
        match &p.compound_filter {
            Some(f) => Entry::Alias(format!("{};{}", f.to_pattern(), p.id_block_vector[0])),
            None => Entry::Alias(p.id_block_vector[0].clone()),
        }
    } else {
        Entry::Compound {
            id_blocks: p.id_block_vector,
            data: p.data_vector,
            filter: p.compound_filter,
        }
    }
}

fn rule_based(id: &str, data: Arc<Data>) -> Transliterator {
    Transliterator::new(id, Kind::RuleBased(data))
}

/// `CompoundRBTEntry.getInstance()`.
fn compound_rbt(
    id: &str,
    id_blocks: &[String],
    data: &[Arc<Data>],
    filter: Option<&UnicodeSet>,
) -> Result<Transliterator, IcuError> {
    let mut list = Vec::new();
    let mut pass = 1u32;
    for i in 0..id_blocks.len().max(data.len()) {
        if let Some(block) = id_blocks.get(i) {
            if !block.is_empty() {
                list.push(get_instance(block, FORWARD)?);
            }
        }
        if let Some(d) = data.get(i) {
            list.push(rule_based(&format!("%Pass{pass}"), d.clone()));
            pass = pass.saturating_add(1);
        }
    }
    let mut t = Transliterator::new(id, Kind::Compound(Arc::new(list)));
    t.set_filter(filter.cloned());
    Ok(t)
}

/// `Transliterator.getBasicInstance(id, canonID)`.
fn basic_instance(id: &str, canon_id: Option<&str>) -> Result<Option<Transliterator>, IcuError> {
    let t = match get(id)? {
        None => None,
        Some(Ok(t)) => Some(t),
        Some(Err(alias)) => Some(get_instance(&alias, FORWARD)?),
    };
    Ok(t.map(|mut t| {
        if let Some(c) = canon_id {
            t.set_id(c);
        }
        t
    }))
}

/// `SingleID.getInstance()`.
pub fn single_instance(single: &SingleId) -> Result<Option<Transliterator>, IcuError> {
    let basic = if single.basic_id.is_empty() {
        "Any-Null"
    } else {
        &single.basic_id
    };
    let t = basic_instance(basic, Some(&single.canon_id))?;
    Ok(match t {
        Some(mut t) => {
            if let Some(f) = &single.filter {
                t.set_filter(Some(UnicodeSet::from_pattern(f)?));
            }
            Some(t)
        }
        None => None,
    })
}

/// `Transliterator.getInstance(id, dir)`.
pub fn get_instance(id_str: &str, dir: i32) -> Result<Transliterator, IcuError> {
    let Some((canon_id, list, global_filter)) = id::parse_compound_id(id_str, dir)? else {
        return Err(IcuError::illegal_argument(format!("Invalid ID {id_str}")));
    };
    let mut translits = Vec::new();
    for single in &list {
        if single.basic_id.is_empty() {
            continue;
        }
        match single_instance(single)? {
            Some(t) => translits.push(t),
            None if names_locale(&single.basic_id) => {
                return Err(IcuError::unsupported(format!(
                    "transliterator {} names a locale, which is not ported",
                    single.canon_id
                )))
            }
            None => {
                return Err(IcuError::illegal_argument(format!(
                    "Illegal ID {}",
                    single.canon_id
                )))
            }
        }
    }
    if translits.is_empty() {
        translits.push(basic_instance("Any-Null", None)?.ok_or_else(|| {
            IcuError::illegal_argument("Internal error; cannot instantiate Any-Null")
        })?);
    }
    let mut t = if list.len() > 1 || canon_id.contains(';') {
        Transliterator::new(&canon_id, Kind::Compound(Arc::new(translits)))
    } else {
        translits.swap_remove(0)
    };
    t.set_id(&canon_id);
    if let Some(f) = global_filter {
        t.set_filter(Some(f));
    }
    Ok(t)
}

/// `Transliterator.createFromRules(ID, rules, dir)`.
pub fn create_from_rules(id: &str, rules: &str, dir: i32) -> Result<Transliterator, IcuError> {
    let p = parser::parse(rules, dir)?;
    if p.id_block_vector.is_empty() && p.data_vector.is_empty() {
        return Ok(Transliterator::new("Any-Null", Kind::Null));
    }
    if p.id_block_vector.is_empty() && p.data_vector.len() == 1 {
        let mut t = rule_based(id, p.data_vector[0].clone());
        t.set_filter(p.compound_filter.clone());
        return Ok(t);
    }
    if p.id_block_vector.len() == 1 && p.data_vector.is_empty() {
        let mut t = match &p.compound_filter {
            Some(f) => get_instance(
                &format!("{};{}", f.to_pattern(), p.id_block_vector[0]),
                FORWARD,
            )?,
            None => get_instance(&p.id_block_vector[0], FORWARD)?,
        };
        t.set_id(id);
        return Ok(t);
    }
    let mut list = Vec::new();
    let mut pass = 1u32;
    for i in 0..p.id_block_vector.len().max(p.data_vector.len()) {
        if let Some(block) = p.id_block_vector.get(i) {
            if !block.is_empty() {
                let temp = get_instance(block, FORWARD)?;
                if !matches!(temp.kind(), Kind::Null) {
                    list.push(get_instance(block, FORWARD)?);
                }
            }
        }
        if let Some(d) = p.data_vector.get(i) {
            list.push(rule_based(&format!("%Pass{pass}"), d.clone()));
            pass = pass.saturating_add(1);
        }
    }
    let mut t = Transliterator::new(id, Kind::Compound(Arc::new(list)));
    t.set_filter(p.compound_filter);
    Ok(t)
}
