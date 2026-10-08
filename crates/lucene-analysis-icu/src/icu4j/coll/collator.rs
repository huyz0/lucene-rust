//! `com.ibm.icu.text.Collator`/`RuleBasedCollator` (sort keys) with
//! `CollationRoot`, `CollationLoader.loadTailoring` and the collator
//! service's locale fallback (`CollatorServiceShim`, `ICULocaleService`).
//!
//! `Collator::get_instance(locale)` is `Collator.getInstance(new
//! ULocale(locale))`: the locale's ID truncated until it names an installed
//! collation bundle, that bundle's tailoring for the requested (or
//! default) `collation` type through the bundle chain, then the locale's
//! attribute keywords (`colStrength`, `colAlternate`, `colBackwards`,
//! `colCaseLevel`, `colCaseFirst`, `colNormalization`, `colNumeric`,
//! `colReorder`, `kv`). The setters are `RuleBasedCollator`'s, with
//! Java's checks and errors. `raw_collation_key` is
//! `getRawCollationKey(source, key)`: the bytes and the terminating `0`
//! that `key.size` counts.
//!
//! Not ported: building a collator from rules (`new
//! RuleBasedCollator(rules)`, which needs ICU's `CollationBuilder`) --
//! `from_rules` returns `UnsupportedOperation`; `compare` and
//! `CollationKey` comparison (sort keys compare as bytes); `variableTop` by
//! character or primary.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::icu4j::coll::data::{
    self, CollationTailoring, REORDER_CODE_CURRENCY, REORDER_CODE_DEFAULT, REORDER_CODE_FIRST,
    REORDER_CODE_NONE,
};
use crate::icu4j::coll::iter::CollationIterator;
use crate::icu4j::coll::keys;
use crate::icu4j::coll::locale::Locale;
use crate::icu4j::coll::res::{self, find_with_fallback, instantiate, pack_file};
use crate::icu4j::coll::settings::{self as s, CollationSettings};
use crate::{IcuError, IcuErrorKind};

/// `Collator.PRIMARY` and the other strengths.
pub use crate::icu4j::coll::settings::{IDENTICAL, PRIMARY, QUATERNARY, SECONDARY, TERTIARY};

/// `Collator.NO_DECOMPOSITION` / `CANONICAL_DECOMPOSITION`.
pub const NO_DECOMPOSITION: i32 = 16;
pub const CANONICAL_DECOMPOSITION: i32 = 17;

fn illegal(msg: impl Into<String>) -> IcuError {
    IcuError::with_kind(IcuErrorKind::IllegalArgument, msg)
}

/// `CollationRoot.getRoot()`.
fn root() -> Result<Arc<CollationTailoring>, IcuError> {
    static ROOT: OnceLock<Result<Arc<CollationTailoring>, IcuError>> = OnceLock::new();
    ROOT.get_or_init(|| {
        let bytes = pack_file("ucadata.icu").ok_or_else(|| {
            IcuError::with_kind(
                IcuErrorKind::MissingResource,
                "IOException while reading CLDR root data",
            )
        })?;
        data::read(None, bytes).map(Arc::new)
    })
    .clone()
}

/// Tailorings already read, by bundle and type.
type TailoringCache = Mutex<HashMap<(String, String), Arc<CollationTailoring>>>;

fn tailoring_cache() -> &'static TailoringCache {
    static CACHE: OnceLock<TailoringCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `CollationLoader.loadTailoring(locale, outValidLocale)`: the tailoring
/// and the valid locale's name.
fn load_tailoring(locale: &Locale) -> Result<(Arc<CollationTailoring>, String), IcuError> {
    let root = root()?;
    let locale_name = locale.name();
    if locale_name.is_empty() || locale_name == "root" {
        return Ok((root, String::new()));
    }
    let chain = instantiate(&locale.base_name)?;
    let Some(bundle) = chain.first() else {
        return Ok((root, String::new()));
    };
    let mut valid_locale = bundle.locale_id.clone();
    if valid_locale == "root" {
        valid_locale.clear();
    }
    if find_with_fallback(&chain, &["collations"]).is_none() {
        return Ok((root, valid_locale));
    }
    let find_string =
        |path: &[&str]| find_with_fallback(&chain, path).and_then(|(b, r)| b.reader.string(r));
    let mut default_type = "standard".to_string();
    if let Some(d) = find_string(&["collations", "default"]) {
        default_type = d;
    }
    let mut ty = match locale.keyword("collation") {
        None | Some("default") => default_type.clone(),
        Some(t) => t.to_ascii_lowercase(),
    };
    let mut found = find_with_fallback(&chain, &["collations", &ty]);
    if found.is_none() && ty.len() > 6 && ty.starts_with("search") {
        ty = "search".into();
        found = find_with_fallback(&chain, &["collations", &ty]);
    }
    if found.is_none() && ty != default_type {
        ty.clone_from(&default_type);
        found = find_with_fallback(&chain, &["collations", &ty]);
    }
    if found.is_none() && ty != "standard" {
        ty = "standard".into();
        found = find_with_fallback(&chain, &["collations", &ty]);
    }
    let Some((data_bundle, data_res)) = found else {
        return Ok((root, valid_locale));
    };
    let mut actual_locale = data_bundle.locale_id.clone();
    if actual_locale.is_empty() || actual_locale == "root" {
        actual_locale.clear();
        if ty == "standard" {
            return Ok((root, valid_locale));
        }
    }
    let key = (actual_locale.clone(), ty.clone());
    let cached = tailoring_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&key)
        .cloned();
    let mut t = match cached {
        Some(t) => (*t).clone(),
        None => {
            let reader = &data_bundle.reader;
            let binary = reader
                .table_get(data_res, "%%CollationBin")
                .and_then(|r| reader.binary(r))
                .ok_or_else(|| {
                    IcuError::with_kind(
                        IcuErrorKind::MissingResource,
                        format!(
                            "Can't find resource for bundle {actual_locale}, key %%CollationBin"
                        ),
                    )
                })?;
            let mut t = data::read(Some(&root), binary).map_err(|e| {
                IcuError::new(format!(
                    "Failed to load collation tailoring data for locale:{actual_locale} type:{ty}: {}",
                    e.message()
                ))
            })?;
            t.actual_locale.clone_from(&actual_locale);
            tailoring_cache()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(key, Arc::new(t.clone()));
            t
        }
    };
    if ty != default_type {
        valid_locale = set_keyword(&valid_locale, "collation", &ty);
    }
    if actual_locale != valid_locale.split('@').next().unwrap_or("") {
        if let Ok(actual_chain) = instantiate(&actual_locale) {
            let ff = find_with_fallback(&actual_chain, &["collations", "default"])
                .and_then(|(b, r)| b.reader.string(r));
            if let Some(d) = ff {
                default_type = d;
            }
        }
    }
    if ty != default_type {
        t.actual_locale = set_keyword(&t.actual_locale, "collation", &ty);
    }
    Ok((Arc::new(t), valid_locale))
}

/// `ULocale.setKeywordValue(name, value)` on a name without keywords.
fn set_keyword(name: &str, key: &str, value: &str) -> String {
    format!("{name}@{key}={value}")
}

/// `RuleBasedCollator`.
#[derive(Debug, Clone)]
pub struct Collator {
    tailoring: Arc<CollationTailoring>,
    settings: CollationSettings,
    valid_locale: String,
}

impl Collator {
    /// `Collator.getInstance(ULocale.ROOT)`.
    pub fn root() -> Result<Collator, IcuError> {
        Self::get_instance("")
    }

    /// `Collator.getInstance(new ULocale(locale))`.
    pub fn get_instance(locale: &str) -> Result<Collator, IcuError> {
        let locale = Locale::new(locale)?;
        // ICULocaleService: truncate the ID until a bundle is installed
        // (`LocaleKey.fallback`: "zh__PINYIN" -> "zh"), then root.
        let mut current = locale.base_name.clone();
        let found = loop {
            if res::is_installed(&current) {
                break current;
            }
            match current.rfind('_') {
                Some(mut x) => {
                    while x > 0 && current.as_bytes().get(x.saturating_sub(1)) == Some(&b'_') {
                        x = x.saturating_sub(1);
                    }
                    current.truncate(x);
                }
                None if !current.is_empty() => current.clear(),
                None => break current,
            }
        };
        let service_locale = Locale {
            base_name: found,
            keywords: locale.keywords.clone(),
        };
        let (tailoring, valid_locale) = load_tailoring(&service_locale)?;
        let mut coll = Collator {
            settings: tailoring.settings.clone(),
            tailoring,
            valid_locale,
        };
        if !locale.keywords.is_empty() {
            coll.set_attributes_from_keywords(&locale)?;
        }
        Ok(coll)
    }

    /// `new RuleBasedCollator(rules)`: needs ICU's collation rule builder,
    /// which is not ported.
    pub fn from_rules(_rules: &str) -> Result<Collator, IcuError> {
        Err(IcuError::with_kind(
            IcuErrorKind::UnsupportedOperation,
            "RuleBasedCollator(String rules): building a collator from rules is not supported",
        ))
    }

    /// `getLocale(ULocale.VALID_LOCALE)`.
    pub fn valid_locale(&self) -> &str {
        &self.valid_locale
    }

    /// `getLocale(ULocale.ACTUAL_LOCALE)`.
    pub fn actual_locale(&self) -> &str {
        &self.tailoring.actual_locale
    }

    /// `setAttributesFromKeywords(loc, coll, rbc)`.
    fn set_attributes_from_keywords(&mut self, loc: &Locale) -> Result<(), IcuError> {
        if loc.keyword("colHiraganaQuaternary").is_some() {
            return Err(IcuError::with_kind(
                IcuErrorKind::UnsupportedOperation,
                "locale keyword kh/colHiraganaQuaternary",
            ));
        }
        if loc.keyword("variableTop").is_some() {
            return Err(IcuError::with_kind(
                IcuErrorKind::UnsupportedOperation,
                "locale keyword vt/variableTop",
            ));
        }
        if let Some(v) = loc.keyword("colStrength") {
            let strength = int_value(
                "colStrength",
                v,
                &[
                    "primary",
                    "secondary",
                    "tertiary",
                    "quaternary",
                    "identical",
                ],
            )?;
            self.set_strength(if strength <= QUATERNARY {
                strength
            } else {
                IDENTICAL
            })?;
        }
        if let Some(v) = loc.keyword("colBackwards") {
            self.set_french_collation(yes_or_no("colBackwards", v)?);
        }
        if let Some(v) = loc.keyword("colCaseLevel") {
            self.set_case_level(yes_or_no("colCaseLevel", v)?);
        }
        if let Some(v) = loc.keyword("colCaseFirst") {
            match int_value("colCaseFirst", v, &["no", "lower", "upper"])? {
                0 => {
                    self.set_lower_case_first(false);
                    self.set_upper_case_first(false);
                }
                1 => self.set_lower_case_first(true),
                _ => self.set_upper_case_first(true),
            }
        }
        if let Some(v) = loc.keyword("colAlternate") {
            self.set_alternate_handling_shifted(
                int_value("colAlternate", v, &["non-ignorable", "shifted"])? != 0,
            );
        }
        if let Some(v) = loc.keyword("colNormalization") {
            self.set_decomposition(if yes_or_no("colNormalization", v)? {
                CANONICAL_DECOMPOSITION
            } else {
                NO_DECOMPOSITION
            })?;
        }
        if let Some(v) = loc.keyword("colNumeric") {
            self.set_numeric_collation(yes_or_no("colNumeric", v)?);
        }
        if let Some(v) = loc.keyword("colReorder") {
            let mut codes = Vec::new();
            // UScript.CODE_LIMIT + ReorderCodes.LIMIT - ReorderCodes.FIRST
            let limit = 213usize;
            for name in v.split('-') {
                if codes.len() == limit {
                    return Err(illegal(format!(
                        "too many script codes for colReorder locale keyword: {v}"
                    )));
                }
                let code = if name.chars().count() == 4 {
                    // UCharacter.getPropertyValueEnum(UProperty.SCRIPT, name)
                    let u = crate::icu4j::uprops::uprops();
                    u.property(0x100a)
                        .and_then(|p| u.value_by_alias(p, name))
                        .ok_or_else(|| {
                            IcuError::with_kind(
                                IcuErrorKind::IllegalIcuArgument,
                                format!("Invalid name: {name}"),
                            )
                        })?
                } else {
                    reorder_code("colReorder", name)?
                };
                codes.push(code);
            }
            self.set_reorder_codes(&codes)?;
        }
        if let Some(v) = loc.keyword("kv") {
            self.set_max_variable(reorder_code("kv", v)?)?;
        }
        Ok(())
    }

    fn default_settings(&self) -> &CollationSettings {
        &self.tailoring.settings
    }

    /// `setStrength(newStrength)`.
    pub fn set_strength(&mut self, strength: i32) -> Result<(), IcuError> {
        if strength == self.settings.strength() {
            return Ok(());
        }
        self.settings.set_strength(strength)
    }

    /// `getStrength()`.
    pub fn strength(&self) -> i32 {
        self.settings.strength()
    }

    /// `setDecomposition(decomposition)`.
    pub fn set_decomposition(&mut self, decomposition: i32) -> Result<(), IcuError> {
        let flag = match decomposition {
            NO_DECOMPOSITION => false,
            CANONICAL_DECOMPOSITION => true,
            _ => return Err(illegal("Wrong decomposition mode.")),
        };
        self.settings.set_flag(s::CHECK_FCD, flag);
        Ok(())
    }

    /// `setFrenchCollation(flag)`.
    pub fn set_french_collation(&mut self, flag: bool) {
        self.settings.set_flag(s::BACKWARD_SECONDARY, flag);
    }

    /// `setCaseLevel(flag)`.
    pub fn set_case_level(&mut self, flag: bool) {
        self.settings.set_flag(s::CASE_LEVEL, flag);
    }

    /// `setUpperCaseFirst(upperfirst)`.
    pub fn set_upper_case_first(&mut self, upper_first: bool) {
        if upper_first == (self.settings.case_first() == s::CASE_FIRST_AND_UPPER_MASK) {
            return;
        }
        self.settings.set_case_first(if upper_first {
            s::CASE_FIRST_AND_UPPER_MASK
        } else {
            0
        });
    }

    /// `setLowerCaseFirst(lowerfirst)`.
    pub fn set_lower_case_first(&mut self, lower_first: bool) {
        if lower_first == (self.settings.case_first() == s::CASE_FIRST) {
            return;
        }
        self.settings
            .set_case_first(if lower_first { s::CASE_FIRST } else { 0 });
    }

    /// `setAlternateHandlingShifted(shifted)`.
    pub fn set_alternate_handling_shifted(&mut self, shifted: bool) {
        self.settings.set_alternate_handling_shifted(shifted);
    }

    /// `setNumericCollation(flag)`.
    pub fn set_numeric_collation(&mut self, flag: bool) {
        self.settings.set_flag(s::NUMERIC, flag);
    }

    /// `setMaxVariable(group)`: `ReorderCodes.SPACE` .. `CURRENCY`, or
    /// `DEFAULT`.
    pub fn set_max_variable(&mut self, group: i32) -> Result<(), IcuError> {
        let value = if group == REORDER_CODE_DEFAULT {
            -1
        } else if (REORDER_CODE_FIRST..=REORDER_CODE_CURRENCY).contains(&group) {
            group.saturating_sub(REORDER_CODE_FIRST)
        } else {
            return Err(illegal(format!("illegal max variable group {group}")));
        };
        if value == self.settings.max_variable() {
            return Ok(());
        }
        let default_options = self.default_settings().options;
        if self.settings == *self.default_settings() && value < 0 {
            return Ok(());
        }
        let group = if group == REORDER_CODE_DEFAULT {
            REORDER_CODE_FIRST | ((default_options & s::MAX_VARIABLE_MASK) >> s::MAX_VARIABLE_SHIFT)
        } else {
            group
        };
        let var_top = self.tailoring.data.get_last_primary_for_group(group);
        self.settings.set_max_variable(value, default_options)?;
        self.settings.variable_top = var_top;
        Ok(())
    }

    /// `setReorderCodes(order...)`.
    pub fn set_reorder_codes(&mut self, order: &[i32]) -> Result<(), IcuError> {
        let order: &[i32] = if order.len() == 1 && order[0] == REORDER_CODE_NONE {
            &[]
        } else {
            order
        };
        let same = if order.is_empty() {
            self.settings.reorder_codes.is_empty()
        } else {
            order == self.settings.reorder_codes.as_slice()
        };
        if same {
            return Ok(());
        }
        if order.len() == 1 && order[0] == REORDER_CODE_DEFAULT {
            if self.settings != *self.default_settings() {
                let d = self.default_settings().clone();
                self.settings.copy_reordering_from(&d);
            }
            return Ok(());
        }
        if order.is_empty() {
            self.settings.reset_reordering();
            Ok(())
        } else {
            let data = self.tailoring.data.clone();
            self.settings.set_reordering(&data, order)
        }
    }

    /// `getRawCollationKey(source, key)`: the sort key bytes, the
    /// terminating `0` included.
    pub fn raw_collation_key(&self, source: &str) -> Vec<u8> {
        let units: Vec<u16> = source.encode_utf16().collect();
        self.raw_collation_key_utf16(&units)
    }

    /// `getRawCollationKey` over UTF-16 text (unpaired surrogates as Java
    /// keeps them).
    pub fn raw_collation_key_utf16(&self, units: &[u16]) -> Vec<u8> {
        let mut key = Vec::with_capacity(units.len().saturating_mul(2).saturating_add(10));
        self.write_sort_key(units, &mut key);
        key
    }

    /// `writeSortKey(s, sink, buffer)`.
    fn write_sort_key(&self, units: &[u16], key: &mut Vec<u8>) {
        let data = &*self.tailoring.data;
        let numeric = self.settings.is_numeric();
        let fcd = !self.settings.dont_check_fcd();
        let mut iter = CollationIterator::new(data, numeric, units, fcd);
        keys::write_sort_key_up_to_quaternary(
            &mut iter,
            &|b| data.is_compressible_lead_byte(b),
            &self.settings,
            key,
        );
        if self.settings.strength() == IDENTICAL {
            keys::write_identical_level(&data.nfc_impl, units, key);
        }
        key.push(0);
    }
}

/// `getYesOrNo(keyword, s)`.
fn yes_or_no(keyword: &str, s: &str) -> Result<bool, IcuError> {
    if s.eq_ignore_ascii_case("yes") {
        Ok(true)
    } else if s.eq_ignore_ascii_case("no") {
        Ok(false)
    } else {
        Err(illegal(format!(
            "illegal locale keyword=value: {keyword}={s}"
        )))
    }
}

/// `getIntValue(keyword, s, values...)`.
fn int_value(keyword: &str, s: &str, values: &[&str]) -> Result<i32, IcuError> {
    values
        .iter()
        .position(|v| s.eq_ignore_ascii_case(v))
        .map(|i| i as i32)
        .ok_or_else(|| illegal(format!("illegal locale keyword=value: {keyword}={s}")))
}

/// `getReorderCode(keyword, s)`.
fn reorder_code(keyword: &str, s: &str) -> Result<i32, IcuError> {
    let i = int_value(
        keyword,
        s,
        &["space", "punct", "symbol", "currency", "digit"],
    )?;
    Ok(REORDER_CODE_FIRST | i)
}
