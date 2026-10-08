//! ICU's collation data files and the resource bundles that hold them:
//! the vendored pack (`resources/coll.pack.z`, written by
//! `tools/GenIcuCollPack.java` from the ICU4J 77.1 jar), a reader of the
//! `.res` resource bundle format (`com.ibm.icu.impl.ICUResourceBundleReader`,
//! data format `ResB`, format versions 1.1-3), and the locale fallback
//! `ICUResourceBundle.instantiateBundle` performs for `OpenType.LOCALE_ROOT`
//! (bundle aliases, `%%Parent`, truncation, the default-script rules of
//! `getParentLocaleID`, then root).
//!
//! Only what collation reads is ported: tables (all three layouts),
//! strings (both forms), binaries and the bundle-level `%%ALIAS`. A pool
//! bundle, item aliases and arrays are not read (ICU's collation bundles
//! use none of them); a resource of another type reads as missing.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::icu4j::binary::read_header;
use crate::icu4j::coll::locale;
use crate::IcuError;

const PACK: &[u8] = include_bytes!("../../resources/coll.pack.z");

/// The pack: name -> bytes.
struct Pack {
    raw: Vec<u8>,
    entries: HashMap<String, (usize, usize)>,
}

fn pack() -> &'static Pack {
    static P: OnceLock<Pack> = OnceLock::new();
    P.get_or_init(|| {
        let raw = miniz_oxide::inflate::decompress_to_vec_zlib(PACK)
            .expect("vendored collation pack inflates");
        let entries = parse_pack(&raw).expect("vendored collation pack parses");
        Pack { raw, entries }
    })
}

fn parse_pack(raw: &[u8]) -> Option<HashMap<String, (usize, usize)>> {
    if raw.get(..4)? != b"ICP1" {
        return None;
    }
    let be32 = |p: usize| -> Option<usize> {
        let b = raw.get(p..p.checked_add(4)?)?;
        Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    let count = be32(4)?;
    let mut pos = 8usize;
    let mut entries = HashMap::with_capacity(count);
    for _ in 0..count {
        let b = raw.get(pos..pos.checked_add(2)?)?;
        let name_len = usize::from(u16::from_be_bytes([b[0], b[1]]));
        pos = pos.checked_add(2)?;
        let name = std::str::from_utf8(raw.get(pos..pos.checked_add(name_len)?)?).ok()?;
        pos = pos.checked_add(name_len)?;
        let len = be32(pos)?;
        pos = pos.checked_add(4)?;
        raw.get(pos..pos.checked_add(len)?)?;
        entries.insert(name.to_string(), (pos, len));
        pos = pos.checked_add(len)?;
    }
    Some(entries)
}

/// A file of the pack (`coll/<name>`).
pub fn pack_file(name: &str) -> Option<&'static [u8]> {
    let p = pack();
    let &(start, len) = p.entries.get(name)?;
    p.raw.get(start..start.saturating_add(len))
}

/// A generated text file of the pack (empty if absent).
pub fn pack_text(name: &str) -> &'static str {
    pack_file(name)
        .and_then(|b| std::str::from_utf8(b).ok())
        .unwrap_or("")
}

/// `UResourceBundle` types.
const BINARY: u32 = 1;
const TABLE: u32 = 2;
const TABLE32: u32 = 4;
const TABLE16: u32 = 5;
const STRING_V2: u32 = 6;

const DATA_FORMAT: u32 = 0x5265_7342; // "ResB"

/// `ICUResourceBundleReader` over one `.res` file.
#[derive(Debug, Clone)]
pub struct ResReader {
    bytes: &'static [u8],
    big_endian: bool,
    root_res: u32,
    local_key_limit: usize,
    b16_start: usize,
    b16_len: usize,
    pool_string_index_limit: u32,
    pool_string_index16_limit: u32,
    no_fallback: bool,
}

// ARITH: (the whole impl) offsets and counts read off the bundle, checked
// against its length by every read (`get`), small header values shifted.
#[allow(clippy::arithmetic_side_effects)]
impl ResReader {
    /// `init(inBytes)`.
    pub fn new(file: &'static [u8]) -> Result<ResReader, IcuError> {
        let (r, version) = read_header(file, DATA_FORMAT, |v| {
            (v[0] == 1 && v[1] >= 1) || (2..=3).contains(&v[0])
        })?;
        let bytes = r.rest();
        let mut reader = ResReader {
            bytes,
            big_endian: r.is_big_endian(),
            root_res: 0,
            local_key_limit: 0,
            b16_start: 0,
            b16_len: 0,
            pool_string_index_limit: 0,
            pool_string_index16_limit: 0,
            no_fallback: false,
        };
        reader.root_res = reader
            .u32_at(0)
            .ok_or_else(|| IcuError::new("not enough bytes"))?;
        let indexes0 = reader
            .index(0)
            .ok_or_else(|| IcuError::new("not enough bytes"))?;
        let index_length = indexes0 & 0xff;
        if index_length <= 4 {
            return Err(IcuError::new("not enough indexes"));
        }
        let bundle_top = reader
            .index(3)
            .ok_or_else(|| IcuError::new("not enough bytes"))?;
        let data_length = bytes.len() as u64;
        if data_length < u64::from(1 + index_length) << 2
            || data_length < u64::from(bundle_top) << 2
        {
            return Err(IcuError::new("not enough bytes"));
        }
        if version[0] >= 3 {
            reader.pool_string_index_limit = indexes0 >> 8;
        }
        if index_length > 5 {
            let att = reader.index(5).unwrap_or(0);
            reader.no_fallback = att & 1 != 0;
            if att & 6 != 0 {
                return Err(IcuError::new("pool bundles are not supported"));
            }
            reader.pool_string_index_limit |= (att & 0xf000) << 12;
            reader.pool_string_index16_limit = att >> 16;
        }
        let keys_top = reader.index(1).unwrap_or(0);
        if keys_top > 1 + index_length {
            reader.local_key_limit = (keys_top as usize).saturating_mul(4);
        }
        if index_length > 6 {
            let top16 = reader.index(6).unwrap_or(0);
            if top16 > keys_top {
                reader.b16_start = (keys_top as usize).saturating_mul(4);
                reader.b16_len = ((top16 - keys_top) as usize).saturating_mul(2);
            }
        }
        Ok(reader)
    }

    fn u32_at(&self, off: usize) -> Option<u32> {
        let b = self.bytes.get(off..off.checked_add(4)?)?;
        let a = [b[0], b[1], b[2], b[3]];
        Some(if self.big_endian {
            u32::from_be_bytes(a)
        } else {
            u32::from_le_bytes(a)
        })
    }

    fn u16_at(&self, off: usize) -> Option<u16> {
        let b = self.bytes.get(off..off.checked_add(2)?)?;
        let a = [b[0], b[1]];
        Some(if self.big_endian {
            u16::from_be_bytes(a)
        } else {
            u16::from_le_bytes(a)
        })
    }

    fn index(&self, i: usize) -> Option<u32> {
        self.u32_at(i.checked_add(1)?.checked_mul(4)?)
    }

    fn b16(&self, i: usize) -> Option<u16> {
        if i >= self.b16_len {
            return None;
        }
        self.u16_at(self.b16_start.checked_add(i.checked_mul(2)?)?)
    }

    /// The root resource.
    pub fn root(&self) -> u32 {
        self.root_res
    }

    /// `getNoFallback()`.
    pub fn no_fallback(&self) -> bool {
        self.no_fallback
    }

    /// `ICUBinary.compareKeys(key, keyBytes, offset)`.
    fn compare_key(&self, key: &[u8], offset: usize) -> Option<std::cmp::Ordering> {
        if offset >= self.local_key_limit {
            return None; // a pool bundle's key
        }
        let mut i = 0usize;
        loop {
            let c2 = *self.bytes.get(offset.checked_add(i)?)?;
            if c2 == 0 {
                return Some(if i == key.len() {
                    std::cmp::Ordering::Equal
                } else {
                    std::cmp::Ordering::Greater
                });
            }
            let Some(&c1) = key.get(i) else {
                return Some(std::cmp::Ordering::Less);
            };
            if c1 != c2 {
                return Some(c1.cmp(&c2));
            }
            i = i.checked_add(1)?;
        }
    }

    /// `getTable(res)` then `getResource(reader, key)`: the item of a table.
    pub fn table_get(&self, res: u32, key: &str) -> Option<u32> {
        let ty = res >> 28;
        let offset = (res & 0x0fff_ffff) as usize;
        if offset == 0 && matches!(ty, TABLE | TABLE16 | TABLE32) {
            return None;
        }
        let key = key.as_bytes();
        // (size, key offset at i, item at i)
        match ty {
            TABLE => {
                let base = offset.checked_mul(4)?;
                let size = usize::from(self.u16_at(base)?);
                let key_at = |i: usize| self.u16_at(base + 2 + 2 * i).map(usize::from);
                let items = base.checked_add(2usize.checked_mul((size + 2) & !1)?)?;
                let i = self.search(size, key, key_at)?;
                self.u32_at(items.checked_add(4usize.checked_mul(i)?)?)
            }
            TABLE16 => {
                let size = usize::from(self.b16(offset)?);
                let key_at = |i: usize| self.b16(offset + 1 + i).map(usize::from);
                let i = self.search(size, key, key_at)?;
                let res16 = u32::from(self.b16(offset.checked_add(1 + size + i)?)?);
                let res16 = if res16 < self.pool_string_index16_limit {
                    res16
                } else {
                    res16 - self.pool_string_index16_limit + self.pool_string_index_limit
                };
                Some((STRING_V2 << 28) | res16)
            }
            TABLE32 => {
                let base = offset.checked_mul(4)?;
                let size = self.u32_at(base)? as usize;
                let key_at = |i: usize| {
                    let k = self.u32_at(base + 4 + 4 * i)? as i32;
                    usize::try_from(k).ok()
                };
                let i = self.search(size, key, key_at)?;
                self.u32_at(base.checked_add(4usize.checked_mul(1 + size + i)?)?)
            }
            _ => None,
        }
    }

    /// `findTableItem`: binary search of the sorted keys.
    fn search(
        &self,
        size: usize,
        key: &[u8],
        key_at: impl Fn(usize) -> Option<usize>,
    ) -> Option<usize> {
        let (mut start, mut limit) = (0usize, size);
        while start < limit {
            let mid = (start + limit) / 2;
            match self.compare_key(key, key_at(mid)?)? {
                std::cmp::Ordering::Less => limit = mid,
                std::cmp::Ordering::Greater => start = mid + 1,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// `getString(res)`.
    pub fn string(&self, res: u32) -> Option<String> {
        let ty = res >> 28;
        let offset = res & 0x0fff_ffff;
        if res != offset && ty != STRING_V2 {
            return None;
        }
        if offset == 0 {
            return Some(String::new());
        }
        let units: Vec<u16> = if res != offset {
            if offset < self.pool_string_index_limit {
                return None; // pool string
            }
            let mut o = (offset - self.pool_string_index_limit) as usize;
            let first = self.b16(o)?;
            if first & 0xfc00 != 0xdc00 {
                let mut v = vec![first];
                loop {
                    o = o.checked_add(1)?;
                    let c = self.b16(o)?;
                    if c == 0 {
                        break;
                    }
                    v.push(c);
                }
                v
            } else {
                let (length, start) = if first < 0xdfef {
                    (usize::from(first & 0x3ff), o + 1)
                } else if first < 0xdfff {
                    (
                        (usize::from(first - 0xdfef) << 16) | usize::from(self.b16(o + 1)?),
                        o + 2,
                    )
                } else {
                    (
                        (usize::from(self.b16(o + 1)?) << 16) | usize::from(self.b16(o + 2)?),
                        o + 3,
                    )
                };
                (start..start.checked_add(length)?)
                    .map(|i| self.b16(i))
                    .collect::<Option<_>>()?
            }
        } else {
            let base = (offset as usize).checked_mul(4)?;
            let length = self.u32_at(base)? as usize;
            (0..length)
                .map(|i| self.u16_at(base + 4 + 2 * i))
                .collect::<Option<_>>()?
        };
        Some(String::from_utf16_lossy(&units))
    }

    /// `getBinary(res)`.
    pub fn binary(&self, res: u32) -> Option<&'static [u8]> {
        if res >> 28 != BINARY {
            return None;
        }
        let offset = (res & 0x0fff_ffff) as usize;
        if offset == 0 {
            return Some(&[]);
        }
        let base = offset.checked_mul(4)?;
        let length = self.u32_at(base)? as usize;
        let start = base.checked_add(4)?;
        self.bytes.get(start..start.checked_add(length)?)
    }

    /// Whether `res` is a table.
    pub fn is_table(res: u32) -> bool {
        matches!(res >> 28, TABLE | TABLE16 | TABLE32)
    }
}

/// One bundle of the chain `instantiateBundle` builds.
#[derive(Debug, Clone)]
pub struct Bundle {
    /// `getLocaleID()` (`root` for the root bundle).
    pub locale_id: String,
    pub reader: ResReader,
}

/// `createBundle(baseName, localeID, root)`: the bundle of a file, its
/// `%%ALIAS` followed.
// ARITH: depth is bounded (8).
#[allow(clippy::arithmetic_side_effects)]
fn create_bundle(locale_id: &str, depth: u32) -> Result<Option<Bundle>, IcuError> {
    let Some(file) = pack_file(&format!("{locale_id}.res")) else {
        return Ok(None);
    };
    let reader = ResReader::new(file)?;
    if let Some(alias) = reader
        .table_get(reader.root(), "%%ALIAS")
        .and_then(|r| reader.string(r))
    {
        if depth > 8 {
            return Err(IcuError::new("collation bundle aliases loop"));
        }
        // UResourceBundle.getBundleInstance(baseName, alias): the target
        // bundle (which the collation data always has).
        let name = if alias.is_empty() {
            "root"
        } else {
            alias.as_str()
        };
        return create_bundle(name, depth + 1);
    }
    Ok(Some(Bundle {
        locale_id: locale_id.to_string(),
        reader,
    }))
}

/// `ICUResourceBundle.getDefaultScript(language, region)`.
fn default_script(language: &str, region: &str) -> String {
    let table = pack_text("default_scripts.txt");
    let find = |id: &str| {
        table.lines().find_map(|l| {
            let (k, v) = l.split_once('=')?;
            (k == id).then(|| v.to_string())
        })
    };
    find(&format!("{language}_{region}"))
        .or_else(|| find(language))
        .unwrap_or_else(|| "Latn".to_string())
}

/// `ICUResourceBundle.getParentLocaleID(name, origName, LOCALE_ROOT)`.
fn parent_locale_id(name: &str, orig_name: &str) -> Option<String> {
    let (language, script, region, variant) = locale::parts(name);
    if name.ends_with('_') || !variant.is_empty() {
        return name.rfind('_').map(|i| name[..i].to_string());
    }
    if !script.is_empty() && !region.is_empty() {
        if default_script(&language, &region) == script {
            Some(format!("{language}_{region}"))
        } else {
            Some(format!("{language}_{script}"))
        }
    } else if !region.is_empty() {
        let orig_script = locale::parts(orig_name).1;
        if !orig_script.is_empty() {
            Some(format!("{language}_{orig_script}"))
        } else {
            Some(format!("{language}_{}", default_script(&language, &region)))
        }
    } else if !script.is_empty() {
        Some(language)
    } else {
        None
    }
}

/// `instantiateBundle(baseName, localeID, origLocaleID, ..., LOCALE_ROOT)`:
/// the bundle chain for a base name, nearest first.
pub fn instantiate(locale_id: &str) -> Result<Vec<Bundle>, IcuError> {
    instantiate_from(locale_id, None, 0)
}

// ARITH: depth is bounded (32).
#[allow(clippy::arithmetic_side_effects)]
fn instantiate_from(
    locale_id: &str,
    orig: Option<&str>,
    depth: u32,
) -> Result<Vec<Bundle>, IcuError> {
    if depth > 32 {
        return Err(IcuError::new("collation bundle fallback loops"));
    }
    let locale_name = if locale_id.is_empty() {
        "root"
    } else {
        locale_id
    };
    match create_bundle(locale_name, 0)? {
        None => {
            let orig_name = orig.unwrap_or(locale_name);
            match parent_locale_id(locale_name, orig_name) {
                Some(fallback) => instantiate_from(&fallback, Some(orig_name), depth + 1),
                None => Ok(create_bundle("root", 0)?.into_iter().collect()),
            }
        }
        Some(b) => {
            if b.reader.no_fallback() {
                return Ok(vec![b]);
            }
            let name = b.locale_id.clone();
            let parent_name = b
                .reader
                .table_get(b.reader.root(), "%%Parent")
                .and_then(|r| b.reader.string(r));
            let parent = if let Some(p) = parent_name {
                instantiate_from(&p, None, depth + 1)?
            } else if let Some(i) = name.rfind('_') {
                instantiate_from(&name[..i], None, depth + 1)?
            } else if name != "root" {
                instantiate_from("root", None, depth + 1)?
            } else {
                Vec::new()
            };
            let mut chain = vec![b];
            chain.extend(parent);
            Ok(chain)
        }
    }
}

/// `findResourceWithFallback(path)` from the head of `chain`: the first
/// bundle whose tables hold the whole path, and the resource there.
pub fn find_with_fallback<'c>(chain: &'c [Bundle], path: &[&str]) -> Option<(&'c Bundle, u32)> {
    chain.iter().find_map(|b| {
        let mut res = b.reader.root();
        for key in path {
            if !ResReader::is_table(res) {
                return None;
            }
            res = b.reader.table_get(res, key)?;
        }
        Some((b, res))
    })
}

/// The installed collation locales (`getFullLocaleNameSet`): every bundle
/// file but `res_index`, root as `""`.
pub fn is_installed(locale_id: &str) -> bool {
    let id = if locale_id.is_empty() {
        "root"
    } else {
        locale_id
    };
    if id == "res_index" {
        return false;
    }
    if (id.len() == 1 || id.len() > 3) && !id.contains('_') && id != "root" {
        return false;
    }
    pack_file(&format!("{id}.res")).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_and_bundles() {
        assert!(pack_file("ucadata.icu").is_some());
        assert!(pack_file("nope").is_none());
        assert!(!pack_text("default_scripts.txt").is_empty());
        assert_eq!(pack_text("nope"), "");
        assert!(is_installed("de"));
        assert!(is_installed(""));
        assert!(!is_installed("res_index"));
        assert!(!is_installed("de_CH"));
        let chain = instantiate("de_CH").unwrap();
        let ids: Vec<&str> = chain.iter().map(|b| b.locale_id.as_str()).collect();
        assert_eq!(ids, ["de", "root"]);
        let chain = instantiate("zh_TW").unwrap();
        assert_eq!(chain[0].locale_id, "zh_Hant_TW");
        let (b, res) = find_with_fallback(&chain, &["collations", "default"]).unwrap();
        assert!(b.reader.string(res).is_some());
        assert!(find_with_fallback(&chain, &["collations", "nope"]).is_none());
        let chain = instantiate("xx").unwrap();
        assert_eq!(chain[0].locale_id, "root");
        assert_eq!(
            parent_locale_id("de__PHONEBOOK", "de__PHONEBOOK").as_deref(),
            Some("de_")
        );
        assert_eq!(
            parent_locale_id("sr_Latn_RS", "sr_Latn_RS").as_deref(),
            Some("sr_Latn")
        );
        assert_eq!(
            parent_locale_id("sr_Cyrl_RS", "sr_Cyrl_RS").as_deref(),
            Some("sr_RS")
        );
        assert_eq!(
            parent_locale_id("yue_HK", "yue_HK").as_deref(),
            Some("yue_Hant")
        );
        assert_eq!(
            parent_locale_id("sr_RS", "sr_Latn_RS").as_deref(),
            Some("sr_Latn")
        );
        assert_eq!(
            parent_locale_id("sr_Latn", "sr_Latn").as_deref(),
            Some("sr")
        );
        assert_eq!(parent_locale_id("sr", "sr"), None);
    }

    #[test]
    fn refuses_bad_files() {
        assert!(parse_pack(b"XXXX").is_none());
        assert!(parse_pack(b"ICP1\0\0\0\x01\0\x05ab").is_none());
        assert!(ResReader::new(&[0; 8]).is_err());
    }
}
