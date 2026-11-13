//! The vector instructions this process may use beyond the build's
//! baseline, found once per process.
//!
//! The build targets x86-64-v3 (AVX2, FMA) or aarch64 (NEON), and kernels for
//! those are picked at compile time. AVX-512 is not in that baseline, so a
//! kernel that has an AVX-512 variant asks [`level`] (or [`avx512_popcount`])
//! which one to run. Both are resolved on first use and cached: the check
//! is a load after that, never a CPUID or an environment read per call.
//!
//! `LUCENE_RUST_SIMD` caps what is used, read once with the detection:
//! `avx2` (or `neon`, or `scalar`) keeps every run-time choice on the
//! baseline kernels, and `avx512` (the default) allows AVX-512 where the CPU
//! has it. It cannot raise the level above what the CPU supports, and it
//! does not change kernels chosen at compile time.

use std::sync::OnceLock;

/// The widest vector instruction set a run-time choice may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// No vector kernels beyond what the compiler chose.
    Scalar,
    /// aarch64's baseline.
    Neon,
    /// x86-64-v3's baseline.
    Avx2,
    /// AVX-512F and AVX-512BW.
    Avx512,
}

#[derive(Debug, Clone, Copy)]
struct Features {
    level: Level,
    /// AVX-512 VPOPCNTDQ as well as [`Level::Avx512`].
    popcount: bool,
}

fn features() -> Features {
    static FEATURES: OnceLock<Features> = OnceLock::new();
    *FEATURES.get_or_init(|| {
        let cap = std::env::var("LUCENE_RUST_SIMD").ok();
        resolve(detect(), cap.as_deref())
    })
}

/// What the CPU supports: the baseline, and on x86-64 whether it has the
/// AVX-512 subsets the kernels use.
fn detect() -> Features {
    #[cfg(target_arch = "x86_64")]
    {
        let avx512 = std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw");
        let base = if cfg!(target_feature = "avx2") {
            Level::Avx2
        } else {
            Level::Scalar
        };
        Features {
            level: if avx512 { Level::Avx512 } else { base },
            popcount: avx512 && std::arch::is_x86_feature_detected!("avx512vpopcntdq"),
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        Features {
            level: Level::Neon,
            popcount: false,
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Features {
            level: Level::Scalar,
            popcount: false,
        }
    }
}

/// `detected`, lowered to the `LUCENE_RUST_SIMD` cap when one is set.
/// `avx2`, `neon` and `scalar` all mean the build's baseline (no run-time
/// choice tests for anything below AVX-512), whichever architecture this
/// is. An unknown value leaves the detection as it is, with a warning.
fn resolve(detected: Features, cap: Option<&str>) -> Features {
    let cap = match cap.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("scalar") => Level::Scalar,
        Some("neon") => Level::Neon,
        Some("avx2") => Level::Avx2,
        None | Some("avx512") => return detected,
        Some(other) => {
            eprintln!("LUCENE_RUST_SIMD={other:?} is not avx512, avx2, neon or scalar; ignored");
            return detected;
        }
    };
    let level = detected.level.min(cap);
    Features {
        level,
        popcount: detected.popcount && level == Level::Avx512,
    }
}

/// The widest instruction set run-time kernel choices may use.
#[inline]
pub fn level() -> Level {
    features().level
}

/// Whether AVX-512 VPOPCNTDQ may be used: eight words counted per
/// instruction.
#[inline]
pub fn avx512_popcount() -> bool {
    features().popcount
}

/// The CPU features the build's baseline was compiled to require and that
/// this host lacks, found once per process. Empty when the host meets it.
///
/// A build for `x86-64-v3` (`.cargo/config.toml`) lets the compiler emit
/// AVX2, FMA, BMI and the rest anywhere, so on an older CPU the first such
/// instruction is `SIGILL`, with no message. A node asks this at startup
/// and refuses to serve with a named reason instead. This is best effort
/// from inside the same binary: the check itself is built for the baseline.
/// A deployment that must never crash can also test `/proc/cpuinfo` before
/// exec. aarch64 (NEON is architectural) and other targets require nothing.
pub fn missing_baseline() -> &'static [&'static str] {
    static MISSING: OnceLock<Vec<&'static str>> = OnceLock::new();
    MISSING.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            missing_from(&required_x86(), detect_x86)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Vec::new()
        }
    })
}

/// [`missing_baseline`] as a `Result`, for a startup check.
pub fn check_baseline() -> Result<(), BaselineError> {
    as_result(missing_baseline())
}

fn as_result(missing: &[&'static str]) -> Result<(), BaselineError> {
    match missing {
        [] => Ok(()),
        m => Err(BaselineError {
            missing: m.to_vec(),
        }),
    }
}

/// The host lacks CPU features the build requires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineError {
    /// Feature names as `std::arch` spells them (`avx2`, `fma`, ...).
    pub missing: Vec<&'static str>,
}

impl std::fmt::Display for BaselineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "this build needs CPU features the host lacks: {} \
             (rebuild for a lower baseline or run on a newer CPU)",
            self.missing.join(", ")
        )
    }
}

impl std::error::Error for BaselineError {}

/// The members of `required` that `has` denies, in order.
#[cfg_attr(not(any(test, target_arch = "x86_64")), allow(dead_code))]
fn missing_from(required: &[&'static str], has: impl Fn(&str) -> bool) -> Vec<&'static str> {
    required.iter().copied().filter(|f| !has(f)).collect()
}

/// The x86-64-v3 features this build was compiled with: only those the
/// compiler was told to assume.
#[cfg(target_arch = "x86_64")]
fn required_x86() -> Vec<&'static str> {
    let mut r = Vec::new();
    macro_rules! need {
        ($($f:literal),*) => { $( if cfg!(target_feature = $f) { r.push($f); } )* };
    }
    need!("avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "lzcnt", "movbe", "popcnt");
    r
}

#[cfg(target_arch = "x86_64")]
fn detect_x86(f: &str) -> bool {
    match f {
        "avx" => std::arch::is_x86_feature_detected!("avx"),
        "avx2" => std::arch::is_x86_feature_detected!("avx2"),
        "bmi1" => std::arch::is_x86_feature_detected!("bmi1"),
        "bmi2" => std::arch::is_x86_feature_detected!("bmi2"),
        "f16c" => std::arch::is_x86_feature_detected!("f16c"),
        "fma" => std::arch::is_x86_feature_detected!("fma"),
        "lzcnt" => std::arch::is_x86_feature_detected!("lzcnt"),
        "movbe" => std::arch::is_x86_feature_detected!("movbe"),
        "popcnt" => std::arch::is_x86_feature_detected!("popcnt"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: Features = Features {
        level: Level::Avx512,
        popcount: true,
    };

    #[test]
    fn a_cap_lowers_the_level_and_never_raises_it() {
        let r = resolve(ALL, Some("avx2"));
        assert_eq!((r.level, r.popcount), (Level::Avx2, false));
        let r = resolve(ALL, Some(" Scalar "));
        assert_eq!((r.level, r.popcount), (Level::Scalar, false));
        let r = resolve(ALL, None);
        assert_eq!((r.level, r.popcount), (Level::Avx512, true));
        let r = resolve(ALL, Some("avx512"));
        assert_eq!((r.level, r.popcount), (Level::Avx512, true));
        let base = Features {
            level: Level::Avx2,
            popcount: false,
        };
        let r = resolve(base, Some("avx512"));
        assert_eq!((r.level, r.popcount), (Level::Avx2, false));
        let r = resolve(base, Some("bogus"));
        assert_eq!(r.level, Level::Avx2);
    }

    #[test]
    fn missing_lists_exactly_the_required_features_the_cpu_lacks() {
        let have = |f: &str| f == "avx2" || f == "fma";
        assert_eq!(
            missing_from(&["avx2", "fma", "bmi2", "movbe"], have),
            vec!["bmi2", "movbe"]
        );
        assert!(missing_from(&["avx2", "fma"], have).is_empty());
        assert!(missing_from(&[], |_| false).is_empty());
    }

    #[test]
    fn this_host_meets_the_baseline_it_was_built_for() {
        // The test binary runs here, so it was built for a CPU like this one.
        assert!(missing_baseline().is_empty(), "{:?}", missing_baseline());
        assert_eq!(check_baseline(), Ok(()));
    }

    #[test]
    fn a_missing_feature_is_an_error_and_none_is_ok() {
        assert_eq!(as_result(&[]), Ok(()));
        assert_eq!(
            as_result(&["avx2"]),
            Err(BaselineError {
                missing: vec!["avx2"]
            })
        );
    }

    #[test]
    fn the_baseline_error_names_every_missing_feature() {
        let e = BaselineError {
            missing: vec!["avx2", "fma"],
        };
        assert_eq!(
            e.to_string(),
            "this build needs CPU features the host lacks: avx2, fma \
             (rebuild for a lower baseline or run on a newer CPU)"
        );
    }

    #[test]
    fn the_process_level_is_stable_and_consistent() {
        assert_eq!(level(), level());
        assert!(!avx512_popcount() || level() == Level::Avx512);
        assert!(level() <= detect().level);
    }
}
