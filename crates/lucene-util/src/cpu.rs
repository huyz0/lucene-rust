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
    fn the_process_level_is_stable_and_consistent() {
        assert_eq!(level(), level());
        assert!(!avx512_popcount() || level() == Level::Avx512);
        assert!(level() <= detect().level);
    }
}
