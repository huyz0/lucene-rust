//! The one place this crate calls a cryptographic library: AES-256-GCM and
//! the random number generator, behind a backend chosen at build time.
//!
//! - Default (feature `rustcrypto`): RustCrypto's `aes-gcm` (AES-NI/PMULL
//!   where the CPU has them) and the operating system's CSPRNG (`getrandom`).
//! - Feature `fips`: AWS-LC's FIPS 140-3 validated module through
//!   `aws-lc-rs` (`aws-lc-fips-sys`), for both the cipher and the DRBG. A
//!   FIPS build is `--no-default-features --features fips`, so no non-validated
//!   AES code is linked; [`is_fips`] confirms at run time that the module is
//!   in FIPS mode (its power-on self tests passed). Building it needs CMake,
//!   Go, Perl and a C compiler; `scripts/build-fips.sh` is the release recipe.
//!
//! Both backends produce the same bytes: the file format does not depend on
//! the build, so an index written by one is read by the other.

use lucene_store::{Error, Result};

#[cfg(not(any(feature = "fips", feature = "rustcrypto")))]
compile_error!(
    "lucene-store-crypto needs a crypto backend: feature `rustcrypto` (default) or `fips`"
);

/// An AES-GCM tag's length.
pub const TAG_LEN: usize = 16;
/// An AES-GCM nonce's length.
pub const NONCE_LEN: usize = 12;

/// Which backend this build uses: `"aws-lc-fips"` or `"rustcrypto"`.
pub fn backend_name() -> &'static str {
    if cfg!(feature = "fips") {
        "aws-lc-fips"
    } else {
        "rustcrypto"
    }
}

/// Whether the cipher and DRBG run in a FIPS 140-3 validated module in FIPS
/// mode. Always `false` for the default backend.
pub fn is_fips() -> bool {
    imp::is_fips()
}

/// Fills `buf` from the backend's random generator.
pub fn fill_random(buf: &mut [u8]) -> Result<()> {
    imp::fill_random(buf)
}

/// An AES-256-GCM key, ready to seal and open.
pub struct Aes256Gcm(imp::Key);

impl Aes256Gcm {
    /// The cipher for a 256-bit `key`.
    pub fn new(key: &[u8; 32]) -> Result<Self> {
        imp::Key::new(key).map(Self)
    }

    /// Encrypts `buf` in place under `nonce` with associated data `aad`,
    /// returning the tag.
    pub fn seal(
        &self,
        nonce: &[u8; NONCE_LEN],
        aad: &[u8],
        buf: &mut [u8],
    ) -> Result<[u8; TAG_LEN]> {
        self.0.seal(nonce, aad, buf)
    }

    /// Decrypts `buf` in place, checking `tag`. `Err(())` on a failed
    /// authentication (the caller names what failed).
    #[allow(clippy::result_unit_err)]
    pub fn open(
        &self,
        nonce: &[u8; NONCE_LEN],
        aad: &[u8],
        buf: &mut [u8],
        tag: &[u8],
    ) -> std::result::Result<(), ()> {
        self.0.open(nonce, aad, buf, tag)
    }
}

fn failed(what: &str) -> Error {
    Error::Io(std::io::Error::other(format!("crypto backend: {what}")))
}

#[cfg(not(feature = "fips"))]
mod imp {
    use super::{failed, NONCE_LEN, TAG_LEN};
    use aes_gcm::aead::AeadInPlace;
    use aes_gcm::{KeyInit, Nonce, Tag};
    use lucene_store::Result;

    pub fn is_fips() -> bool {
        false
    }

    pub fn fill_random(buf: &mut [u8]) -> Result<()> {
        getrandom::fill(buf).map_err(|e| failed(&e.to_string()))
    }

    pub struct Key(aes_gcm::Aes256Gcm);

    impl Key {
        pub fn new(key: &[u8; 32]) -> Result<Self> {
            Ok(Self(aes_gcm::Aes256Gcm::new(key.into())))
        }

        pub fn seal(
            &self,
            nonce: &[u8; NONCE_LEN],
            aad: &[u8],
            buf: &mut [u8],
        ) -> Result<[u8; TAG_LEN]> {
            let tag = self
                .0
                .encrypt_in_place_detached(Nonce::from_slice(nonce), aad, buf)
                .map_err(|_| failed("seal"))?;
            let mut out = [0u8; TAG_LEN];
            out.copy_from_slice(&tag);
            Ok(out)
        }

        pub fn open(
            &self,
            nonce: &[u8; NONCE_LEN],
            aad: &[u8],
            buf: &mut [u8],
            tag: &[u8],
        ) -> std::result::Result<(), ()> {
            if tag.len() != TAG_LEN {
                return Err(());
            }
            self.0
                .decrypt_in_place_detached(Nonce::from_slice(nonce), aad, buf, Tag::from_slice(tag))
                .map_err(|_| ())
        }
    }
}

#[cfg(feature = "fips")]
mod imp {
    use super::{failed, NONCE_LEN, TAG_LEN};
    use aws_lc_rs::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
    use lucene_store::Result;

    pub fn is_fips() -> bool {
        aws_lc_rs::try_fips_mode().is_ok()
    }

    pub fn fill_random(buf: &mut [u8]) -> Result<()> {
        aws_lc_rs::rand::fill(buf).map_err(|_| failed("random"))
    }

    pub struct Key(LessSafeKey);

    impl Key {
        pub fn new(key: &[u8; 32]) -> Result<Self> {
            let key = UnboundKey::new(&AES_256_GCM, key).map_err(|_| failed("key"))?;
            Ok(Self(LessSafeKey::new(key)))
        }

        pub fn seal(
            &self,
            nonce: &[u8; NONCE_LEN],
            aad: &[u8],
            buf: &mut [u8],
        ) -> Result<[u8; TAG_LEN]> {
            let tag = self
                .0
                .seal_in_place_separate_tag(
                    Nonce::assume_unique_for_key(*nonce),
                    Aad::from(aad),
                    buf,
                )
                .map_err(|_| failed("seal"))?;
            let mut out = [0u8; TAG_LEN];
            out.copy_from_slice(tag.as_ref());
            Ok(out)
        }

        pub fn open(
            &self,
            nonce: &[u8; NONCE_LEN],
            aad: &[u8],
            buf: &mut [u8],
            tag: &[u8],
        ) -> std::result::Result<(), ()> {
            if tag.len() != TAG_LEN {
                return Err(());
            }
            let mut sealed = Vec::with_capacity(buf.len().saturating_add(TAG_LEN));
            sealed.extend_from_slice(buf);
            sealed.extend_from_slice(tag);
            let plain = self
                .0
                .open_in_place(
                    Nonce::assume_unique_for_key(*nonce),
                    Aad::from(aad),
                    &mut sealed,
                )
                .map_err(|_| ())?;
            buf.copy_from_slice(plain);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NIST SP 800-38D / McGrew-Viega test case 16 (AES-256, 60-byte
    /// plaintext, 20-byte AAD): both backends must reproduce it.
    #[test]
    fn a_known_answer_vector_seals_and_opens() {
        let hex = |s: &str| -> Vec<u8> {
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i.saturating_add(2)], 16).unwrap())
                .collect()
        };
        let key: [u8; 32] = hex("feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308")
            .try_into()
            .unwrap();
        let nonce: [u8; 12] = hex("cafebabefacedbaddecaf888").try_into().unwrap();
        let aad = hex("feedfacedeadbeeffeedfacedeadbeefabaddad2");
        let plain = hex(
            "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72\
             1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",
        );
        let cipher = hex(
            "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa\
             8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662",
        );
        let tag = hex("76fc6ece0f4e1768cddf8853bb2d551b");
        let aes = Aes256Gcm::new(&key).unwrap();
        let mut buf = plain.clone();
        assert_eq!(aes.seal(&nonce, &aad, &mut buf).unwrap().to_vec(), tag);
        assert_eq!(buf, cipher);
        aes.open(&nonce, &aad, &mut buf, &tag).unwrap();
        assert_eq!(buf, plain);
        let mut bad = cipher.clone();
        assert!(aes.open(&nonce, &aad[1..], &mut bad, &tag).is_err());
        assert!(aes.open(&nonce, &aad, &mut bad, &tag[1..]).is_err());
    }

    #[test]
    fn the_backend_reports_itself_and_draws_random_bytes() {
        let (mut a, mut b) = ([0u8; 32], [0u8; 32]);
        fill_random(&mut a).unwrap();
        fill_random(&mut b).unwrap();
        assert_ne!(a, b);
        assert_eq!(is_fips(), cfg!(feature = "fips"), "{}", backend_name());
        assert!(failed("x").to_string().contains("crypto backend: x"));
    }
}
