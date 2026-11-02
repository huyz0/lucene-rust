//! The one place this crate calls a cryptographic library: AES-256-GCM and
//! the random number generator, from AWS-LC through `aws-lc-rs`.
//!
//! - Default (feature `aws-lc`): the regular AWS-LC build.
//! - Feature `fips`: AWS-LC's FIPS 140-3 validated module
//!   (`aws-lc-fips-sys`). Build it with `--no-default-features --features
//!   fips` so the non-validated library is not linked; [`is_fips`] confirms
//!   at run time that the module is in FIPS mode (its power-on self tests
//!   passed). Building it needs CMake, Go, Perl and a C compiler;
//!   `scripts/build-fips.sh` is the release recipe.
//!
//! Only FIPS-approved services are used, in both builds: sealing goes through
//! `RandomizedNonceKey` (the module draws each 96-bit GCM nonce from its own
//! DRBG, as SP 800-38D requires of a FIPS module) and opening through
//! `open_in_place`. Both builds produce the same format, so an index written
//! by one is read by the other.

use aws_lc_rs::aead::{Aad, Nonce, RandomizedNonceKey, AES_256_GCM};
use lucene_store::{Error, Result};

#[cfg(not(any(feature = "fips", feature = "aws-lc")))]
compile_error!("lucene-store-crypto needs a crypto backend: feature `aws-lc` (default) or `fips`");

/// An AES-GCM tag's length.
pub const TAG_LEN: usize = 16;
/// An AES-GCM nonce's length.
pub const NONCE_LEN: usize = 12;

/// Which build of AWS-LC this is: `"aws-lc-fips"` or `"aws-lc"`.
pub fn backend_name() -> &'static str {
    if cfg!(feature = "fips") {
        "aws-lc-fips"
    } else {
        "aws-lc"
    }
}

/// Whether the cipher and DRBG run in a FIPS 140-3 validated module in FIPS
/// mode. `false` for the default build.
pub fn is_fips() -> bool {
    cfg!(feature = "fips") && aws_lc_rs::try_fips_mode().is_ok()
}

/// Fills `buf` from AWS-LC's DRBG.
pub fn fill_random(buf: &mut [u8]) -> Result<()> {
    aws_lc_rs::rand::fill(buf).map_err(|_| failed("random"))
}

fn failed(what: &str) -> Error {
    Error::Io(std::io::Error::other(format!("crypto backend: {what}")))
}

/// An AES-256-GCM key, ready to seal and open.
pub struct Aes256Gcm(RandomizedNonceKey);

impl Aes256Gcm {
    /// The cipher for a 256-bit `key`.
    pub fn new(key: &[u8; 32]) -> Result<Self> {
        RandomizedNonceKey::new(&AES_256_GCM, key)
            .map(Self)
            .map_err(|_| failed("key"))
    }

    /// Encrypts `buf` in place with associated data `aad` under a nonce the
    /// module draws, returning `(nonce, tag)`.
    pub fn seal(&self, aad: &[u8], buf: &mut [u8]) -> Result<([u8; NONCE_LEN], [u8; TAG_LEN])> {
        let (nonce, tag) = self
            .0
            .seal_in_place_separate_tag(Aad::from(aad), buf)
            .map_err(|_| failed("seal"))?;
        let mut t = [0u8; TAG_LEN];
        t.copy_from_slice(tag.as_ref());
        Ok((*nonce.as_ref(), t))
    }

    /// Decrypts `sealed` (ciphertext followed by its tag) in place; on success
    /// its first `sealed.len() - TAG_LEN` bytes are the plaintext. `Err(())`
    /// on a failed authentication (the caller names what failed).
    #[allow(clippy::result_unit_err)]
    pub fn open(
        &self,
        nonce: &[u8; NONCE_LEN],
        aad: &[u8],
        sealed: &mut [u8],
    ) -> std::result::Result<(), ()> {
        self.0
            .open_in_place(Nonce::from(nonce), Aad::from(aad), sealed)
            .map(|_| ())
            .map_err(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NIST SP 800-38D / McGrew-Viega test case 16 (AES-256, 60-byte
    /// plaintext, 20-byte AAD), opened: the module is real AES-GCM.
    #[test]
    fn a_known_answer_vector_opens() {
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
        let mut sealed = hex(
            "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa\
             8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662\
             76fc6ece0f4e1768cddf8853bb2d551b",
        );
        let aes = Aes256Gcm::new(&key).unwrap();
        let mut bad = sealed.clone();
        assert!(aes.open(&nonce, &aad[1..], &mut bad).is_err());
        aes.open(&nonce, &aad, &mut sealed).unwrap();
        assert_eq!(&sealed[..plain.len()], &plain[..]);
    }

    #[test]
    fn seal_draws_fresh_nonces_and_round_trips() {
        let aes = Aes256Gcm::new(&[7; 32]).unwrap();
        let mut a = b"same plaintext".to_vec();
        let mut b = a.clone();
        let (na, ta) = aes.seal(b"x", &mut a).unwrap();
        let (nb, _) = aes.seal(b"x", &mut b).unwrap();
        assert_ne!(na, nb, "the module draws a new nonce per seal");
        assert_ne!(a, b);
        a.extend_from_slice(&ta);
        aes.open(&na, b"x", &mut a).unwrap();
        assert_eq!(&a[..14], b"same plaintext");
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
