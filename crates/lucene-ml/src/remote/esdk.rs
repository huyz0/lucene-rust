//! Connector credentials as ml-commons encrypts them: AWS Encryption SDK
//! messages (format version 2, framed) under the algorithm suite the SDK
//! requires with `RequireEncryptRequireDecrypt` -- AES-256-GCM with an
//! HKDF-SHA512 key commitment and an ECDSA P-384 signature
//! (`ALG_AES_256_GCM_HKDF_SHA512_COMMIT_KEY_ECDSA_P384`, 0x0578) -- with the
//! data key wrapped by the cluster's master key the way the SDK's
//! `JceMasterKey` wraps it (provider `Custom`, key id empty, AES-GCM).
//!
//! A stock node's encrypted credentials decrypt here and these decrypt on a
//! stock node: the master key is the same document either way.

use aws_lc_rs::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use aws_lc_rs::encoding::{AsBigEndian, EcPublicKeyCompressedBin};
use aws_lc_rs::hkdf::{Salt, HKDF_SHA512};
use aws_lc_rs::rand::{SecureRandom, SystemRandom};
use aws_lc_rs::signature::{
    EcdsaKeyPair, KeyPair, UnparsedPublicKey, ECDSA_P384_SHA384_ASN1,
    ECDSA_P384_SHA384_ASN1_SIGNING,
};

use crate::{MlError, Result};

const VERSION: u8 = 2;
const SUITE: u16 = 0x0578;
const FRAMED: u8 = 2;
const FRAME_LENGTH: u32 = 4096;
const FINAL_FRAME: u32 = 0xFFFF_FFFF;
const IV_LEN: usize = 12;
const TAG_LEN: usize = 16;
const KEY_LEN: usize = 32;
const PUBLIC_KEY: &str = "aws-crypto-public-key";
const PROVIDER: &str = "Custom";
const FRAME_AAD: &[u8] = b"AWSKMSEncryptionClient Frame";
const FINAL_FRAME_AAD: &[u8] = b"AWSKMSEncryptionClient Final Frame";

fn bad(why: &str) -> MlError {
    MlError::Model(format!("cannot decrypt credential: {why}"))
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .i
            .checked_add(n)
            .ok_or_else(|| bad("truncated message"))?;
        let s = self
            .b
            .get(self.i..end)
            .ok_or_else(|| bad("truncated message"))?;
        self.i = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let s = self.take(2)?;
        Ok(u16::from_be_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn short_bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.u16()? as usize;
        self.take(n)
    }
}

/// The encryption context, as the SDK serializes it: the pair count and
/// each key and value with its length, sorted by key.
fn serialize_context(context: &[(Vec<u8>, Vec<u8>)]) -> Vec<u8> {
    if context.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<_> = context.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = (sorted.len() as u16).to_be_bytes().to_vec();
    for (k, v) in sorted {
        out.extend_from_slice(&(k.len() as u16).to_be_bytes());
        out.extend_from_slice(k);
        out.extend_from_slice(&(v.len() as u16).to_be_bytes());
        out.extend_from_slice(v);
    }
    out
}

fn parse_context(bytes: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let mut r = Reader { b: bytes, i: 0 };
    let n = r.u16()?;
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let k = r.short_bytes()?.to_vec();
        let v = r.short_bytes()?.to_vec();
        out.push((k, v));
    }
    if r.i != bytes.len() {
        return Err(bad("malformed encryption context"));
    }
    Ok(out)
}

struct Len(usize);
impl aws_lc_rs::hkdf::KeyType for Len {
    fn len(&self) -> usize {
        self.0
    }
}

/// The message's encryption key and its commitment, from the data key.
fn derive(data_key: &[u8], message_id: &[u8]) -> Result<([u8; KEY_LEN], [u8; KEY_LEN])> {
    let prk = Salt::new(HKDF_SHA512, message_id).extract(data_key);
    let mut key = [0u8; KEY_LEN];
    let mut commit = [0u8; KEY_LEN];
    let mut info = SUITE.to_be_bytes().to_vec();
    info.extend_from_slice(b"DERIVEKEY");
    prk.expand(&[&info], Len(KEY_LEN))
        .and_then(|okm| okm.fill(&mut key))
        .map_err(|_| bad("key derivation"))?;
    prk.expand(&[b"COMMITKEY"], Len(KEY_LEN))
        .and_then(|okm| okm.fill(&mut commit))
        .map_err(|_| bad("key derivation"))?;
    Ok((key, commit))
}

fn gcm(key: &[u8]) -> Result<LessSafeKey> {
    Ok(LessSafeKey::new(
        UnboundKey::new(&AES_256_GCM, key).map_err(|_| bad("bad key length"))?,
    ))
}

fn open(
    key: &LessSafeKey,
    iv: &[u8],
    aad: &[u8],
    ciphertext: &[u8],
    tag: &[u8],
) -> Result<Vec<u8>> {
    let nonce = Nonce::try_assume_unique_for_key(iv).map_err(|_| bad("bad IV"))?;
    let mut buf = ciphertext.to_vec();
    buf.extend_from_slice(tag);
    let plain = key
        .open_in_place(nonce, Aad::from(aad), &mut buf)
        .map_err(|_| bad("authentication failed"))?;
    Ok(plain.to_vec())
}

fn seal(key: &LessSafeKey, iv: &[u8], aad: &[u8], plaintext: &[u8]) -> Result<Vec<u8>> {
    let nonce = Nonce::try_assume_unique_for_key(iv).map_err(|_| bad("bad IV"))?;
    let mut buf = plaintext.to_vec();
    key.seal_in_place_append_tag(nonce, Aad::from(aad), &mut buf)
        .map_err(|_| bad("encryption failed"))?;
    Ok(buf)
}

fn frame_aad(message_id: &[u8], final_frame: bool, seq: u32, len: u64) -> Vec<u8> {
    let mut aad = message_id.to_vec();
    aad.extend_from_slice(if final_frame {
        FINAL_FRAME_AAD
    } else {
        FRAME_AAD
    });
    aad.extend_from_slice(&seq.to_be_bytes());
    aad.extend_from_slice(&len.to_be_bytes());
    aad
}

fn frame_iv(seq: u32) -> [u8; IV_LEN] {
    let mut iv = [0u8; IV_LEN];
    iv[8..].copy_from_slice(&seq.to_be_bytes());
    iv
}

/// Decrypts an SDK message with the 32-byte master key.
pub fn decrypt(master_key: &[u8], message: &[u8]) -> Result<Vec<u8>> {
    let mut r = Reader { b: message, i: 0 };
    if r.u8()? != VERSION {
        return Err(bad("unsupported message format version"));
    }
    if r.u16()? != SUITE {
        return Err(bad("unsupported algorithm suite"));
    }
    let message_id = r.take(32)?;
    let context_bytes = r.short_bytes()?;
    let context = parse_context(context_bytes)?;
    let edk_count = r.u16()?;
    let mut wrapped = Vec::new();
    for _ in 0..edk_count {
        let provider = r.short_bytes()?;
        let info = r.short_bytes()?;
        let edk = r.short_bytes()?;
        wrapped.push((provider, info, edk));
    }
    if r.u8()? != FRAMED {
        return Err(bad("unsupported content type"));
    }
    let frame_length = r.u32()?;
    let commitment = r.take(KEY_LEN)?;
    let header_end = r.i;
    let header_tag = r.take(TAG_LEN)?;

    let master = gcm(master_key)?;
    let context_aad = serialize_context(&context);
    let data_key = wrapped
        .iter()
        .filter(|(p, _, _)| *p == PROVIDER.as_bytes())
        .find_map(|(_, info, edk)| {
            // Key id (empty) || tag length in bits || IV length || IV.
            let iv = info.strip_prefix(&[0, 0, 0, 128, 0, 0, 0, IV_LEN as u8][..])?;
            if iv.len() != IV_LEN || edk.len() != KEY_LEN + TAG_LEN {
                return None;
            }
            open(&master, iv, &context_aad, &edk[..KEY_LEN], &edk[KEY_LEN..]).ok()
        })
        .ok_or_else(|| bad("no data key decrypts with the master key"))?;
    let (key, commit) = derive(&data_key, message_id)?;
    if commit[..] != *commitment {
        return Err(bad("key commitment mismatch"));
    }
    let key = gcm(&key)?;
    open(
        &key,
        &[0u8; IV_LEN],
        &message[..header_end],
        &[],
        header_tag,
    )?;

    let mut plain = Vec::new();
    let mut seq = 1u32;
    loop {
        let marker = r.u32()?;
        let final_frame = marker == FINAL_FRAME;
        let frame_seq = if final_frame { r.u32()? } else { marker };
        if frame_seq != seq {
            return Err(bad("frame out of order"));
        }
        let iv = r.take(IV_LEN)?;
        let len = if final_frame { r.u32()? } else { frame_length };
        if final_frame && len > frame_length {
            return Err(bad("final frame too long"));
        }
        let ciphertext = r.take(len as usize)?;
        let tag = r.take(TAG_LEN)?;
        let aad = frame_aad(message_id, final_frame, seq, u64::from(len));
        plain.extend(open(&key, iv, &aad, ciphertext, tag)?);
        if final_frame {
            break;
        }
        seq = seq.checked_add(1).ok_or_else(|| bad("too many frames"))?;
    }
    let body_end = r.i;
    let signature = r.short_bytes()?;
    if r.i != message.len() {
        return Err(bad("trailing bytes"));
    }
    let public_key = context
        .iter()
        .find(|(k, _)| k == PUBLIC_KEY.as_bytes())
        .and_then(|(_, v)| base64_decode(std::str::from_utf8(v).ok()?))
        .ok_or_else(|| bad("no signing key"))?;
    UnparsedPublicKey::new(&ECDSA_P384_SHA384_ASN1, &public_key)
        .verify(&message[..body_end], signature)
        .map_err(|_| bad("signature does not verify"))?;
    Ok(plain)
}

/// A new master key, as ml-commons makes one: 32 random bytes, base64.
pub fn generate_master_key() -> Result<String> {
    let mut key = [0u8; 32];
    SystemRandom::new()
        .fill(&mut key)
        .map_err(|_| bad("no randomness"))?;
    Ok(base64_encode(&key))
}

/// The credential value ml-commons stores: `plaintext` encrypted under the
/// base64 master key, base64.
pub fn encrypt_credential(master_key: &str, plaintext: &str) -> Result<String> {
    let key = base64_decode(master_key).ok_or_else(|| bad("the master key is not base64"))?;
    encrypt(&key, plaintext.as_bytes()).map(|m| base64_encode(&m))
}

/// A stored credential value decrypted under the base64 master key.
pub fn decrypt_credential(master_key: &str, stored: &str) -> Result<String> {
    let key = base64_decode(master_key).ok_or_else(|| bad("the master key is not base64"))?;
    let message = base64_decode(stored).ok_or_else(|| bad("the credential is not base64"))?;
    String::from_utf8(decrypt(&key, &message)?).map_err(|_| bad("the credential is not UTF-8"))
}

/// Encrypts `plaintext` as the SDK does, under the 32-byte master key.
pub fn encrypt(master_key: &[u8], plaintext: &[u8]) -> Result<Vec<u8>> {
    let rng = SystemRandom::new();
    let fill = |buf: &mut [u8]| rng.fill(buf).map_err(|_| bad("no randomness"));
    let signer = EcdsaKeyPair::generate(&ECDSA_P384_SHA384_ASN1_SIGNING)
        .map_err(|_| bad("key generation"))?;
    let compressed: EcPublicKeyCompressedBin<'_> = signer
        .public_key()
        .as_be_bytes()
        .map_err(|_| bad("key encoding"))?;
    let context = vec![(
        PUBLIC_KEY.as_bytes().to_vec(),
        base64_encode(compressed.as_ref()).into_bytes(),
    )];
    let context_aad = serialize_context(&context);

    let mut message_id = [0u8; 32];
    fill(&mut message_id)?;
    let mut data_key = [0u8; KEY_LEN];
    fill(&mut data_key)?;
    let mut wrap_iv = [0u8; IV_LEN];
    fill(&mut wrap_iv)?;
    let edk = seal(&gcm(master_key)?, &wrap_iv, &context_aad, &data_key)?;
    let (key, commit) = derive(&data_key, &message_id)?;
    let key = gcm(&key)?;

    let mut out = vec![VERSION];
    out.extend_from_slice(&SUITE.to_be_bytes());
    out.extend_from_slice(&message_id);
    out.extend_from_slice(&(context_aad.len() as u16).to_be_bytes());
    out.extend_from_slice(&context_aad);
    out.extend_from_slice(&1u16.to_be_bytes());
    out.extend_from_slice(&(PROVIDER.len() as u16).to_be_bytes());
    out.extend_from_slice(PROVIDER.as_bytes());
    let mut info = vec![0, 0, 0, 128, 0, 0, 0, IV_LEN as u8];
    info.extend_from_slice(&wrap_iv);
    out.extend_from_slice(&(info.len() as u16).to_be_bytes());
    out.extend_from_slice(&info);
    out.extend_from_slice(&(edk.len() as u16).to_be_bytes());
    out.extend_from_slice(&edk);
    out.push(FRAMED);
    out.extend_from_slice(&FRAME_LENGTH.to_be_bytes());
    out.extend_from_slice(&commit);
    let tag = seal(&key, &[0u8; IV_LEN], &out, &[])?;
    out.extend_from_slice(&tag);

    let chunks: Vec<&[u8]> = if plaintext.is_empty() {
        vec![&[]]
    } else {
        plaintext.chunks(FRAME_LENGTH as usize).collect()
    };
    // The last frame is always a final frame, even when it is full.
    let last = chunks.len().saturating_sub(1);
    for (n, chunk) in chunks.iter().enumerate() {
        let seq = u32::try_from(n.saturating_add(1)).map_err(|_| bad("too many frames"))?;
        let final_frame = n == last;
        let iv = frame_iv(seq);
        if final_frame {
            out.extend_from_slice(&FINAL_FRAME.to_be_bytes());
        }
        out.extend_from_slice(&seq.to_be_bytes());
        out.extend_from_slice(&iv);
        if final_frame {
            out.extend_from_slice(&(chunk.len() as u32).to_be_bytes());
        }
        let aad = frame_aad(&message_id, final_frame, seq, chunk.len() as u64);
        out.extend(seal(&key, &iv, &aad, chunk)?);
    }
    let signature = signer.sign(&rng, &out).map_err(|_| bad("signing failed"))?;
    out.extend_from_slice(&(signature.as_ref().len() as u16).to_be_bytes());
    out.extend_from_slice(signature.as_ref());
    Ok(out)
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding, as `java.util.Base64.getEncoder()`.
// ARITH: the capacity is 4/3 of an in-memory slice's length, and the shift
// `18 - 6 * i` has `i` in 0..4.
#[allow(clippy::arithmetic_side_effects)]
pub fn base64_encode(b: &[u8]) -> String {
    let mut out = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= c.len() {
                out.push(B64[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Standard base64, as `java.util.Base64.getDecoder()` reads it.
// ARITH: `bits` stays in 0..14 (6 is added below 8, 8 taken at or above
// it), so the shifts are below 32; the capacity divides before multiplying.
#[allow(clippy::arithmetic_side_effects)]
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.trim_end_matches('=');
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 3);
    let mut acc = 0u32;
    let mut bits = 0;
    for c in s.bytes() {
        let v = B64.iter().position(|&x| x == c)? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    (bits < 6).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A credential a stock node encrypted, and its master key.
    const MASTER_KEY: &str = "NX/bPD1UpNIkPWG6gGqmH/xTQWrBOlrQSZBrpX/vJTw=";
    const STOCK: &str = "AgV4ttpR5SDDp+aHaXEeMBKLyg6ZGDsnVwqRI78an6j26ooAXwABABVhd3MtY3J5cHRvLXB1YmxpYy1rZXkAREFzeHJqaG11N2IreHE1ZzBGeVB0UEFLNnRLTTd4WExEOHJiWFFGQUpob0VjbjVmKzE4dE41NEs5Sk5jeXU1UTN1QT09AAEABkN1c3RvbQAUAAAAgAAAAAx67n5+rWHptlmTScQAMPtjaORb7EQB2FOov3WIAUD9jZPPzA3le7TD7uNeh5VIkqF04UTh9jcOeiDQJnM7OAIAABAAXFIWHxD2cFxTY7VgQM6FcW8vg169KP9CRdwxLwPiNNsWrPWg+loDrME52ev2x9ZP/////wAAAAEAAAAAAAAAAAAAAAEAAAAJQRfLFw25xDD26UsNCJrZPWjmPXWSAq7YlgBnMGUCMD8E1GUzvcH9l8jtW46rj0KSg5lXVONajFTFlgRxP01CncLkArBT37NMYsTjCvcpVwIxALnUyVk0WEzl1wSSEMeejfvKGikUav+eTNfMV7PXQHonSY5wX6fJwHuYPvDQ5APeAA==";

    #[test]
    fn decrypts_what_stock_encrypted() {
        let key = base64_decode(MASTER_KEY).unwrap();
        let msg = base64_decode(STOCK).unwrap();
        assert_eq!(decrypt(&key, &msg).unwrap(), b"sk-secret");
        let mut wrong = key.clone();
        wrong[0] ^= 1;
        assert!(decrypt(&wrong, &msg)
            .unwrap_err()
            .to_string()
            .contains("no data key"));
        let mut tampered = msg.clone();
        let n = tampered.len();
        tampered[n - 10] ^= 1;
        assert!(decrypt(&key, &tampered).is_err());
        for cut in [1, 3, 40, 200, n - 1] {
            assert!(decrypt(&key, &msg[..cut]).is_err(), "{cut}");
        }
    }

    #[test]
    fn round_trips() {
        let key = base64_decode(MASTER_KEY).unwrap();
        for len in [0, 1, 4095, 4096, 4097, 9000] {
            let plain: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let msg = encrypt(&key, &plain).unwrap();
            assert_eq!(decrypt(&key, &msg).unwrap(), plain, "{len}");
        }
        assert!(encrypt(&key[..16], b"x").is_err());
    }

    #[test]
    fn base64_matches_java() {
        for s in ["", "f", "fo", "foo", "foob", "fooba", "foobar"] {
            assert_eq!(
                base64_decode(&base64_encode(s.as_bytes())).unwrap(),
                s.as_bytes()
            );
        }
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert!(base64_decode("Zm9v!").is_none());
        assert!(base64_decode("Z").is_none());
        assert!(parse_context(&[0, 1, 0]).is_err());
        assert!(serialize_context(&[]).is_empty());
    }
}
