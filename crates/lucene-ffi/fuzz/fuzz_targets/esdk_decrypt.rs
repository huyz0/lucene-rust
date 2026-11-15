//! The AWS Encryption SDK message parser behind connector credentials, on bytes any index writer
//! can store: a wrong-length key or a damaged message is an `Err`, never a panic or a
//! length-sized allocation; a message produced by `encrypt` decrypts to its plaintext.
#![no_main]

use libfuzzer_sys::fuzz_target;
use lucene_ml::remote::esdk;

fuzz_target!(|data: &[u8]| {
    let Some((&k, rest)) = data.split_first() else { return };
    let key = [k; 32];
    let _ = esdk::decrypt(&key, rest);
    let _ = esdk::decrypt(&key[..usize::from(k) % 33], rest);
    if let Ok(msg) = esdk::encrypt(&key, rest) {
        assert_eq!(esdk::decrypt(&key, &msg).expect("own message"), rest);
    }
    if let Ok(text) = std::str::from_utf8(rest) {
        let _ = esdk::base64_decode(text);
        let _ = esdk::decrypt_credential(&esdk::base64_encode(&key), text);
    }
});
