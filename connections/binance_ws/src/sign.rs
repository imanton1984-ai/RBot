use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

pub fn sign_hmac_sha256_hex(secret: &str, payload: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(payload.as_bytes());
    let res = mac.finalize().into_bytes();
    hex::encode(res)
}