//! `InetAddressPoint`: an IPv4 or IPv6 address as a one-dimension, 16-byte
//! point.
//!
//! What reaches disk: the address's network-order bytes, an IPv4 address
//! first mapped into IPv6 (`::ffff:a.b.c.d`, `IPV4_PREFIX`), so both families
//! share one order.
//!
//! Rust shape: `java.net.InetAddress` is [`IpAddr`]. Java's `decode` returns
//! an `Inet4Address` for a mapped address (`InetAddress.getByAddress` does
//! the unmapping); [`InetAddressPoint::decode`] does the same.

use std::borrow::Cow;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use lucene_analysis::Analyzer;

use super::{illegal, FieldTokens, FieldType, IndexableField, Result};

/// `InetAddressPoint`.
#[derive(Debug, Clone, PartialEq)]
pub struct InetAddressPoint {
    name: String,
    field_type: FieldType,
    packed: [u8; 16],
}

impl InetAddressPoint {
    /// `BYTES`.
    pub const BYTES: usize = 16;

    /// `IPV4_PREFIX`.
    pub const IPV4_PREFIX: [u8; 12] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff];

    /// `TYPE`.
    pub fn field_type_of() -> FieldType {
        let mut ft = FieldType::new();
        ft.set_dimensions(1, 16).expect("a 16-byte point is valid");
        ft.frozen()
    }

    /// `new InetAddressPoint(name, point)`.
    pub fn new(name: impl Into<String>, point: IpAddr) -> Self {
        InetAddressPoint {
            name: name.into(),
            field_type: Self::field_type_of(),
            packed: Self::encode(point),
        }
    }

    /// `setInetAddressValue(value)`.
    pub fn set_inet_address_value(&mut self, value: IpAddr) {
        self.packed = Self::encode(value);
    }

    /// `MIN_VALUE`: `::`.
    pub const MIN_VALUE: IpAddr = IpAddr::V6(Ipv6Addr::UNSPECIFIED);

    /// `MAX_VALUE`: `ffff:...:ffff`.
    pub const MAX_VALUE: IpAddr = IpAddr::V6(Ipv6Addr::new(
        0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    ));

    /// `encode(value)`: the 16-byte form.
    pub fn encode(value: IpAddr) -> [u8; 16] {
        match value {
            IpAddr::V4(v4) => {
                let mut mapped = [0u8; 16];
                mapped[..12].copy_from_slice(&Self::IPV4_PREFIX);
                mapped[12..].copy_from_slice(&v4.octets());
                mapped
            }
            IpAddr::V6(v6) => v6.octets(),
        }
    }

    /// `decode(value)`: a 4- or 16-byte address; a mapped IPv4 address comes
    /// back as IPv4, as `InetAddress.getByAddress` returns it.
    pub fn decode(value: &[u8]) -> Result<IpAddr> {
        match value.len() {
            4 => {
                let b: [u8; 4] = value.try_into().expect("four bytes");
                Ok(IpAddr::V4(Ipv4Addr::from(b)))
            }
            16 => {
                let b: [u8; 16] = value.try_into().expect("sixteen bytes");
                if b[..12] == Self::IPV4_PREFIX {
                    let v4: [u8; 4] = b[12..].try_into().expect("four bytes");
                    return Ok(IpAddr::V4(Ipv4Addr::from(v4)));
                }
                Ok(IpAddr::V6(Ipv6Addr::from(b)))
            }
            _ => Err(illegal("encoded bytes are of incorrect length")),
        }
    }

    /// `nextUp(address)`: the next address in the 16-byte order.
    pub fn next_up(address: IpAddr) -> Result<IpAddr> {
        let n = u128::from_be_bytes(Self::encode(address));
        let up = n.checked_add(1).ok_or_else(|| {
            illegal(format!(
                "Overflow: there is no greater InetAddress than {address}"
            ))
        })?;
        Self::decode(&up.to_be_bytes())
    }

    /// `nextDown(address)`.
    pub fn next_down(address: IpAddr) -> Result<IpAddr> {
        let n = u128::from_be_bytes(Self::encode(address));
        let down = n.checked_sub(1).ok_or_else(|| {
            illegal(format!(
                "Underflow: there is no smaller InetAddress than {address}"
            ))
        })?;
        Self::decode(&down.to_be_bytes())
    }

    /// `newPrefixQuery`'s bounds: `value` with every bit past `prefix_length`
    /// cleared (lower) and set (upper), in the address's own family.
    pub fn prefix_bounds(value: IpAddr, prefix_length: u32) -> Result<(IpAddr, IpAddr)> {
        let bits = match value {
            IpAddr::V4(_) => 32,
            IpAddr::V6(_) => 128,
        };
        if prefix_length > bits {
            return Err(illegal(format!(
                "illegal prefixLength '{prefix_length}'. Must be 0-32 for IPv4 ranges, 0-128 \
                 for IPv6 ranges"
            )));
        }
        Ok(match value {
            IpAddr::V4(v4) => {
                let n = u32::from(v4);
                let mask = u32::MAX.checked_shr(prefix_length).unwrap_or(0);
                (
                    IpAddr::V4(Ipv4Addr::from(n & !mask)),
                    IpAddr::V4(Ipv4Addr::from(n | mask)),
                )
            }
            IpAddr::V6(v6) => {
                let n = u128::from(v6);
                let mask = u128::MAX.checked_shr(prefix_length).unwrap_or(0);
                (
                    IpAddr::V6(Ipv6Addr::from(n & !mask)),
                    IpAddr::V6(Ipv6Addr::from(n | mask)),
                )
            }
        })
    }

    /// `binaryValue()`.
    pub fn packed(&self) -> &[u8; 16] {
        &self.packed
    }
}

impl IndexableField for InetAddressPoint {
    fn name(&self) -> &str {
        &self.name
    }
    fn field_type(&self) -> &FieldType {
        &self.field_type
    }
    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        Some(Cow::Borrowed(&self.packed))
    }
    fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn encodes_v4_mapped_and_v6() {
        let e = InetAddressPoint::encode(ip("1.2.3.4"));
        assert_eq!(&e[..12], &InetAddressPoint::IPV4_PREFIX);
        assert_eq!(&e[12..], &[1, 2, 3, 4]);
        assert_eq!(InetAddressPoint::decode(&e).unwrap(), ip("1.2.3.4"));
        let v6 = ip("2001:db8::1");
        assert_eq!(
            InetAddressPoint::decode(&InetAddressPoint::encode(v6)).unwrap(),
            v6
        );
        assert_eq!(
            InetAddressPoint::decode(&[9, 8, 7, 6]).unwrap(),
            ip("9.8.7.6")
        );
        assert!(InetAddressPoint::decode(&[1, 2]).is_err());
        let mut p = InetAddressPoint::new("a", ip("::1"));
        p.set_inet_address_value(ip("10.0.0.1"));
        assert_eq!(p.packed()[15], 1);
        assert_eq!(p.binary_value().unwrap().len(), 16);
        assert_eq!(p.name(), "a");
        assert_eq!(p.field_type().point_num_bytes(), 16);
        assert!(p.token_stream(&Analyzer::keyword()).unwrap().is_none());
    }

    #[test]
    fn next_up_down_and_prefixes() {
        assert_eq!(
            InetAddressPoint::next_up(ip("1.2.3.4")).unwrap(),
            ip("1.2.3.5")
        );
        assert_eq!(
            InetAddressPoint::next_down(ip("1.2.3.0")).unwrap(),
            ip("1.2.2.255")
        );
        assert!(InetAddressPoint::next_up(InetAddressPoint::MAX_VALUE).is_err());
        assert!(InetAddressPoint::next_down(InetAddressPoint::MIN_VALUE).is_err());
        let (lo, hi) = InetAddressPoint::prefix_bounds(ip("192.168.7.9"), 16).unwrap();
        assert_eq!((lo, hi), (ip("192.168.0.0"), ip("192.168.255.255")));
        let (lo, hi) = InetAddressPoint::prefix_bounds(ip("192.168.7.9"), 0).unwrap();
        assert_eq!((lo, hi), (ip("0.0.0.0"), ip("255.255.255.255")));
        let (lo, hi) = InetAddressPoint::prefix_bounds(ip("192.168.7.9"), 32).unwrap();
        assert_eq!(lo, hi);
        let (lo, hi) = InetAddressPoint::prefix_bounds(ip("2001:db8::7"), 32).unwrap();
        assert_eq!(lo, ip("2001:db8::"));
        assert_eq!(hi, ip("2001:db8:ffff:ffff:ffff:ffff:ffff:ffff"));
        assert!(InetAddressPoint::prefix_bounds(ip("1.1.1.1"), 33).is_err());
    }
}
