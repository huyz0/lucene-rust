//! Mapping the library's memory accounting ([`lucene_util::memory`]) onto
//! statuses.

use lucene_util::memory::{Exceeded, Reason};

use crate::status::{set_last_error, Status};

/// The status and message for a request a bound or cancel stopped.
pub fn exceeded(e: &Exceeded) -> Status {
    set_last_error(e.message.clone());
    match e.reason {
        Reason::Cancelled => Status::Cancelled,
        Reason::Request | Reason::Process => Status::MemoryLimit,
    }
}
