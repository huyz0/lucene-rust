//! Remote models, as ml-commons calls them through a connector: the request
//! a connector's action template makes for an input, signed when the
//! connector is AWS's, and the model tensors read back from the response.
//! The caller sends the request and runs any Painless pre- or post-process
//! script; this module does the rest, as ml-commons' `ConnectorUtils`,
//! `HttpConnector` and the built-in process functions do.

pub mod connector;
pub mod esdk;
pub mod json;
pub mod jsonpath;
pub mod sigv4;
pub mod substitute;
