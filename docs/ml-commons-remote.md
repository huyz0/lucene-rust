# ml-commons remote connectors

How the plugin serves ml-commons' remote models and connectors, and what is compared with stock. The parity ledger rows are in [parity/ml-commons.md](parity/ml-commons.md).

## ml-commons `Connector`, `HttpConnector`, `AwsConnector`, `ConnectorUtils`, `RemoteConnectorExecutor`

Rust: `lucene-ml/src/remote/connector.rs::{Connector, Protocol, Prepared, Tensor}`; `lucene-ml/src/remote/mod.rs`.

**rust-only** (port of ml-commons). Connector validation, `${parameters.*}`/`${credential.*}` substitution (StringSubstitutor's rules), commons-text `escapeJson`, payloads checked as lenient org.json reads them, built-in pre/post-process functions, `response_filter`, text_docs splitting by `input_docs_processed_step_size`; Painless pre/post scripts are handed back to the plugin. The plugin sends the requests (JDK client, HTTP/1.1) on stock's `opensearch_ml_predict_remote` pool, a split input's at once; `client_config` timeouts in seconds, retries on 429, 5xx and AWS's throttling header up to `max_retry_times` (-1: no limit) within `retry_timeout_seconds`. Not yet: `mcp_sse`/`mcp_streamable_http` (accepted at create, refused at deploy), `skip_ssl_verification`, mutual TLS, `max_connection`, trusted-private-endpoint and restricted-IP settings. Tests: unit tests; `scripts/check-ml-plugin.py` remote section (provider blueprints on a mock server: OpenAI, Azure OpenAI, Bedrock Titan and Claude, Anthropic Messages, Cohere embed and rerank, Gemini, DeepSeek, Mistral; Painless scripts, error statuses, retries, timeouts, the private-IP check).

## StringSubstitutor, Gson, org.json, Jayway JsonPath

Rust: `lucene-ml/src/remote/substitute.rs`, `lucene-ml/src/remote/json.rs`, `lucene-ml/src/remote/jsonpath.rs`.

**rust-only**. What connectors use: substitution with defaults and escapes; lenient parsing and Gson's escaping; JsonPath dot, bracket, index, wildcard and filter paths. Tests: unit tests.

## AWS SDK v2 `AwsV4HttpSigner`

Rust: `lucene-ml/src/remote/sigv4.rs::sign`.

**rust-only**. Path encoded twice, every header signed, `x-amz-content-sha256` always the payload hash, session token as `X-Amz-Security-Token`. Tests: signatures from the Java SDK; the remote section recomputes each signature.

## ml-commons `EncryptorImpl`, AWS Encryption SDK

Rust: `lucene-ml/src/remote/esdk.rs::{generate_master_key, encrypt_credential, decrypt_credential}`.

**rust-only**. Credentials stored as stock stores them (AES-256-GCM, HKDF-SHA384, ECDSA P-384 signature; master key in `.plugins-ml-config`), on aws-lc-rs (FIPS feature available). Tests: unit tests (decrypts a stock ciphertext).

## ml-commons `GoogleCredentialProvider`, google-auth-library `ServiceAccountCredentials`

Rust: `lucene-ml/src/remote/google.rs::{token_request, refresh_request, validate_token_uri}`.

**rust-only**. The `google_cloud` (Vertex AI) connector's service-account JWT bearer assertion (RS256) and token_uri checks; ADC from a `GOOGLE_APPLICATION_CREDENTIALS` file (service account or `authorized_user` refresh token) or the metadata server, in the plugin. Stock forces Google's token host, so only refusals are compared. Tests: unit tests (signature verified); remote section refusals.

## the plugin's FFM bridge

Rust: `lucene-ffi/src/ml_remote.rs::{ffi_ml_master_key_alloc, ffi_ml_credential_encrypt_alloc, ffi_ml_remote_open, ffi_ml_remote_call_alloc}`.

**rust-only**. A connector handle answering JSON requests (`split`, `prepare`, `parse`, `google_token_request`). Tests: unit tests.
