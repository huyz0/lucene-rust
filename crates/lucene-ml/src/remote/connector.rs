//! A connector and what ml-commons does with it for one predict call: the
//! inputs split into requests, each request's parameters, payload, URL and
//! headers, and the model tensors read back from each response.

use std::collections::HashMap;

use super::json::{escape_json, gson_inner, gson_quote, is_json, is_json_or_ndjson, Json};
use super::jsonpath::{JsonPath, PathError};
use super::sigv4::{self, url_parts::Url, Signing};
use super::substitute::substitute;
use crate::{MlError, Result};

const PARAMETERS: &str = "${parameters.";
const CREDENTIAL: &str = "${credential.";

/// How requests are authenticated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// Plain HTTP; credentials go in headers.
    Http,
    /// AWS Signature Version 4.
    AwsSigV4,
}

/// One action of a connector.
#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    /// `PREDICT`, `EXECUTE`, ...
    pub action_type: String,
    /// A named action's name.
    pub name: Option<String>,
    /// The HTTP method.
    pub method: String,
    /// The URL template.
    pub url: String,
    /// Header templates.
    pub headers: Vec<(String, String)>,
    /// The request body template.
    pub request_body: Option<String>,
    /// A built-in pre-process function's name or a Painless script.
    pub pre_process_function: Option<String>,
    /// A built-in post-process function's name or a Painless script.
    pub post_process_function: Option<String>,
}

/// A connector with its credentials decrypted.
#[derive(Debug, Clone, PartialEq)]
pub struct Connector {
    /// How requests are authenticated.
    pub protocol: Protocol,
    /// The connector's parameters.
    pub parameters: HashMap<String, String>,
    /// The decrypted credentials.
    pub credential: HashMap<String, String>,
    /// The actions.
    pub actions: Vec<Action>,
}

/// The input of a predict call.
#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    /// `text_docs`.
    TextDocs(Vec<Option<String>>),
    /// A query and the documents to score against it.
    TextSimilarity {
        /// The query.
        query: String,
        /// The documents.
        docs: Vec<String>,
    },
    /// Parameters for the connector's templates (a remote model's own input).
    Remote(Vec<(String, Option<String>)>),
}

/// A Painless script the caller runs as a template script, with `params`
/// (a JSON object) as its parameters; its string result goes back to
/// [`Connector::prepare`] or [`Connector::parse`].
#[derive(Debug, Clone, PartialEq)]
pub struct Script {
    /// The script source, the default `escape` method added where it calls
    /// one without defining it.
    pub source: String,
    /// The script's parameters, as a JSON object.
    pub params: String,
}

/// An HTTP request to send.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    /// The method.
    pub method: String,
    /// The URL.
    pub url: String,
    /// The headers, in order.
    pub headers: Vec<(String, String)>,
    /// The body.
    pub body: Vec<u8>,
}

/// One request of a call, prepared.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    /// The request.
    pub request: Request,
    /// The parameters it was made from, which reading its response uses.
    pub parameters: HashMap<String, Option<String>>,
}

/// What preparing a request needs next.
#[derive(Debug, Clone, PartialEq)]
pub enum Prepared {
    /// The request is ready.
    Call(Box<Call>),
    /// A pre-process script must run first; prepare again with its result.
    Script(Script),
}

/// One model tensor read from a response.
#[derive(Debug, Clone, PartialEq)]
pub enum Tensor {
    /// A tensor of numbers.
    Numbers {
        /// Its name.
        name: String,
        /// `FLOAT32`, `INT8`, ...
        data_type: String,
        /// Its shape.
        shape: Vec<usize>,
        /// Its values, as the response wrote them.
        data: Vec<Json>,
    },
    /// A JSON map (`dataAsMap`), under the name `response`.
    Map(Json),
    /// Tensors a post-process script wrote as model tensor JSON.
    Raw(Json),
}

/// What reading a response needs next.
#[derive(Debug, Clone, PartialEq)]
pub enum Parsed {
    /// The response's tensors.
    Tensors(Vec<Tensor>),
    /// A post-process script must run on the response (its parameters are
    /// the response, a JSON array under `result`); parse again with its
    /// result.
    Script(Script),
}

/// The built-in pre-process functions.
const PRE_FUNCTIONS: &[&str] = &[
    "connector.pre_process.cohere.embedding",
    "connector.pre_process.cohere.multimodal_embedding",
    "connector.pre_process.openai.embedding",
    "connector.pre_process.bedrock.embedding",
    "connector.pre_process.bedrock.multimodal_embedding",
    "connector.pre_process.bedrock.nova.text_embedding",
    "connector.pre_process.bedrock.nova.image_embedding",
    "connector.pre_process.default.embedding",
    "connector.pre_process.cohere.rerank",
    "connector.pre_process.bedrock.rerank",
    "connector.pre_process.default.rerank",
];
const DEFAULT_EMBEDDING_INPUT: &str = "connector.pre_process.default.embedding";
const BEDROCK_EMBEDDING_INPUT: &str = "connector.pre_process.bedrock.embedding";

/// The built-in post-process functions and their default response paths.
const POST_FUNCTIONS: &[(&str, &str)] = &[
    (
        "connector.post_process.openai.embedding",
        "$.data[*].embedding",
    ),
    ("connector.post_process.cohere.embedding", "$.embeddings"),
    (
        "connector.post_process.cohere_v2.embedding.float",
        "$.embeddings.float",
    ),
    (
        "connector.post_process.cohere_v2.embedding.int8",
        "$.embeddings.int8",
    ),
    (
        "connector.post_process.cohere_v2.embedding.uint8",
        "$.embeddings.uint8",
    ),
    (
        "connector.post_process.cohere_v2.embedding.binary",
        "$.embeddings.binary",
    ),
    (
        "connector.post_process.cohere_v2.embedding.ubinary",
        "$.embeddings.ubinary",
    ),
    ("connector.post_process.default.embedding", "$[*]"),
    ("connector.post_process.bedrock.embedding", "$.embedding"),
    (
        "connector.post_process.bedrock_v2.embedding.float",
        "$.embeddingsByType.float",
    ),
    (
        "connector.post_process.bedrock_v2.embedding.binary",
        "$.embeddingsByType.binary",
    ),
    (
        "connector.post_process.bedrock.nova.embedding",
        "$.embeddings[*].embedding",
    ),
    ("connector.post_process.cohere.rerank", "$.results"),
    ("connector.post_process.bedrock.rerank", "$.results"),
    ("connector.post_process.default.rerank", "$[*]"),
];

/// ml-commons' result data types, as a response filter may name one.
const DATA_TYPES: &[&str] = &[
    "FLOAT32", "FLOAT64", "FLOAT16", "INT64", "INT32", "INT16", "INT8", "UINT8", "BOOLEAN",
    "STRING", "UNKNOWN",
];

/// Headers a `${parameters.*}` placeholder may not fill.
const BLOCKED_DYNAMIC_HEADERS: &[&str] = &[
    "authorization",
    "x-api-key",
    "api-key",
    "x-amz-security-token",
    "cookie",
    "set-cookie",
    "proxy-authorization",
    "x-auth-token",
    "x-access-token",
    "x-csrf-token",
    "x-xsrf-token",
    "x-auth-header",
    "x-forwarded-for",
    "x-real-ip",
    "x-client-ip",
    "cf-connecting-ip",
    "true-client-ip",
    "x-originating-ip",
    "host",
    "x-forwarded-host",
    "x-forwarded-server",
    "forwarded",
];

fn arg(msg: impl Into<String>) -> MlError {
    MlError::InvalidArgument(msg.into())
}

fn str_field(v: &Json, k: &str) -> Option<String> {
    match v.get(k)? {
        Json::String(s) => Some(s.clone()),
        Json::Number(n) => Some(n.clone()),
        Json::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// A map of strings, a value that is not a string kept as its JSON (as
/// ml-commons' `getParameterMap` keeps it).
fn string_map(v: Option<&Json>) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Some(Json::Object(m)) = v {
        for (k, v) in m {
            let s = match v {
                Json::String(s) => s.clone(),
                other => other.to_compact(),
            };
            out.insert(k.clone(), s);
        }
    }
    out
}

impl Connector {
    /// Reads a connector document (with `credential` holding the decrypted
    /// values).
    pub fn from_json(doc: &str) -> Result<Connector> {
        let v = Json::parse(doc).ok_or_else(|| arg("the connector is not JSON"))?;
        let protocol = match str_field(&v, "protocol").as_deref() {
            Some("http") => Protocol::Http,
            Some("aws_sigv4") => Protocol::AwsSigV4,
            Some(p) => return Err(arg(format!("unsupported connector protocol: {p}"))),
            None => return Err(arg("connector protocol is null")),
        };
        let mut actions = Vec::new();
        if let Some(Json::Array(a)) = v.get("actions") {
            for a in a {
                let mut headers = Vec::new();
                if let Some(Json::Object(h)) = a.get("headers") {
                    for (k, v) in h {
                        if let Json::String(s) = v {
                            headers.push((k.clone(), s.clone()));
                        }
                    }
                }
                actions.push(Action {
                    action_type: str_field(a, "action_type")
                        .unwrap_or_default()
                        .to_ascii_uppercase(),
                    name: str_field(a, "name"),
                    method: str_field(a, "method").unwrap_or_default(),
                    url: str_field(a, "url").unwrap_or_default(),
                    headers,
                    request_body: str_field(a, "request_body"),
                    pre_process_function: str_field(a, "pre_process_function"),
                    post_process_function: str_field(a, "post_process_function"),
                });
            }
        }
        Ok(Connector {
            protocol,
            parameters: string_map(v.get("parameters")),
            credential: string_map(v.get("credential")),
            actions,
        })
    }

    /// The action `action` names: its type, or else its name.
    pub fn find_action(&self, action: &str) -> Option<&Action> {
        const TYPES: &[&str] = &[
            "PREDICT",
            "EXECUTE",
            "BATCH_PREDICT",
            "CANCEL_BATCH_PREDICT",
            "BATCH_PREDICT_STATUS",
        ];
        if TYPES.iter().any(|t| t.eq_ignore_ascii_case(action)) {
            self.actions
                .iter()
                .find(|a| a.action_type.eq_ignore_ascii_case(action))
        } else {
            self.actions
                .iter()
                .find(|a| a.name.as_deref() == Some(action))
        }
    }

    fn action(&self, action: &str) -> Result<&Action> {
        self.find_action(action)
            .ok_or_else(|| arg(format!("no {action} action found")))
    }

    /// The inputs of each request a call makes: text documents go one per
    /// request to Bedrock and to a pre-process script, all together
    /// otherwise, or in steps of the connector's
    /// `input_docs_processed_step_size`.
    pub fn split(&self, action: &str, input: Input) -> Result<Vec<Input>> {
        let Input::TextDocs(docs) = input else {
            return Ok(vec![input]);
        };
        let step = if let Some(s) = self.parameters.get("input_docs_processed_step_size") {
            let s: i64 = s
                .parse()
                .map_err(|_| arg(format!("For input string: \"{s}\"")))?;
            if s <= 0 {
                return Err(arg(
                    "Invalid parameter: input_docs_processed_step_size. It must be positive integer.",
                ));
            }
            s as usize
        } else {
            match self.action(action)?.pre_process_function.as_deref() {
                Some(f) if f == BEDROCK_EMBEDDING_INPUT || !PRE_FUNCTIONS.contains(&f) => 1,
                _ => docs.len().max(1),
            }
        };
        if docs.is_empty() {
            return Ok(vec![Input::TextDocs(docs)]);
        }
        Ok(docs
            .chunks(step)
            .map(|c| Input::TextDocs(c.to_vec()))
            .collect())
    }

    /// Prepares the request for one input (one of [`Connector::split`]'s):
    /// `algorithm` is the predict call's function name (`TEXT_EMBEDDING`,
    /// `REMOTE`, ...), `extra` the call's model parameters, `script_output`
    /// the result of the pre-process script a first call asked for, and
    /// `amz_date` the signing time (`yyyyMMddTHHmmssZ`) for an AWS
    /// connector.
    pub fn prepare(
        &self,
        action: &str,
        algorithm: &str,
        input: &Input,
        extra: &[(String, Option<String>)],
        script_output: Option<&str>,
        amz_date: &str,
    ) -> Result<Prepared> {
        let act = self.action(action)?;
        let mut params: HashMap<String, Option<String>> = self
            .parameters
            .iter()
            .map(|(k, v)| (k.clone(), Some(v.clone())))
            .collect();
        let mut input_params: Vec<(String, Option<String>)> = Vec::new();
        if let Input::Remote(p) = input {
            input_params = escape_params(p);
            params.extend(input_params.iter().cloned());
        }
        params.extend(extra.iter().cloned());

        let processed = match self.pre_process(act, algorithm, input, &params, script_output)? {
            Ok(p) => p,
            Err(script) => return Ok(Prepared::Script(script)),
        };
        params.extend(escape_params(&processed));
        params.extend(input_params);

        let payload = self.payload(act, &mut params)?;
        if !flag(&params, "skip_validating_missing_parameters") {
            validate_payload(payload.as_deref())?;
        }
        let request = self.request(act, action, &params, payload, amz_date)?;
        Ok(Prepared::Call(Box::new(Call {
            request,
            parameters: params,
        })))
    }

    /// The input's parameters after the pre-process function, or the script
    /// to run for them.
    fn pre_process(
        &self,
        act: &Action,
        algorithm: &str,
        input: &Input,
        params: &HashMap<String, Option<String>>,
        script_output: Option<&str>,
    ) -> Result<std::result::Result<Params, Script>> {
        let function = match (&act.pre_process_function, input) {
            (Some(f), _) => Some(fill_function(params, f)?),
            (None, Input::TextDocs(_)) => Some(DEFAULT_EMBEDDING_INPUT.to_string()),
            (None, _) => None,
        };
        let Some(function) = function else {
            return match input {
                Input::Remote(p) => Ok(Ok(p.clone())),
                _ => Err(arg("pre_process_function not defined in connector")),
            };
        };
        if PRE_FUNCTIONS.contains(&function.as_str()) {
            return builtin_pre(&function, input).map(Ok);
        }
        let script_params = match input {
            Input::Remote(p) => {
                if !flag(
                    params,
                    "pre_process_function.process_remote_inference_input",
                ) {
                    return Ok(Ok(p.clone()));
                }
                let to_object = flag(
                    params,
                    "pre_process_function.convert_remote_inference_param_to_object",
                );
                Json::Object(
                    p.iter()
                        .map(|(k, v)| {
                            let v = match v {
                                None => Json::Null,
                                Some(s) if to_object && is_json(s) => {
                                    Json::parse_lenient(s).unwrap_or(Json::Null)
                                }
                                Some(s) => Json::String(s.clone()),
                            };
                            (k.clone(), v)
                        })
                        .collect(),
                )
            }
            other => {
                let doc = ml_input_json(algorithm, &escape_input(other));
                if flag(params, "pre_process_function.convert_input_to_json_string") {
                    let Json::Object(m) = doc else { unreachable!() };
                    Json::Object(
                        m.into_iter()
                            .map(|(k, v)| (k, json_string_value(v)))
                            .collect(),
                    )
                } else {
                    doc
                }
            }
        };
        match script_output {
            None => Ok(Err(Script {
                source: add_default_escape(&function),
                params: script_params.to_compact(),
            })),
            Some(out) => script_parameters(out).map(Ok),
        }
    }

    /// The request body: the action's template filled, or `http_body`.
    fn payload(
        &self,
        act: &Action,
        params: &mut HashMap<String, Option<String>>,
    ) -> Result<Option<String>> {
        let Some(template) = &act.request_body else {
            return Ok(params.get("http_body").cloned().flatten());
        };
        let mut payload = template.clone();
        for key in null_default_keys(template) {
            if params.get(&key).cloned().flatten().is_none() {
                payload = payload.replace(&format!("\"${{parameters.{key}:-null}}\""), "null");
            }
        }
        add_to_string_params(params);
        let payload = substitute(&payload, PARAMETERS, params)?;
        if !is_json_or_ndjson(&payload) {
            return Err(arg(format!("Invalid payload: {payload}")));
        }
        Ok(Some(payload))
    }

    fn request(
        &self,
        act: &Action,
        action: &str,
        params: &HashMap<String, Option<String>>,
        payload: Option<String>,
        amz_date: &str,
    ) -> Result<Request> {
        let method = act.method.to_ascii_uppercase();
        if !["POST", "GET", "PUT", "DELETE"].contains(&method.as_str()) {
            return Err(arg("unsupported http method"));
        }
        let body = payload.unwrap_or_default().into_bytes();
        if method == "POST" && body.is_empty() && action != "CANCEL_BATCH_PREDICT" {
            return Err(arg("Content length is 0. Aborting request to remote model"));
        }
        let url = substitute(&act.url, PARAMETERS, params)?;
        let parsed = Url::parse(&url).ok_or_else(|| {
            arg("Encountered error when trying to create uri from endpoint in ml connector. Please update the endpoint in connection configuration: ")
        })?;
        let mut headers = Vec::new();
        for (k, v) in &act.headers {
            let v = substitute(v, CREDENTIAL, &self.credential)?;
            if v.contains(PARAMETERS)
                && BLOCKED_DYNAMIC_HEADERS.contains(&k.to_ascii_lowercase().as_str())
            {
                return Err(arg(format!(
                    "Header '{k}' cannot use ${{parameters.*}} placeholders for security reasons. Use ${{credential.*}} instead."
                )));
            }
            let v = if v.contains(PARAMETERS) {
                let s = substitute(&v, PARAMETERS, params)?;
                if s.contains(PARAMETERS) {
                    return Err(arg(format!(
                        "Header '{k}' contains unresolved placeholder. Required parameter is missing."
                    )));
                }
                s
            } else {
                v
            };
            headers.push((k.clone(), v));
        }
        validate_headers(&headers)?;
        if !headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        {
            headers.push(("Content-Type".into(), "application/json".into()));
        }
        if !headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        {
            headers.push(("Content-Length".into(), body.len().to_string()));
        }
        if self.protocol == Protocol::AwsSigV4 {
            let get = |k: &str| self.credential.get(k).map(String::as_str);
            let from_params = |k: &str| {
                self.parameters
                    .get(k)
                    .map(String::as_str)
                    .or_else(|| get(k))
            };
            let signing = Signing {
                access_key: get("access_key").ok_or_else(|| arg("Missing credential"))?,
                secret_key: get("secret_key").ok_or_else(|| arg("Missing credential"))?,
                session_token: get("session_token"),
                service: from_params("service_name").ok_or_else(|| arg("Missing service name"))?,
                region: from_params("region").ok_or_else(|| arg("Missing region"))?,
            };
            sigv4::sign(&method, &parsed, &mut headers, &body, &signing, amz_date);
        }
        Ok(Request {
            method,
            url,
            headers,
            body,
        })
    }

    /// Reads a successful response of `call`; `script_output` is the result
    /// of the post-process script a first call asked for.
    pub fn parse(
        &self,
        action: &str,
        call: &Call,
        body: &str,
        script_output: Option<&str>,
    ) -> Result<Parsed> {
        let act = self.action(action)?;
        let params = &call.parameters;
        let function = match &act.post_process_function {
            Some(f) => Some(fill_function(params, f)?),
            None => None,
        };
        let filter = params.get("response_filter").cloned().flatten();
        if let Some(f) = function.as_deref() {
            if let Some((_, default)) = POST_FUNCTIONS.iter().find(|(n, _)| *n == f) {
                let path = match filter.as_deref() {
                    Some(p) if !p.trim().is_empty() => p.to_string(),
                    _ => default.to_string(),
                };
                let doc = Json::parse(body).ok_or_else(|| {
                    MlError::Model(format!(
                        "Fail to execute {action}: the response is not JSON"
                    ))
                })?;
                let value = JsonPath::compile(&path)
                    .and_then(|p| p.read(&doc))
                    .map_err(|e| MlError::Model(e.to_string()))?;
                let data_type = DATA_TYPES
                    .iter()
                    .find(|t| path.to_ascii_uppercase().contains(&format!(".{t}")))
                    .copied();
                return builtin_post(f, value, data_type).map(Parsed::Tensors);
            }
        }
        let (response, from_script) = match (function, script_output) {
            (Some(f), None) => {
                let result = match Json::parse(body) {
                    Some(v @ Json::Object(_)) => v,
                    Some(v @ Json::Array(_)) => Json::Object(vec![("result".into(), v)]),
                    _ => {
                        return Err(arg(
                            "Invalid JSON format: the response is not a JSON object or array",
                        ))
                    }
                };
                return Ok(Parsed::Script(Script {
                    source: add_default_escape(&f),
                    params: result.to_compact(),
                }));
            }
            (Some(_), Some(out)) => (out.to_string(), true),
            (None, _) => (body.to_string(), false),
        };
        if from_script && is_json(&response) {
            let v = Json::parse_lenient(&response).unwrap_or(Json::Null);
            let tensors = match v {
                Json::Array(a) => a.into_iter().map(Tensor::Raw).collect(),
                other => vec![Tensor::Raw(other)],
            };
            return Ok(Parsed::Tensors(tensors));
        }
        let value = match filter {
            None => Json::String(response),
            Some(f) => match Json::parse(&response) {
                Some(doc) => match JsonPath::compile(&f).and_then(|p| p.read(&doc)) {
                    Ok(v) => v,
                    Err(PathError::NotFound(_)) => Json::String(response),
                    Err(e) => return Err(MlError::Model(e.to_string())),
                },
                None => Json::String(response),
            },
        };
        Ok(Parsed::Tensors(vec![response_tensor(value)?]))
    }
}

/// The tensor `parseResponse` makes of a response (or the part a filter
/// read): a JSON object as the map, a JSON array or other value under
/// `response`.
fn response_tensor(v: Json) -> Result<Tensor> {
    let as_map = |v: Json| match v {
        m @ Json::Object(_) => m,
        other => Json::Object(vec![("response".into(), other)]),
    };
    Ok(Tensor::Map(match v {
        Json::String(s) if is_json(&s) => as_map(Json::parse_lenient(&s).unwrap_or(Json::Null)),
        other => as_map(other),
    }))
}

/// Whether a parameter is `true` (Java's `Boolean.parseBoolean`).
fn flag(params: &HashMap<String, Option<String>>, key: &str) -> bool {
    params
        .get(key)
        .and_then(|v| v.as_deref())
        .is_some_and(|v| v.eq_ignore_ascii_case("true"))
}

/// `escapeRemoteInferenceInputData`: a value that is not a JSON object or
/// array escaped for a JSON string, unless `no_escape_params` names it.
fn escape_params(params: &[(String, Option<String>)]) -> Vec<(String, Option<String>)> {
    let keep: Vec<&str> = params
        .iter()
        .rev()
        .find(|(k, _)| k == "no_escape_params")
        .and_then(|(_, v)| v.as_deref())
        .map(|v| v.split(',').map(str::trim).collect())
        .unwrap_or_default();
    params
        .iter()
        .map(|(k, v)| {
            let v = v.as_ref().map(|v| {
                if is_json(v) || keep.contains(&k.as_str()) {
                    v.clone()
                } else {
                    escape_json(v)
                }
            });
            (k.clone(), v)
        })
        .collect()
}

/// A function name or script with `${parameters.x}` filled by each
/// parameter's Gson JSON.
fn fill_function(params: &HashMap<String, Option<String>>, f: &str) -> Result<String> {
    if !f.contains(PARAMETERS) {
        return Ok(f.to_string());
    }
    let quoted: HashMap<String, Option<String>> = params
        .iter()
        .map(|(k, v)| {
            let s = match v {
                Some(v) => {
                    let mut out = String::new();
                    gson_quote(v, &mut out);
                    out
                }
                None => "null".into(),
            };
            (k.clone(), Some(s))
        })
        .collect();
    substitute(f, PARAMETERS, &quoted)
}

/// The text inputs Gson-escaped, as `escapeMLInput` does before a script.
fn escape_input(input: &Input) -> Input {
    match input {
        Input::TextDocs(d) => {
            Input::TextDocs(d.iter().map(|s| s.as_deref().map(gson_inner)).collect())
        }
        Input::TextSimilarity { query, docs } => Input::TextSimilarity {
            query: gson_inner(query),
            docs: docs.iter().map(|s| gson_inner(s)).collect(),
        },
        other => other.clone(),
    }
}

/// `MLInput.toXContent` for a script's parameters.
fn ml_input_json(algorithm: &str, input: &Input) -> Json {
    let mut m = vec![("algorithm".to_string(), Json::String(algorithm.to_string()))];
    let docs = |d: &[Option<String>]| {
        Json::Array(
            d.iter()
                .map(|s| s.clone().map_or(Json::Null, Json::String))
                .collect(),
        )
    };
    match input {
        Input::TextDocs(d) if !d.is_empty() => m.push(("text_docs".into(), docs(d))),
        Input::TextSimilarity { query, docs: d } => {
            m.push(("query_text".into(), Json::String(query.clone())));
            if !d.is_empty() {
                m.push((
                    "text_docs".into(),
                    Json::Array(d.iter().cloned().map(Json::String).collect()),
                ));
            }
        }
        _ => {}
    }
    Json::Object(m)
}

/// `convertScriptStringToJsonString` of one value: a string as it is,
/// anything else as its Gson JSON (numbers read by Gson print as doubles).
fn json_string_value(v: Json) -> Json {
    match v {
        s @ Json::String(_) => s,
        other => Json::String(other.to_gson_doubles()),
    }
}

/// The `parameters` of a pre-process script's result.
fn script_parameters(out: &str) -> Result<Vec<(String, Option<String>)>> {
    let v = Json::parse_lenient(out)
        .ok_or_else(|| arg(format!("Preprocess function output is not JSON: {out}")))?;
    let Some(Json::Object(m)) = v.get("parameters") else {
        return Ok(Vec::new());
    };
    Ok(m.iter()
        .map(|(k, v)| {
            let s = match v {
                Json::String(s) => s.clone(),
                other => other.to_gson_doubles(),
            };
            (k.clone(), Some(s))
        })
        .collect())
}

fn gson_list(items: &[Option<String>]) -> String {
    Json::Array(
        items
            .iter()
            .map(|s| s.clone().map_or(Json::Null, Json::String))
            .collect(),
    )
    .to_compact()
}

/// The built-in pre-process functions.
fn builtin_pre(function: &str, input: &Input) -> Result<Vec<(String, Option<String>)>> {
    if let Input::Remote(p) = input {
        // Each built-in function passes a remote model's own input through.
        return Ok(p.clone());
    }
    let docs = |input: &Input| -> Result<Vec<Option<String>>> {
        match input {
            Input::TextDocs(d) => Ok(d.clone()),
            _ => Err(arg(
                "This pre_process_function can only support TextDocsInputDataSet which including a list of string with key 'text_docs'",
            )),
        }
    };
    let one = |k: &str, v: Option<String>| vec![(k.to_string(), v)];
    Ok(match function {
        "connector.pre_process.openai.embedding" | "connector.pre_process.default.embedding" => {
            one("input", Some(gson_list(&docs(input)?)))
        }
        "connector.pre_process.cohere.embedding" => one("texts", Some(gson_list(&docs(input)?))),
        "connector.pre_process.bedrock.embedding"
        | "connector.pre_process.bedrock.nova.text_embedding" => {
            let d = docs(input)?;
            let first = d
                .into_iter()
                .next()
                .ok_or_else(|| arg("Index 0 out of bounds for length 0"))?;
            one("inputText", first)
        }
        "connector.pre_process.bedrock.nova.image_embedding" => {
            let d = docs(input)?;
            let first = d
                .into_iter()
                .next()
                .ok_or_else(|| arg("No input image provided"))?;
            one("inputImage", first)
        }
        "connector.pre_process.bedrock.multimodal_embedding" => {
            let d = docs(input)?;
            if d.is_empty() || (d.len() == 1 && d[0].is_none()) {
                return Err(arg("No input text or image provided"));
            }
            let mut p = one("inputText", d[0].clone());
            if d.len() > 1 {
                p.push(("inputImage".into(), d[1].clone()));
            }
            p
        }
        "connector.pre_process.cohere.multimodal_embedding" => {
            let d = docs(input)?;
            if d.first().is_none_or(|f| f.is_none()) {
                return Err(arg("No image provided"));
            }
            one("images", Some(gson_list(&d)))
        }
        "connector.pre_process.cohere.rerank" | "connector.pre_process.default.rerank" => {
            let Input::TextSimilarity { query, docs } = input else {
                return Err(arg(
                    "This pre_process_function can only support TextSimilarityInputDataSet",
                ));
            };
            vec![
                ("query".into(), Some(query.clone())),
                (
                    "documents".into(),
                    Some(gson_list(
                        &docs.iter().cloned().map(Some).collect::<Vec<_>>(),
                    )),
                ),
                ("top_n".into(), Some(docs.len().to_string())),
            ]
        }
        "connector.pre_process.bedrock.rerank" => {
            let Input::TextSimilarity { query, docs } = input else {
                return Err(arg(
                    "This pre_process_function can only support TextSimilarityInputDataSet",
                ));
            };
            let s = |t: &str| Json::String(t.to_string());
            let queries = Json::Array(vec![Json::Object(vec![
                (
                    "textQuery".into(),
                    Json::Object(vec![("text".into(), s(query))]),
                ),
                ("type".into(), s("TEXT")),
            ])]);
            let sources = Json::Array(
                docs.iter()
                    .map(|d| {
                        Json::Object(vec![
                            (
                                "inlineDocumentSource".into(),
                                Json::Object(vec![
                                    (
                                        "textDocument".into(),
                                        Json::Object(vec![("text".into(), s(d))]),
                                    ),
                                    ("type".into(), s("TEXT")),
                                ]),
                            ),
                            ("type".into(), s("INLINE")),
                        ])
                    })
                    .collect(),
            );
            vec![
                ("queries".into(), Some(queries.to_compact())),
                ("sources".into(), Some(sources.to_compact())),
                ("numberOfResults".into(), Some(docs.len().to_string())),
            ]
        }
        other => return Err(arg(format!("unknown pre_process_function {other}"))),
    })
}

/// The built-in post-process functions, on the value the response path
/// read.
fn builtin_post(function: &str, value: Json, data_type: Option<&str>) -> Result<Vec<Tensor>> {
    let not_list = || MlError::Model("Post process function input is not a List.".into());
    let ty = data_type.unwrap_or("FLOAT32").to_string();
    let tensor = |values: Vec<Json>| Tensor::Numbers {
        name: "sentence_embedding".into(),
        data_type: ty.clone(),
        shape: vec![values.len()],
        data: values,
    };
    match function {
        "connector.post_process.bedrock.embedding"
        | "connector.post_process.bedrock_v2.embedding.float"
        | "connector.post_process.bedrock_v2.embedding.binary" => {
            let Json::Array(a) = value else {
                return Err(not_list());
            };
            if a.first().is_some_and(|v| !matches!(v, Json::Number(_))) {
                return Err(MlError::Model(
                    "The embedding should be a non-empty List containing Float values.".into(),
                ));
            }
            Ok(vec![tensor(a)])
        }
        "connector.post_process.cohere.rerank"
        | "connector.post_process.default.rerank"
        | "connector.post_process.bedrock.rerank" => {
            let bedrock = function.ends_with("bedrock.rerank");
            let score_key = if bedrock {
                "relevanceScore"
            } else {
                "relevance_score"
            };
            let Json::Array(a) = value else {
                return Err(not_list());
            };
            if bedrock && a.is_empty() {
                return Err(MlError::Model(
                    "Post process function input is empty.".into(),
                ));
            }
            let mut scores = vec![Json::Null; a.len()];
            for r in &a {
                let (Some(Json::Number(i)), Some(score)) = (r.get("index"), r.get(score_key))
                else {
                    return Err(MlError::Model(if bedrock {
                        "Rerank result should have both index and relevanceScore.".into()
                    } else {
                        "The rerank result should contain index and relevance_score.".into()
                    }));
                };
                let i: usize = i
                    .parse()
                    .map_err(|_| MlError::Model("bad rerank index".into()))?;
                *scores.get_mut(i).ok_or_else(|| {
                    MlError::Model(format!("Index {i} out of bounds for length {}", a.len()))
                })? = score.clone();
            }
            Ok(scores
                .into_iter()
                .map(|s| Tensor::Numbers {
                    name: "similarity".into(),
                    data_type: "FLOAT32".into(),
                    shape: vec![1],
                    data: vec![s],
                })
                .collect())
        }
        _ => {
            let Json::Array(a) = value else {
                return Err(not_list());
            };
            if let Some(first) = a.first() {
                let Json::Array(inner) = first else {
                    return Err(MlError::Model(
                        "The embedding should be a non-empty List containing List of Float values."
                            .into(),
                    ));
                };
                if !matches!(inner.first(), Some(Json::Number(_))) {
                    return Err(MlError::Model(
                        "The embedding should be a non-empty List containing Float values.".into(),
                    ));
                }
            }
            a.into_iter()
                .map(|e| match e {
                    Json::Array(v) => Ok(tensor(v)),
                    _ => Err(MlError::Model(
                        "The embedding should be a non-empty List containing List of Float values."
                            .into(),
                    )),
                })
                .collect()
        }
    }
}

/// Parameters as a call holds them: a name and its value, if any.
type Params = Vec<(String, Option<String>)>;

/// The keys of `"${parameters.key:-null}"` placeholders.
// ARITH: offsets just past an ASCII match `find` returned in `rest`.
#[allow(clippy::arithmetic_side_effects)]
fn null_default_keys(template: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = template;
    while let Some(i) = rest.find("\"${parameters.") {
        let after = &rest[i + 14..];
        let key: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !key.is_empty() && after[key.len()..].starts_with(":-null}\"") {
            out.push(key);
        }
        rest = &rest[i + 1..];
    }
    out
}

/// `parseParameters`: for each `${parameters.x.toString()}` some value
/// holds, `x.toString()` is `x` Gson-escaped.
// ARITH: offsets just past an ASCII match `find` returned in `rest`.
#[allow(clippy::arithmetic_side_effects)]
fn add_to_string_params(params: &mut HashMap<String, Option<String>>) {
    let mut names = Vec::new();
    for v in params.values().flatten() {
        let mut rest = v.as_str();
        while let Some(i) = rest.find(PARAMETERS) {
            let after = &rest[i + PARAMETERS.len()..];
            if let Some(end) = after.find(".toString()}") {
                let name = &after[..end];
                if !name.is_empty() && !name.contains('}') {
                    names.push(name.to_string());
                }
            }
            rest = &rest[i + 1..];
        }
    }
    for name in names {
        if let Some(Some(v)) = params.get(&name) {
            let v = gson_inner(v);
            params.insert(format!("{name}.toString()"), Some(v));
        }
    }
}

/// `validatePayload`: every placeholder must be filled.
// ARITH: offsets just past an ASCII match `find` returned in `rest`.
#[allow(clippy::arithmetic_side_effects)]
fn validate_payload(payload: Option<&str>) -> Result<()> {
    let Some(p) = payload.filter(|p| p.contains("${parameters")) else {
        return Ok(());
    };
    let mut names = Vec::new();
    let mut rest = p;
    while let Some(i) = rest.find(PARAMETERS) {
        let after = &rest[i + PARAMETERS.len()..];
        match after.find('}') {
            Some(end) if end > 0 => {
                names.push(&after[..end]);
                rest = &after[end + 1..];
            }
            _ => rest = after,
        }
    }
    Err(arg(format!(
        "Some parameter placeholder not filled in payload: {}",
        names.join(", ")
    )))
}

/// `validateSubstitutedHeaders`: no control characters, 8 KB a header, 64
/// KB in all.
fn validate_headers(headers: &[(String, String)]) -> Result<()> {
    let mut total = 0;
    for (k, v) in headers {
        if v.chars().any(|c| (c < ' ' && c != '\t') || c == '\u{7f}') {
            return Err(arg("Header value contains invalid control character"));
        }
        let size = k
            .encode_utf16()
            .count()
            .saturating_add(v.encode_utf16().count());
        if size > 8192 {
            return Err(arg("Header size (key + value) exceeds 8KB limit"));
        }
        total = size.saturating_add(total);
    }
    if total > 65536 {
        return Err(arg("Total headers size (key + value) exceeds 64KB limit"));
    }
    Ok(())
}

/// ml-commons adds a default `escape` method to a script that calls one
/// without defining it.
fn add_default_escape(script: &str) -> String {
    const DEFAULT_ESCAPE_FUNCTION: &str = "\n    String escape(def input) { \n      if (input.contains(\"\\\\\")) {\n        input = input.replace(\"\\\\\", \"\\\\\\\\\");\n      }\n      if (input.contains(\"\\\"\")) {\n        input = input.replace(\"\\\"\", \"\\\\\\\"\");\n      }\n      if (input.contains('\r')) {\n        input = input = input.replace('\r', '\\\\r');\n      }\n      if (input.contains(\"\\\\t\")) {\n        input = input.replace(\"\\\\t\", \"\\\\\\\\\\\\t\");\n      }\n      if (input.contains('\n')) {\n        input = input.replace('\n', '\\\\n');\n      }\n      if (input.contains('\u{8}')) {\n        input = input.replace('\u{8}', '\\\\b');\n      }\n      if (input.contains('\u{c}')) {\n        input = input.replace('\u{c}', '\\\\f');\n      }\n      return input;\n    }\n";
    let defines = defines_escape(script);
    if !defines && calls_escape(script) {
        format!("{DEFAULT_ESCAPE_FUNCTION}{script}")
    } else {
        script.to_string()
    }
}

/// `(?<!\bString\s+)\bescape\s*\(`.
// ARITH: offsets around an ASCII match `find` returned in `s`; `start` is
// taken after `before` was seen to end with the six bytes of `String`.
#[allow(clippy::arithmetic_side_effects)]
fn calls_escape(s: &str) -> bool {
    let b = s.as_bytes();
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut from = 0;
    while let Some(i) = s[from..].find("escape").map(|i| i + from) {
        from = i + 1;
        if i > 0 && word(b[i - 1]) {
            continue;
        }
        let after = s[i + 6..].trim_start();
        if !after.starts_with('(') {
            continue;
        }
        let before = s[..i].trim_end();
        if before.len() < s[..i].len() && before.ends_with("String") {
            let start = before.len() - 6;
            if start == 0 || !word(b[start - 1]) {
                continue;
            }
        }
        return true;
    }
    false
}

/// `String\s+escape\s*\(\s*(def|String)\s+.*?\)\s*\{?`.
// ARITH: offsets just past an ASCII match `find` returned in `s`.
#[allow(clippy::arithmetic_side_effects)]
fn defines_escape(s: &str) -> bool {
    let mut from = 0;
    while let Some(i) = s[from..].find("String").map(|i| i + from) {
        from = i + 1;
        let rest = &s[i + 6..];
        let trimmed = rest.trim_start();
        if trimmed.len() == rest.len() {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix("escape") else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('(') else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest
            .strip_prefix("def")
            .or_else(|| rest.strip_prefix("String"))
        else {
            continue;
        };
        let after = rest.trim_start();
        if after.len() < rest.len() && after.contains(')') {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connector(protocol: &str, action: &str, extra: &str) -> Connector {
        Connector::from_json(&format!(
            r#"{{"name":"c","protocol":"{protocol}","parameters":{{"model":"m","endpoint":"localhost:8791"{extra}}},
               "credential":{{"key":"sk-secret","access_key":"AKIDEXAMPLE","secret_key":"wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"}},
               "actions":[{action}]}}"#
        ))
        .unwrap()
    }

    fn call(
        c: &Connector,
        algorithm: &str,
        input: &Input,
        extra: &[(String, Option<String>)],
    ) -> Result<Call> {
        match c.prepare("predict", algorithm, input, extra, None, "20261111T022922Z")? {
            Prepared::Call(call) => Ok(*call),
            Prepared::Script(s) => panic!("script {s:?}"),
        }
    }

    fn body(c: &Call) -> String {
        String::from_utf8(c.request.body.clone()).unwrap()
    }

    fn header<'a>(c: &'a Call, k: &str) -> Option<&'a str> {
        c.request
            .headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(k))
            .map(|(_, v)| v.as_str())
    }

    fn docs(d: &[&str]) -> Input {
        Input::TextDocs(d.iter().map(|s| Some(s.to_string())).collect())
    }

    fn remote(p: &[(&str, &str)]) -> Input {
        Input::Remote(
            p.iter()
                .map(|(k, v)| (k.to_string(), Some(v.to_string())))
                .collect(),
        )
    }

    const OPENAI: &str = r#"{"action_type":"predict","method":"POST","url":"http://${parameters.endpoint}/v1/openai/embeddings",
        "headers":{"Authorization":"Bearer ${credential.key}"},
        "request_body":"{ \"input\": ${parameters.input}, \"model\": \"${parameters.model}\" }",
        "pre_process_function":"connector.pre_process.openai.embedding","post_process_function":"connector.post_process.openai.embedding"}"#;

    #[test]
    fn builds_requests_as_stock_sent_them() {
        let c = connector("http", OPENAI, "");
        let parts = c.split("predict", docs(&["hello world", "b"])).unwrap();
        assert_eq!(parts.len(), 1);
        let r = call(&c, "TEXT_EMBEDDING", &parts[0], &[]).unwrap();
        assert_eq!(
            body(&r),
            r#"{ "input": ["hello world","b"], "model": "m" }"#
        );
        assert_eq!(r.request.url, "http://localhost:8791/v1/openai/embeddings");
        assert_eq!(header(&r, "authorization"), Some("Bearer sk-secret"));
        assert_eq!(header(&r, "content-length"), Some("46"));
        assert_eq!(header(&r, "content-type"), Some("application/json"));
        let r = call(&c, "REMOTE", &remote(&[("input", "[\"hello\"]")]), &[]).unwrap();
        assert_eq!(body(&r), r#"{ "input": ["hello"], "model": "m" }"#);

        let bedrock = r#"{"action_type":"predict","method":"POST","url":"http://${parameters.endpoint}/bedrock/model/amazon.titan-embed-text-v1/invoke",
            "headers":{"content-type":"application/json","x-amz-content-sha256":"required"},
            "request_body":"{ \"inputText\": \"${parameters.inputText}\" }",
            "pre_process_function":"connector.pre_process.bedrock.embedding","post_process_function":"connector.post_process.bedrock.embedding"}"#;
        let c = connector(
            "aws_sigv4",
            bedrock,
            r#","region":"us-east-1","service_name":"bedrock""#,
        );
        let parts = c
            .split("predict", docs(&["hello \"q\" world", "b"]))
            .unwrap();
        assert_eq!(parts.len(), 2);
        let r = call(&c, "TEXT_EMBEDDING", &parts[0], &[]).unwrap();
        assert_eq!(body(&r), r#"{ "inputText": "hello \"q\" world" }"#);
        assert!(header(&r, "authorization").unwrap().ends_with(
            "Signature=3436fa765d1e2479a91a756ad4c36a4665684a86d88a8ab4fda8603e556444df"
        ));
    }

    const GENERIC: &str = r#"{"action_type":"predict","method":"POST","url":"http://${parameters.endpoint}/gen/${parameters.model}",
        "request_body":"{ \"q\": \"${parameters.q}\", \"n\": ${parameters.n:-null}, \"d\": \"${parameters.d:-dflt}\" }"}"#;

    #[test]
    fn fills_remote_parameters() {
        let c = connector("http", GENERIC, "");
        let r = call(&c, "REMOTE", &remote(&[("q", "a \"b\"\nc/é")]), &[]).unwrap();
        assert_eq!(
            body(&r),
            r#"{ "q": "a \"b\"\nc\/\u00E9", "n": null, "d": "dflt" }"#
        );
        assert_eq!(r.request.url, "http://localhost:8791/gen/m");
        let e = call(&c, "REMOTE", &remote(&[]), &[])
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("Some parameter placeholder not filled in payload: q"),
            "{e}"
        );
        let skip = [(
            "skip_validating_missing_parameters".to_string(),
            Some("true".to_string()),
        )];
        assert!(body(&call(&c, "REMOTE", &remote(&[]), &skip).unwrap()).contains("${parameters.q}"));
        let r = call(
            &c,
            "REMOTE",
            &remote(&[("q", "x\"y"), ("no_escape_params", "q")]),
            &[],
        )
        .unwrap_err();
        assert!(r.to_string().contains("Invalid payload"), "{r}");
        assert!(c
            .prepare("nope", "REMOTE", &remote(&[]), &[], None, "")
            .unwrap_err()
            .to_string()
            .contains("no nope action found"));
    }

    #[test]
    fn reads_responses() {
        let c = connector("http", OPENAI, "");
        let r = call(&c, "REMOTE", &remote(&[("input", "[\"a\"]")]), &[]).unwrap();
        let body = r#"{"data":[{"embedding":[0.1,0.25,-5e-5]},{"embedding":[1,2]}]}"#;
        let Parsed::Tensors(t) = c.parse("predict", &r, body, None).unwrap() else {
            panic!()
        };
        assert_eq!(t.len(), 2);
        assert_eq!(
            t[0],
            Tensor::Numbers {
                name: "sentence_embedding".into(),
                data_type: "FLOAT32".into(),
                shape: vec![3],
                data: ["0.1", "0.25", "-5e-5"]
                    .iter()
                    .map(|n| Json::Number(n.to_string()))
                    .collect()
            }
        );
        assert!(c
            .parse("predict", &r, r#"{"data":[{"embedding":["x"]}]}"#, None)
            .is_err());

        let c = connector("http", GENERIC, "");
        let r = call(&c, "REMOTE", &remote(&[("q", "x")]), &[]).unwrap();
        let body = r#"{"echo":"e","nested":{"vals":[1,2.5,"s"]}}"#;
        let Parsed::Tensors(t) = c.parse("predict", &r, body, None).unwrap() else {
            panic!()
        };
        assert_eq!(t, vec![Tensor::Map(Json::parse(body).unwrap())]);
        let filtered = [
            ("q".to_string(), Some("x".to_string())),
            (
                "response_filter".to_string(),
                Some("$.nested.vals".to_string()),
            ),
        ];
        let r = call(&c, "REMOTE", &Input::Remote(filtered.to_vec()), &[]).unwrap();
        let Parsed::Tensors(t) = c.parse("predict", &r, body, None).unwrap() else {
            panic!()
        };
        assert_eq!(
            t[0],
            Tensor::Map(Json::parse(r#"{"response":[1,2.5,"s"]}"#).unwrap())
        );
        let missing = [
            ("q".to_string(), Some("x".to_string())),
            ("response_filter".to_string(), Some("$.nope".to_string())),
        ];
        let r = call(&c, "REMOTE", &Input::Remote(missing.to_vec()), &[]).unwrap();
        let Parsed::Tensors(t) = c.parse("predict", &r, body, None).unwrap() else {
            panic!()
        };
        assert_eq!(t[0], Tensor::Map(Json::parse(body).unwrap()));
        let Parsed::Tensors(t) = c.parse("predict", &r, "[1]", None).unwrap() else {
            panic!()
        };
        assert_eq!(
            t[0],
            Tensor::Map(Json::parse(r#"{"response":[1]}"#).unwrap())
        );
    }

    #[test]
    fn hands_scripts_to_the_caller() {
        let painless = r#"{"action_type":"predict","method":"POST","url":"http://h/pl","request_body":"{\"q\": ${parameters.q}}",
            "pre_process_function":"return params.text_docs[0];","post_process_function":"return escape(params.x);"}"#;
        let c = connector("http", painless, "");
        let input = docs(&["h\"i"]);
        let Prepared::Script(s) = c
            .prepare("predict", "TEXT_EMBEDDING", &input, &[], None, "")
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(
            s.params,
            r#"{"algorithm":"TEXT_EMBEDDING","text_docs":["h\\\"i"]}"#
        );
        assert_eq!(s.source, "return params.text_docs[0];");
        let quoted = r#"{"parameters":{"q":"\"hi\"","n":1}}"#;
        let e = c
            .prepare("predict", "TEXT_EMBEDDING", &input, &[], Some(quoted), "")
            .unwrap_err()
            .to_string();
        assert!(e.contains(r#"Invalid payload: {"q": \"hi\"}"#), "{e}");
        let out = r#"{"parameters":{"q":"hi","n":1}}"#;
        let Prepared::Call(r) = c
            .prepare("predict", "TEXT_EMBEDDING", &input, &[], Some(out), "")
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(body(&r), r#"{"q": hi}"#);
        assert_eq!(r.parameters.get("n"), Some(&Some("1.0".to_string())));
        let Parsed::Script(s) = c.parse("predict", &r, "[1]", None).unwrap() else {
            panic!()
        };
        assert!(s.source.starts_with("\n    String escape(def input)"));
        assert_eq!(s.params, r#"{"result":[1]}"#);
        let tensors = r#"[{"name":"sentence_embedding","data":[1]}]"#;
        let Parsed::Tensors(t) = c.parse("predict", &r, "[1]", Some(tensors)).unwrap() else {
            panic!()
        };
        assert!(matches!(&t[0], Tensor::Raw(_)));
        let Parsed::Tensors(t) = c.parse("predict", &r, "[1]", Some("plain")).unwrap() else {
            panic!()
        };
        assert_eq!(
            t[0],
            Tensor::Map(Json::parse(r#"{"response":"plain"}"#).unwrap())
        );
        assert!(c.parse("predict", &r, "x", None).is_err());
    }

    #[test]
    fn splits_and_reranks() {
        let rerank = r#"{"action_type":"predict","method":"POST","url":"http://h/r",
            "request_body":"{\"query\":\"${parameters.query}\",\"documents\":${parameters.documents},\"top_n\":${parameters.top_n}}",
            "pre_process_function":"connector.pre_process.cohere.rerank","post_process_function":"connector.post_process.cohere.rerank"}"#;
        let c = connector("http", rerank, r#","input_docs_processed_step_size":"2""#);
        assert_eq!(c.split("predict", docs(&["a", "b", "c"])).unwrap().len(), 2);
        let input = Input::TextSimilarity {
            query: "q".into(),
            docs: vec!["a".into(), "b".into()],
        };
        let r = call(&c, "TEXT_SIMILARITY", &input, &[]).unwrap();
        assert_eq!(body(&r), r#"{"query":"q","documents":["a","b"],"top_n":2}"#);
        let resp =
            r#"{"results":[{"index":1,"relevance_score":0.5},{"index":0,"relevance_score":0.25}]}"#;
        let Parsed::Tensors(t) = c.parse("predict", &r, resp, None).unwrap() else {
            panic!()
        };
        assert!(
            matches!(&t[0], Tensor::Numbers { name, data, .. } if name == "similarity" && data[0] == Json::Number("0.25".into()))
        );
        let c = connector("http", rerank, r#","input_docs_processed_step_size":"0""#);
        assert!(c.split("predict", docs(&["a"])).is_err());
    }

    #[test]
    fn checks_headers_and_escapes() {
        let bad = r#"{"action_type":"predict","method":"POST","url":"http://h/x","headers":{"Authorization":"${parameters.k}"},"request_body":"{}"}"#;
        let c = connector("http", bad, "");
        assert!(call(&c, "REMOTE", &remote(&[("k", "v")]), &[])
            .unwrap_err()
            .to_string()
            .contains("security reasons"));
        let unresolved = r#"{"action_type":"predict","method":"POST","url":"http://h/x","headers":{"X-T":"${parameters.k}"},"request_body":"{}"}"#;
        let c = connector("http", unresolved, "");
        assert!(call(&c, "REMOTE", &remote(&[]), &[])
            .unwrap_err()
            .to_string()
            .contains("unresolved placeholder"));
        assert_eq!(
            header(
                &call(&c, "REMOTE", &remote(&[("k", "v")]), &[]).unwrap(),
                "x-t"
            ),
            Some("v")
        );
        assert_eq!(
            header(
                &call(&c, "REMOTE", &remote(&[("k", "a\nb")]), &[]).unwrap(),
                "x-t"
            ),
            Some("a\\nb")
        );
        let raw = [("k".to_string(), Some("a\nb".to_string()))];
        assert!(call(&c, "REMOTE", &remote(&[]), &raw)
            .unwrap_err()
            .to_string()
            .contains("control character"));
        assert!(
            calls_escape("x = escape(params.a)")
                && !calls_escape("String escape(def a) {}")
                && !calls_escape("unescape(x)")
        );
        assert!(
            defines_escape("String escape(def input) { return input; }")
                && !defines_escape("escape(x)")
        );
        assert_eq!(
            add_default_escape("String escape(def i) {} escape(x)"),
            "String escape(def i) {} escape(x)"
        );
        assert_eq!(
            null_default_keys(r#"{"a":"${parameters.x:-null}","b":"${parameters.y}"}"#),
            vec!["x"]
        );
        let mut p: HashMap<String, Option<String>> = HashMap::new();
        p.insert("t".into(), Some("${parameters.v.toString()}".into()));
        p.insert("v".into(), Some("a\"b".into()));
        add_to_string_params(&mut p);
        assert_eq!(p["v.toString()"], Some("a\\\"b".into()));
        let mut fp = HashMap::new();
        fp.insert(
            "f".to_string(),
            Some("connector.post_process.openai.embedding".to_string()),
        );
        assert_eq!(
            fill_function(&fp, "${parameters.f}").unwrap(),
            "\"connector.post_process.openai.embedding\""
        );
        assert!(validate_headers(&[("k".into(), "v".repeat(9000))]).is_err());
    }
}
