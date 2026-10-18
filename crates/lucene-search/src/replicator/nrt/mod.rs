//! `org.apache.lucene.replicator.nrt`: near-real-time replication of one
//! index from a primary node to replicas.
//!
//! The primary ([`PrimaryNode`]) turns each NRT point of its `IndexWriter`
//! into a [`CopyState`]: the files of the point with their header, footer,
//! length and checksum ([`FileMetaData`]), and the `segments_N` bytes naming
//! them. A replica ([`ReplicaNode`]) copies the files it lacks
//! ([`CopyJob`], [`CopyOneFile`]), keeps reference counts so a file is
//! deleted only once no point uses it ([`ReplicaFileDeleter`]), and opens
//! readers over the point ([`SegmentInfosSearcherManager`]).
//!
//! How the bytes travel is the transport's business: [`ReplicaHooks`] and
//! [`PrimaryHooks`] are the abstract methods, and [`local`] is an in-process
//! transport (the test framework's `Simple*` classes).

pub mod copy;
pub mod local;
pub mod node;
pub mod primary;
pub mod replica;
pub mod replica_file_deleter;
pub mod searcher_manager;
pub mod types;

pub use copy::{CopyJob, CopyJobCore, CopyOneFile, CopyStream, OnceDone, ReplicaCore};
pub use node::{
    bytes_to_string, node_message, node_message_id, set_global_start, NodeCore, Printer,
    PRIMARY_GEN_KEY, VERSION_KEY,
};
pub use primary::{PrimaryHooks, PrimaryNode, PrimaryShared};
pub use replica::{ReplicaHooks, ReplicaNode};
pub use replica_file_deleter::ReplicaFileDeleter;
pub use searcher_manager::SegmentInfosSearcherManager;
pub use types::{CopyState, FileMetaData, NodeCommunicationError};
