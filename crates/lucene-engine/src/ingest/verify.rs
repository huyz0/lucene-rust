//! Compares two operation blobs as the documents they decode to, so a test can check source
//! ingest against the JVM's mapper path field for field.

use crate::status::Status;
use crate::writer_blob::{decode_op, Op};

fn entries(op: &[u8]) -> Result<Vec<[Vec<String>; 4]>, Status> {
    let Op::Add { docs, .. } = decode_op(op)? else {
        return Err(Status::InvalidArgument);
    };
    Ok(docs
        .iter()
        .map(|doc| {
            let mut sections = [
                doc.stored
                    .iter()
                    .map(|f| format!("{f:?}"))
                    .collect::<Vec<_>>(),
                doc.fields
                    .inverted
                    .iter()
                    .map(|f| format!("{f:?}"))
                    .collect(),
                doc.fields
                    .doc_values
                    .iter()
                    .map(|f| format!("{f:?}"))
                    .collect(),
                doc.fields.points.iter().map(|f| format!("{f:?}")).collect(),
            ];
            for s in &mut sections {
                s.sort();
            }
            sections
        })
        .collect())
}

/// `None` when both blobs decode to the same documents, else the first difference.
pub fn verify(expected: &[u8], actual: &[u8]) -> Result<Option<String>, Status> {
    let (e, a) = (entries(expected)?, entries(actual)?);
    if e.len() != a.len() {
        return Ok(Some(format!(
            "the mapper's operation has {} documents, the fast path's {}",
            e.len(),
            a.len()
        )));
    }
    for (n, (e, a)) in e.iter().zip(&a).enumerate() {
        for (i, name) in ["stored", "inverted", "doc values", "points"]
            .iter()
            .enumerate()
        {
            if e[i] != a[i] {
                let only_expected: Vec<_> = e[i].iter().filter(|x| !a[i].contains(x)).collect();
                let only_actual: Vec<_> = a[i].iter().filter(|x| !e[i].contains(x)).collect();
                return Ok(Some(format!(
                    "document {n}, {name}: only in the mapper's: {only_expected:?}; only in the fast path's: {only_actual:?}"
                )));
            }
        }
    }
    Ok(None)
}
