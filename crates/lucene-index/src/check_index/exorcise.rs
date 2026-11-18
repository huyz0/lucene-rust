use super::*;

/// `CheckIndex.exorciseIndex(status)` (`-exorcise`): writes the next
/// `segments_N` holding only the segments `results` -- a
/// [`check_directory`] of `dir`'s latest commit -- found sound, so every
/// document of a broken segment is lost and the rest of the index opens.
/// Returns the new commit's file name, or `None` when no segment is broken
/// and nothing is written (Java's "No problems were detected").
///
/// As Java's: under the index's write lock (`CheckIndex` holds it for its
/// whole life), the commit is the latest one read again, its version bumped
/// (`changed()`), and its segment counter moved past every remaining name
/// when the check found it behind (`newSegments.counter = maxSegmentName +
/// 1`). The dropped segments' files are left for the next writer's deleter.
///
/// # Errors
/// `results` does not describe `dir`'s latest commit (a different
/// commit, or a subset -- Java's "can only exorcise an index that was fully
/// checked"), the lock is held, or the read or write fails.
pub fn exorcise(
    dir: &dyn Directory,
    results: &[CheckResult],
) -> segment_infos::Result<Option<String>> {
    let _lock = dir.obtain_lock("write.lock")?;
    let mut infos = segment_infos::read_latest(dir)?;
    let commit_name = lucene_store::directory::segments_file_name(infos.generation);
    let segment_results = results.get(1..).unwrap_or_default();
    let covers = results.first().map(|r| Some(&r.segment_name)) == Some(commit_name.as_ref())
        && segment_results.len() == infos.segments.len()
        && segment_results
            .iter()
            .zip(&infos.segments)
            .all(|(r, s)| r.segment_name == s.segment_name);
    if !covers {
        return Err(segment_infos::Error::Store(lucene_store::Error::Io(
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "can only exorcise an index that was fully checked (these results are not a check \
             of every segment of the latest commit)",
            ),
        )));
    }
    if segment_results.iter().all(CheckResult::all_passed) {
        return Ok(None);
    }
    let mut kept = Vec::with_capacity(infos.segments.len());
    for (result, segment) in segment_results
        .iter()
        .zip(std::mem::take(&mut infos.segments))
    {
        if result.all_passed() {
            kept.push(segment);
        }
    }
    infos.segments = kept;
    let max_segment_name = infos
        .segments
        .iter()
        .filter_map(|c| {
            c.segment_name
                .strip_prefix('_')
                .and_then(|n| i64::from_str_radix(n, 36).ok())
        })
        .max()
        .unwrap_or(-1);
    if infos.counter <= max_segment_name {
        infos.counter = max_segment_name.saturating_add(1);
    }
    infos.generation = infos.generation.saturating_add(1);
    infos.version = infos.version.saturating_add(1);
    infos.id = crate::index_writer::generate_segment_id(infos.generation);
    segment_infos::write(&infos, dir).map(Some)
}
