//! The `lucene-tools` binary is [`lucene_tools::run`] over the process's
//! arguments and streams, with its result as the exit code.

use std::process::Command;

fn lucene_tools(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_lucene-tools"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn no_tool_lists_the_tools_and_exits_2() {
    let o = lucene_tools(&[]);
    assert_eq!(o.status.code(), Some(2));
    assert!(o.stdout.is_empty());
    assert!(String::from_utf8_lossy(&o.stderr).contains("  GetTermInfo\n"));
}

#[test]
fn a_tool_runs_with_its_arguments() {
    let o = lucene_tools(&["GetTermInfo", "only-one"]);
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&o.stdout),
        format!("{}\n", lucene_tools::misc::get_term_info::USAGE)
    );
}
