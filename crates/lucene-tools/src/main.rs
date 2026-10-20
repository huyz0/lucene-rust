#![forbid(unsafe_code)]
//! The `lucene-tools` binary: `lucene-tools <Tool> [args...]`, the shell
//! over [`lucene_tools::run`].

fn main() {
    let args: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    let code = lucene_tools::run(&args, &mut std::io::stdout().lock(), &mut std::io::stderr());
    std::process::exit(code);
}
