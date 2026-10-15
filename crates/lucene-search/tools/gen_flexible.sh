#!/usr/bin/env bash
# Regenerates the flexible query parser's translated JavaCC code
# (crates/lucene-search/src/queryparser/flexible/{token_manager,constants,
# lookahead}.rs) from Lucene 10.5.0's generated Java:
#
#   crates/lucene-search/tools/gen_flexible.sh LUCENE_CHECKOUT
#
# LUCENE_CHECKOUT is a Lucene 10.5.0 source tree (its lucene/queryparser
# holds the generated StandardSyntaxParser*.java). The output is formatted
# with cargo fmt; a clean `git diff` afterwards shows the files are exactly
# the tools' output.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
lucene="${1:?usage: gen_flexible.sh LUCENE_CHECKOUT}"
java="$lucene/lucene/queryparser/src/java/org/apache/lucene/queryparser/flexible/standard/parser"
tools=crates/lucene-search/tools
out=crates/lucene-search/src/queryparser/flexible
python3 "$tools/javacc_lexer.py" "$java/StandardSyntaxParserTokenManager.java" \
  "$tools/flexible_token_manager.header.rs" > "$out/token_manager.rs"
{ cat "$tools/flexible_constants.header.rs"; echo
  python3 "$tools/javacc_parser.py" "$java/StandardSyntaxParserConstants.java" "$java/StandardSyntaxParser.java" constants
} > "$out/constants.rs"
{ cat "$tools/flexible_lookahead.header.rs"; echo
  python3 "$tools/javacc_parser.py" "$java/StandardSyntaxParserConstants.java" "$java/StandardSyntaxParser.java" lookahead
} > "$out/lookahead.rs"
rustfmt --edition 2021 "$out/token_manager.rs" "$out/constants.rs" "$out/lookahead.rs"
