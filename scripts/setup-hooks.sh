#!/usr/bin/env bash
# Point git at the repo-local hooks (.githooks/) instead of .git/hooks/.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
echo "hooksPath set to .githooks"
