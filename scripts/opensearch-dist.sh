#!/usr/bin/env bash
# Extracts lib/ (and the lang-painless module) of the pinned OpenSearch distribution, for
# opensearch-plugin/'s build to compile against. No Docker: the distribution is the release
# tarball (downloaded once into $NODE_HOME, ~/os-native by default, by nodectl.py).
#
# The plugin must match its node's exact version, and the tarball is the distribution the
# end-to-end tests run -- so compiling against its own jars means the build and the node cannot
# disagree, and the build needs no Maven repository.
#
# The tarball's bundled JDK 25 is also registered as a Gradle toolchain (the plugin needs JDK 25,
# Gradle 8.14 itself runs on 21) unless ~/.gradle/gradle.properties or JAVA_HOME_25_X64 already
# names one.
#
#   scripts/opensearch-dist.sh            # -> target/opensearch-dist/3.9.0/lib
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION="${OPENSEARCH_VERSION:-3.9.0}"
OUT="target/opensearch-dist/${VERSION}"

SPI="modules/lang-painless/opensearch-scripting-painless-spi-${VERSION}.jar"
PAINLESS="modules/lang-painless/lang-painless-${VERSION}.jar"
if [[ -f "$OUT/lib/opensearch-${VERSION}.jar" && -f "$OUT/$SPI" && -f "$OUT/$PAINLESS" ]]; then
    echo "opensearch-dist: $OUT/lib already present"
else
    src=$(python3 - <<PY
import sys
sys.path.insert(0, "opensearch-plugin/e2e")
import nodectl
print(nodectl.pristine("${VERSION}"))
PY
)
    rm -rf "$OUT"
    mkdir -p "$OUT/modules"
    cp -r "$src/lib" "$OUT/lib"
    # Painless (its SPI, which the k-NN allowlist extension compiles against, and the module itself,
    # whose front end the native Painless runtime takes its IR from).
    cp -r "$src/modules/lang-painless" "$OUT/modules/lang-painless"
    echo "opensearch-dist: extracted $(ls "$OUT/lib" | wc -l) jars to $OUT/lib"
fi

if [[ -z "${JAVA_HOME_25_X64:-}" && -z "${JAVA_HOME_25_ARM64:-}" ]] && ! grep -qs 'installations.paths' ~/.gradle/gradle.properties; then
    jdk=$(python3 -c "import sys; sys.path.insert(0,'opensearch-plugin/e2e'); import nodectl; print(nodectl.pristine('${VERSION}'))")/jdk
    mkdir -p ~/.gradle
    echo "org.gradle.java.installations.paths=$jdk" >> ~/.gradle/gradle.properties
    echo "opensearch-dist: registered $jdk as a Gradle toolchain"
fi
