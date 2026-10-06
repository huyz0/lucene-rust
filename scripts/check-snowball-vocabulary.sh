#!/usr/bin/env bash
# Opt-in differential: every Snowball stemmer over the Snowball project's own
# test vocabularies (github.com/snowballstem/snowball-data), against the stems
# its reference implementation gives (`output.txt`).
#
#   scripts/check-snowball-vocabulary.sh [SNOWBALL_DATA_CHECKOUT]
#
# Lucene 10.5.0's gradle/generation/snowball.gradle pins the Snowball compiler
# (34f3612e, which crates/lucene-analysis/tools/gen_snowball.sh also uses) but
# no data commit. DATA_COMMIT is the last snowball-data commit before that
# compiler commit; there Lucene 10.5.0's 30 stemmers reproduce output.txt for
# every word (checked with Lucene's own classes when this was pinned), so the
# data is Lucene's ground truth as well as Snowball's.
#
# The 60 files (voc.txt and output.txt per language, Arabic's gzipped) are
# downloaded from that commit, or copied from a checkout of it, and each is
# checked against its pinned SHA-256: a moved file is a failure, not a new
# baseline. They are never committed here -- Greek's and French's
# vocabularies are partly CC BY-SA, Arabic's GPL-3.0 (docs/licences.md) --
# only unpacked into a temporary directory, which
# crates/lucene-analysis/tests/snowball_vocabulary.rs reads through
# SNOWBALL_DATA (without it, that test is skipped).
# Needs curl (without a checkout), gzip, sha256sum and cargo.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_COMMIT=f08c4d63b9bc223aa7a1e706f4e8db8dd4ced28a
BASE="https://raw.githubusercontent.com/snowballstem/snowball-data/${DATA_COMMIT}"
# path  sha256
FILES=(
    "arabic/voc.txt.gz c42521f14a557a63fd8457332c49668ee7c5cfde3028097be512e12c77cbd838"
    "arabic/output.txt.gz bd248a66578894fa3ccf2055e84d4fa8a0717dc6443bea0dbd10df9b5ac88c79"
    "armenian/voc.txt 28a0e05bdf2410a46f5178bafbbfcbe05d834c049e2dbdbe15fd098ccb335a8c"
    "armenian/output.txt 570c4bea5d0880fea7e15980a5555c22cef79b53650b3b2a6549f2ff793d444e"
    "basque/voc.txt 077d807b70b28eea1a011e41c9232cee19f3d84bd44d82d14785fd89a2b0d74e"
    "basque/output.txt 1cabfbf6bc606ec72cfb9ca8e61d993e2700242e6490776e58247fff5125ea12"
    "catalan/voc.txt 31b2ea26352e2bb68f4069e2c84dc4b5abfe4bd3857f4e45eb8244ae3a9732f1"
    "catalan/output.txt 0598b1929582f9778231245c0ad4bfa8cfb2d3f29ac050f8f0403f68a76b41e8"
    "danish/voc.txt 2d0c4501461c7a6fe33fb427e30d5fc3f4d2e0a6b0db0353bd7ce6e075efff3e"
    "danish/output.txt 8fe1322f1642efba6db13fcdfb6263efea2c82bbe71e8fc16c294be16f8364e2"
    "dutch/voc.txt 8f1b8ce71af34f5caec1641ab93c5acc65373d8a481105443380ec582058d3d3"
    "dutch/output.txt c5f4e5b542a0308f2e843c3f7831e395d906921bc267b1abd9fc3d0861377e7c"
    "english/voc.txt fe09ef577f1c8e4872b79a4f6995e62d782bbdbcda03f65a5ddd3e07b8b0044f"
    "english/output.txt 7319d67e73f8a6d2abe5fdc041aa803f84aa2a08ac4642ada4c1a4315fec8c72"
    "estonian/voc.txt 35ab26014f08e9c3297e5e923a4991c38a9baaf78465b39f214964970625818d"
    "estonian/output.txt d74ce38c3a868a1b7019986952ac80086b7f5047616d561978b252ded838026e"
    "finnish/voc.txt 5c5bd31e81b5708ab608266a0314bdbf9ccad50ae94bd0844b8d590deaf2f182"
    "finnish/output.txt b459f1b9d261902220c1e43b8e1a0b86abc181afddff200c8ddb7272fb1d5251"
    "french/voc.txt 4ad03e5b7e632b997fc69f0cd140e48ffdbcfdb5f5958403dcd9c274355b2b10"
    "french/output.txt feb1c29b7e166c6f9d9ae56ab1f745513d7ffc67ce48ec253e0aecfaa6b2a571"
    "german/voc.txt f680b1d49eb56bd6145a0ca03484fa489f0c79073ec406e80fef71d924c475af"
    "german/output.txt 38e67986ca844667ed7db8fb7862a71a47128ab3725d54e6c230a5fad916d852"
    "greek/voc.txt 8364221e612ca78c25d18ffe8df516fa839ba57c2e14bffe41d87ddfa6431d90"
    "greek/output.txt 7e35bdd1588752b5579e84e13cd522433fb3e378cb6bb5c3b1e16bd344edd394"
    "hindi/voc.txt 3afec633b8844b38161bdf20efcc291349ab3001f40f1c10604fcd12d96b66d3"
    "hindi/output.txt 99eea4297d4705f92368a1ba902e17dea4bc8e22618d6aa0a6058711c6166aab"
    "hungarian/voc.txt 9f2b472d4a6713edb034d27186e3511f84aaa3ca91d2cd829acdfcc92731be9a"
    "hungarian/output.txt 5586e92b4667a8cb276e93c1bdfd7422e97dea3c065de18341fbfa778d77f226"
    "indonesian/voc.txt 0ba355326683d708c17ad65b7f89a6fb55ac437a2cdd89f9b1f6e663b7bd826e"
    "indonesian/output.txt 3a21e503760a42005f2b3ebfd9162a1bc25313124b6b45bf96e4ea3378b6f110"
    "irish/voc.txt d8873abed89e598fb870bc08d87389da943febeb1297bf2432ddbb1072549938"
    "irish/output.txt 603029cf240e3950027af6e8751a14da895514931cd31d2204bc26c956ea4eac"
    "italian/voc.txt e1975e6793938bee0bc7393e99f819d24c0e58ff384ed63ada679d2c15b6ddd0"
    "italian/output.txt beeea24a82fec06bea1544443e4b2a6329022a88bf4c56e516c32238f394c17f"
    "lithuanian/voc.txt e61d4f728f8cd9392ceb524e39f105781523128a1fad358503584b2098e57a9c"
    "lithuanian/output.txt e915ec81dc55ae76b593dc4a6016f7e22a47f1a9dcb5e2be7ee8290913f1d41b"
    "nepali/voc.txt 8edb711669c57d57981b027237b299d9c4c856027c343e6942bc1874456d5994"
    "nepali/output.txt 74eb74acc22e32b37c8ec98585ffeec2d2d77476f6921e0106609804b0b09fee"
    "norwegian/voc.txt f16ab5534328d7f4a12bc24235ee6981a017157b2e34c9ab0edba495726bc7a1"
    "norwegian/output.txt b2db9db55734f905f6d7fba197ddb9c465fb85fddcd791fd9f1b5cb209ec190b"
    "porter/voc.txt fe09ef577f1c8e4872b79a4f6995e62d782bbdbcda03f65a5ddd3e07b8b0044f"
    "porter/output.txt 6b91a82e05fb70a829af2fcc757599d2db3eccf9f3255343199817efbb4618a1"
    "portuguese/voc.txt f207664c28500f53f7d8294c891e842aa99261f4f42da3ec38712d82ea99ad08"
    "portuguese/output.txt 7562ac2aaa88be13aa1871e0450399eee187936ea4ff1efe7a0ed4332c57010d"
    "romanian/voc.txt 88f55d0384c41e2837b733c410394d81b72eb6a5cf60feb6fe4db222b51e35c2"
    "romanian/output.txt c8f5cdca11747db407ed09ac14adb0c0189092419be8d6256f3cee1cdcb27a8e"
    "russian/voc.txt 6e4cd2ed5c908fe3d4cabaa74a088d9043cb626417c1eadc2c157cafbf9772a0"
    "russian/output.txt c6265f9071e41b24590d567ea69181f5781790930907c77b28044f1d71b1bab3"
    "serbian/voc.txt d4161835e7d32ce8dcca74a499ebc6a321874a55da2e640fb01e886897c95270"
    "serbian/output.txt 5f913f42367cffabe27e8443b5b3b9d7b16420b5bf0db9bd12375f8dab8ba274"
    "spanish/voc.txt c2ea573edbe1fc5afe62dfc4c3986ed0e66c250e8e5d3887f923e20455709f58"
    "spanish/output.txt ec801d7bc2e78caa5a4204708e32e081db307e0a58c3655d6f849e31ba8a432d"
    "swedish/voc.txt c5fc4d9c6c599a9c74d632591075c6c8fdd10b10ddb79ee2c854110c5175b176"
    "swedish/output.txt fc0cbbd7c85b4c33c20d52ffea133ed34fe470d0fd8a9f12fa934542a9f3eeca"
    "tamil/voc.txt 3bfbd5898c849bdba3c60147555239dfcf7227050a11f6a5ddcde1414d2667e3"
    "tamil/output.txt ea215a4098148e9fbec47a58193cd13f5285d243e721c56bae06a6af9b274190"
    "turkish/voc.txt be3174a99eb68b5297a2bb3817944e09cfee924c9b4ab44f3c0026814a35777d"
    "turkish/output.txt 961b842d41ea12cfd4a82c3c49360b2f3d54bdaeb6e2f1a81cd31e8f5f09fc2f"
    "yiddish/voc.txt a837cd51b8a72cb9e8450b9caff1d4d3f7a316a9f7c7c00db46a8f328397f682"
    "yiddish/output.txt aeeccd89df46c5883d460a96fd25303e607d8a270f17bc4add8183147af77f9d"
)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for entry in "${FILES[@]}"; do
    read -r path sum <<<"$entry"
    mkdir -p "$work/$(dirname "$path")"
    if [[ $# -ge 1 ]]; then
        cp "$1/$path" "$work/$path"
    else
        curl -sSfL --retry 4 --retry-delay 2 -o "$work/$path" "$BASE/$path"
    fi
    echo "$sum  $work/$path" | sha256sum -c --quiet - || {
        echo "check-snowball-vocabulary: $path does not match its pinned SHA-256" >&2
        exit 1
    }
    if [[ $path == *.gz ]]; then
        gzip -d "$work/$path"
    fi
done
echo "check-snowball-vocabulary: ${#FILES[@]} files of snowball-data ${DATA_COMMIT:0:8} match their pins"
SNOWBALL_DATA="$work" cargo test --release -p lucene-analysis --test snowball_vocabulary -- --nocapture
