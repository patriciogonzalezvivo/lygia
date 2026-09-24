#!/usr/bin/env bash
# Compiles every *.msl file in LYGIA as a standalone translation unit
# using the Metal compiler. Requires macOS with the Metal toolchain.
#
# usage: test/msl/compile.sh [file.msl ...]

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STD="${METAL_STD:-metal3.1}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cd "$ROOT"

if [ "$#" -gt 0 ]; then
    printf '%s\n' "$@" | sed 's|^\./||' > "$TMP/list"
else
    find . -name '*.msl' -not -path './.git/*' -not -path './node_modules/*' | sed 's|^\./||' | sort > "$TMP/list"
fi

compile_one() {
    f="$1"
    id="$(echo "$f" | tr '/.' '__')"
    printf '#include <metal_stdlib>\nusing namespace metal;\n#include "%s/%s"\n' "$ROOT" "$f" > "$TMP/$id.metal"
    # print each result in a single write so parallel jobs don't interleave
    if xcrun -sdk macosx metal -std="$STD" -c "$TMP/$id.metal" -o "$TMP/$id.air" 2> "$TMP/$id.err"; then
        printf 'OK   %s\n' "$f"
    else
        printf 'FAIL %s\n%s\n' "$f" "$(grep -m3 'error:' "$TMP/$id.err" | sed "s|$ROOT/||g; s|^|     |")"
    fi
}
export -f compile_one
export ROOT STD TMP

xargs -P "$JOBS" -I{} bash -c 'compile_one "$@"' _ {} < "$TMP/list" > "$TMP/results"

awk '/^FAIL/{p=1; print; next} /^OK/{p=0} p' "$TMP/results"
total=$(grep -c '' "$TMP/list")
failed=$(grep -c '^FAIL' "$TMP/results")
echo "MSL: $((total - failed))/$total compiled"

# All files together in one translation unit, to catch clashes between modules
# (names that shadow Metal types, ...). Each file is included twice, to catch
# missing include guards.
{
    printf '#include <metal_stdlib>\nusing namespace metal;\n'
    sed "s|.*|#include \"$ROOT/&\"|" "$TMP/list"
    sed "s|.*|#include \"$ROOT/&\"|" "$TMP/list"
} > "$TMP/all.metal"
if xcrun -sdk macosx metal -std="$STD" -c "$TMP/all.metal" -o "$TMP/all.air" 2> "$TMP/all.err"; then
    echo "MSL: combined translation unit compiled"
    combined=0
else
    echo "FAIL combined translation unit"
    grep 'error:' "$TMP/all.err" | sed "s|$ROOT/||g; s|^|     |" | head -20
    combined=1
fi

[ "$failed" -eq 0 ] && [ "$combined" -eq 0 ]
