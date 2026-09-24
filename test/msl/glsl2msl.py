#!/usr/bin/env python3
"""
First-pass GLSL -> MSL converter for LYGIA files, following the porting
methodology in README_METAL.md. The output still needs review, and must be
verified with test/msl/compile.sh.

usage: test/msl/glsl2msl.py path/to/file.glsl [...]   (writes path/to/file.msl)
       add --force to overwrite existing .msl files
"""

import re
import sys
from pathlib import Path

# lines that are documentation (inside the /* */ header) are left mostly untouched
TYPES = [
    (r'\bmat([234])\b', r'float\1x\1'),
    (r'\bvec([234])\b', r'float\1'),
    (r'\bivec([234])\b', r'int\1'),
    (r'\buvec([234])\b', r'uint\1'),
    (r'\bbvec([234])\b', r'bool\1'),
]

CODE = [
    # parameter qualifiers: `in` is the default, `out`/`inout` become thread references
    (r'([(,]\s*)(?:const\s+)?in\s+(?=\w)', r'\1'),
    (r'([(,]\s*)(?:inout|out)\s+([\w<>]+)\s+(\w+)', r'\1thread \2& \3'),
    # const on a by-value parameter is harmless but noisy
    (r'([(,]\s*)const\s+(?=\w)', r'\1'),
    # two-argument atan is atan2 in Metal
    (r'\batan\(([^(),]+),\s*([^(),]+)\)', r'atan2(\1, \2)'),
    # scalar-splat constructors like vec3(0.0) are valid in MSL, nothing to do
]


def convert(src: str) -> str:
    out = []
    in_comment = False
    for line in src.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith('/*'):
            in_comment = True
        if stripped.startswith('#include'):
            line = re.sub(r'\.glsl"', '.msl"', line)
        for pat, rep in TYPES:
            line = re.sub(pat, rep, line)
        if not in_comment:
            for pat, rep in CODE:
                line = re.sub(pat, rep, line)
            # program-scope constants must live in the constant address space
            line = re.sub(r'^const\s+', 'constant ', line)
        if '*/' in stripped:
            in_comment = False
        out.append(line)
    return ''.join(out)


def main(argv):
    force = '--force' in argv
    files = [a for a in argv if a != '--force']
    if not files:
        print(__doc__)
        return 1
    for f in files:
        src = Path(f)
        dst = src.with_suffix('.msl')
        if dst.exists() and not force:
            print(f'skip {dst} (exists)')
            continue
        dst.write_text(convert(src.read_text()))
        print(f'wrote {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
