# METAL version

Metal support is currently highly experimental and very work in progress.

Every `*.msl` file is compiled by CI, both on its own and all together in a single translation unit. Run the same check locally on macOS with:

```sh
test/msl/compile.sh               # every *.msl file
test/msl/compile.sh sdf/*.msl     # just some
```

LYGIA files don't include the Metal standard library themselves, so include it before any of them:

```cpp
#include <metal_stdlib>
using namespace metal;

#include "lygia/sdf/starSDF.msl"
#include "lygia/space/kaleidoscope.msl"
```

## Porting Progress

- [x] Animation
- [x] Blend
- [x] Color
  - [x] Blend
  - [x] Dither (not fully vetted / just spot checked)
  - [x] Palette
  - [x] Levels
  - [x] Space
  - [x] Tonemap
- [x] Distort
- [x] Draw - (not fully vetted / just spot checked)
- [x] Filters
- [x] Generative (not fully vetted / just spot checked)
- [x] Geometry
- [ ] Lighting (only `lighting/ray`)
- [x] Math - (not fully vetted / just spot checked)
- [x] Morphological
- [ ] Sample (clamp2edge, nearest and sprite done)
- [x] Sampler
- [x] SDF - (not fully vetted / just spot checked)
- [x] Simulate
- [x] Space

## Porting Methodology

`test/msl/glsl2msl.py path/to/file.glsl` does a first pass of the steps below. Always review its output and run `test/msl/compile.sh` on it.

- dupe `*.glsl` files -> and rename them to `*.msl`
- find replace `.glsl` -> `.msl` and ensure you repeat the above for imports
- find replace `vec2` -> `float2`
- find replace `vec3` -> `float3`
- find replace `vec4` -> `float4`
- find replace `matN` -> `floatNxN`. Avoid spelling it `matrix<float, N, N>`: `draw/matrix.msl` defines a `matrix()` function, which makes that spelling ambiguous
- find replace two-argument `atan(y, x)` -> `atan2(y, x)`
- find replace `in ` function argument keyword -> `` as metal doesn't have the in function keyword
- find `inout` and determine which thread local memory keyword should replace it, and make it a reference
- ensure `const` is only used within functions, `constant` must be used for global scoped constants
- make sure every `#ifndef FNC_*` include guard is followed by its `#define`
- rename anything that collides with a Metal reserved word or built-in: the `char()` function is `drawChar()`, and local variables named `kernel` are `kern`. Don't redefine functions Metal already has, like `atan2` or `transpose`
- `dFdx`/`dFdy`/`fwidth` are `dfdx`/`dfdy`/`fwidth`, and `discard` is `discard_fragment()`. These only work in fragment functions, so say so in the description of files that use them

## Things to look out for

- Metal does not have the same basic math functions signatures as GLSL. We are adding all the polyfill functions in the `math/` folder.
- Texture precision and filtering.
  - Added `SAMPLER_TYPE` which specifies the texture precisions. Defaults to `texture2d<float>`
  - This means your texture definition must match the default `float` precision, or you will need to override `SAMPLER_TYPE`
  - Added `SAMPLER` which specifies the Metal sampler object. Defaults to `sampler( min_filter::linear, mag_filter::linear )`

- Metal already has a native `atan2(y, x)`, equivalent to GLSL's `atan(y, x)`. LYGIA's GLSL `atan2` (with a 0 to TAU range) can't be redefined without making every call ambiguous, so `math/atan2.msl` only documents it.
- Metal has no global uniforms, textures or `gl_FragCoord`, and library functions can't see the entry point's arguments. So:
  - functions that default to `gl_FragCoord` in GLSL (e.g. dithering) only have overloads taking the coordinate explicitly. Pass the `[[position]]` coords from your main shader.
  - functions that default to a uniform in GLSL (e.g. `u_projectionMatrix`) take it as an explicit argument. The overload without it only exists when you `#define` the option (e.g. `CAMERA_PROJECTION_MATRIX`) to something in scope. See `space/view2screenPosition.msl`.

## Things not yet done

- Porting the modules still missing above. Lighting and sample depend heavily on GLSL global uniforms and textures (`LIGHT_*`, `SCENE_*`), so they need the explicit-argument approach above rather than a straight translation.
