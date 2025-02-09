# METAL version

Metal support is currently highly experimental and very work in progress.

## Porting Progress

- [ ] Animation
- [x] Blend
- [ ] Color 
  - [x] Blend
  - [x] Dither (not fully vetted / just spot checked)
  - [ ] Palette 
  - [x] Levels
  - [x] Space
  - [ ] Tonemap
- [x] Distort
- [x] Draw - (not fully vetted / just spot checked)
- [ ] Filters
    - [x] Gaussian Blur
    - [x] Box 2d 
- [x] Generative (not fully vetted / just spot checked)
- [ ] Geometry
- [ ] Lighting 
- [x] Math - (not fully vetted / just spot checked)
- [ ] Morphological
- [ ] Sample
- [x] Sampler
- [x] SDF - (not fully vetted / just spot checked)
- [ ] Space

## Porting Methodology

- dupe `*.glsl` files -> and rename them to `*.msl`
- find replace `.glsl` -> `.msl` and ensure you repeat the above for imports
- find replace `vec2` -> `float2`
- find replace `vec3` -> `float3`
- find replace `vec4` -> `float4`
- find replace `matN` -> `matrix<float, n, n>`
- find replace `in ` function argument keyword -> `` as metal doesn't have the in function keyword
- find `inout` and determine which thread local memory keyword should replace it, and make it a reference
- ensure `const` is only used within functions, `constant` must be used for global scoped constants

## Things to look out for

- Metal does not have the same basic math functions signatures as GLSL. We are adding all the polyfill functions in the `math/` folder.
- Texture precision and filtering.
  - Added `SAMPLER_TYPE` which specifies the texture precisions. Defaults to `texture2d<float>`
  - This means your texture definition must match the default `float` precision, or you will need to override `SAMPLER_TYPE`
  - Added `SAMPLER` which specifies the Metal sampler object. Defaults to `sampler( min_filter::linear, mag_filter::linear )`

## Things not yet done

- `gl_FragCoord` compatibility. Not sure if there is a nice way to make this work without end users annotating their root Metal shader entry point.
  - For now, use the function definitions which pass the [[position]] coords from your main shader.
- `atan` / `atan2` compatibility. Need to see if there is a nice way to override the function signatures to match.
