# Metal Support in Lygia

Metal support is currently highly experimental and very work in progress.

## Porting Progress

- [ ] Animation
- [x] Blend
- [ ] Color 
  - [x] Blend
  - [ ] Dither
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

- 1. dupe `*.glsl` files-> and rename them to  `*.msl`
- 2. find replace `.glsl` -> `.msl` and ensure you repeat the above for imports
- 3. find replace `vec2` -> `float2`
- 4. find replace `vec3` -> `float3`
- 5. find replace `vec4` -> `float4`
- 6. find replace `matN` -> `matrix<float, n, n>`
- 7. find replace `in ` function argument keyword -> `` as metal doesn't have the in function keyword
- 8. find `inout` and determine which thread local memory keyword should replace it, and make it a reference
- 9. ensure `const` is only used within functions, `constant` must be used for global scoped constants

## Things to look out for

- Metal does not have the same basic math functions signatures as GLSL. We have a `math_compat.msl` import you can use which has defines that should help.
- Texture precision and filtering.
    - Added `SAMPLER_TYPE` which specifies the texture precisions. Defaults to `texture2d<float>`
    - Added `SAMPLER` which specifies the Metal sampler object. Defaults to `sampler( min_filter::linear, mag_filter::linear )
    
## Things not yet done

- `gl_FragCoord` compatibilty. Not sure if there is a nice way to make this work without end users annotating their root Metal shader entry point.
    
