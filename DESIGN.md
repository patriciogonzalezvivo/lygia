
## Design Guidelines
        
* **Granularity**. One function per file. The file and the function share the same name, namely: `myFunc.glsl` contains `myFunct()`. There are some files that just include a collection of files inside a folder with the same name. For example:

```
    color/blend.glsl
    // which includes
    color/blend/*.glsl

```

* **Multi-language**. Right now, most of the code is GLSL (`*.glsl`) and HLSL (`*.hlsl`), but we are slowly extending to WGSL (`*.wgsl`), CUDA (`*.cuh`) and Metal (`*.msl`).

```
    math/mirror.glsl
    math/mirror.hlsl
    math/mirror.wgsl
    math/mirror.msl
    math/mirror.cuh
```

* **Self documented**. Each file contains a structured comment (in YAML) at the top of the file. This one contains the name of the original author, description, use, `#define` options, and any implementation-specific notes.

```glsl

    /*
    contributors: <FULL NAME>
    description: [DESCRIPTION + URL]
    use: <vec2> myFunc(<vec2> st, <float> x [, <float> y])
    notes:
        - The option MYFUNC_TYPE is not supported on WGSL.
    options:
        - MYFUNC_TYPE
        - MYFUNC_SAMPLER_FNC()
    */

```

* **Prevent name collisions** by using the following pattern where `FNC_` is followed with the function name:

```glsl

    #ifndef FNC_MYFUNC
    #define FNC_MYFUNC

    float myFunc(float in) {
        return in;
    }

    #endif

```

* **Templating capabilities through `#defines`**. Probably the most frequent use is templating the sampling function for reusability. The `#define` options start with the name of the function, in this example `MYFUNC_`. They are added as `options:` in the header.
 
```glsl

    #ifndef MYFUNC_TYPE
    #define MYFUNC_TYPE vec4
    #endif

    #ifndef MYFUNC_SAMPLER_FNC
    #define MYFUNC_SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
    #endif

    #ifndef FNC_MYFUNC
    #define FNC_MYFUNC
    MYFUNC_TYPE myFunc(SAMPLER_TYPE tex, vec2 st) {
        return MYFUNC_SAMPLER_FNC(tex, st);
    }
    #endif

```

* **Argument order**. Optional elements are at the end. When possible sort them according their memory footprint (except textures that remain at the top). Ex.: `SAMPLER_TYPE, mat4, mat3, mat2, vec4, vec3, vec2, float, ivec4, ivec3, ivec2, int, bool`

```glsl

    /*
    ...
    use: myFunc(<vec2> st, <vec2|float> x[, <float> y])
    */

    #ifndef FNC_MYFUNC
    #define FNC_MYFUNC
    vec2 myFunc(vec2 st, vec2 x) {
        return st * x;
    }

    vec2 myFunc(vec2 st, float x) {
        return st * x;
    }

    vec2 myFunc(vec2 st, float x, float y) {
        return st * vec2(x, y);
    }
    #endif

```

### WGSL Specifics

WGSL as a language has some fundamental differences from GLSL, HLSL and METAL. Here are some guidelines to help with the transition:

* **WGSL Function Renaming**. WGSL [does not support function overloading](https://github.com/gpuweb/gpuweb/issues/876) and as such function names must be unique and should reflect the size of parameter, return types. See documented examples below.

```wgsl
    // When the parameter types and return type is consistent and scalar, no suffix is needed.
    fn random(p: f32) -> f32 { ... }

    /*
    When only the return type is consistent and scalar, the function name is suffixed based on the size of the parameter types, i.e.
      - vec2<T> -> 2
      - vec3<T> -> 3
      - vec4<T> -> 4
    */
    fn random2(p: vec2f) -> f32 { ... }
    fn random3(p: vec3f) -> f32 { ... }
    fn random3(p: vec4f) -> f32 { ... }

    /* 
    When both parameters and return types are inconsistent, function name has two suffixes:
      - first for return type size
      - second for parameter type size
    */
    fn random21(p: f32) -> vec2f { ... }
    fn random22(p: vec2f) -> vec2f { ... }
    fn random23(p: vec3f) -> vec2f { ... }
    fn random31(p: f32) -> vec3f { ... }
    fn random32(p: vec2f) -> vec3f { ... }
    fn random33(p_: vec3f) -> vec3f { ... }
    fn random41(p: f32) -> vec4f { ... }
    fn random42(p: vec2f) -> vec4f { ... }
    fn random43(p: vec3f) -> vec4f { ... }
    fn random44(p: vec4f) -> vec4f { ... }
```
