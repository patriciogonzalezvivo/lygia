
## Design Principles

1. It relies on `#include "path/to/file.*lsl"` which is defined by Khronos GLSL standard and requires a typical C-like pre-compiler MACRO which is easy to implement with just basic string operations to resolve dependencies. 

Here you can find some implementations on different languages:

  - C#:
  
    . [GLSLIncludes](https://github.com/seb776/GLSLIncludes) a small utility to add the include feature to glsl by [z0rg](https://github.com/seb776).

  - C++:

    . [VERA's routines](https://github.com/patriciogonzalezvivo/vera/blob/main/src/ops/fs.cpp#L110-L171) for resolving GLSL dependencies.

  - Python:

    . [Small and simple routing to resolve includes](https://gist.github.com/patriciogonzalezvivo/9a50569c2ef9b08058706443a39d838e)

  - JavaScript: 
  
    . [vanilla JS (online resolver)](https://lygia.xyz/resolve.js) This small file brings `resolveLygia()` which takes a `string` or `string[]` and parses it, solving all the `#include` dependencies into a single `string` you can load on your shaders. It also has a `resolveLygiaAsync()` version that resolves all the dependencies in parallel. Both support dependencies to previous versions of LYGIA by using this pattern `lygia/vX.X.X/...` on you dependency paths. 
  
    . [npm module (online resolver)](https://www.npmjs.com/package/resolve-lygia) by Eduardo Fossas. This is bring the same `resolveLygia()` and `resolveLygiaAsync()` function but as a npm module.

    . [vite glsl plugin (local bundle)](https://github.com/UstymUkhman/vite-plugin-glsl) by Ustym Ukhman. Imports `.glsl` local dependencies, or load inline shaders through vite.
  
    . [esbuild glsl plugin (local bundle)](https://github.com/ricardomatias/esbuild-plugin-glsl-include) by Ricardo Matias. Imports local `.glsl` dependencies through esbuild.

    . [webpack glsl plugin (local bundle)](https://github.com/grieve/webpack-glsl-loader) by Ryan Grieve that imports local `.glsl` dependencies through webpack.
        
* It's **very granular**. One function per file. The file and the function share the same name, namely: `myFunc.glsl` contains `myFunct()`. There are some files that just include a collection of files inside a folder with the same name. For example:

```
    color/blend.glsl
    // which includes
    color/blend/*.glsl

```

* It's **multi language**. Right now most of is GLSL (`*.glsl`) and HLSL (`*.hlsl`), but we are slowly extending to WGSL (`*.wgsl`), CUDA (`*.cuh`) and Metal (`*.msl`).

```
    math/mirror.glsl
    math/mirror.hlsl
    math/mirror.wgsl
    math/mirror.msl
    math/mirror.cuh
```

* **Self documented**. Each file contains a structured comment (in YAML) at the top of the file. This one contains the name of the original author, description, use, and `#define` options

```glsl

    /*
    contributors: <FULL NAME>
    description: [DESCRIPTION + URL]
    use: <vec2> myFunc(<vec2> st, <float> x [, <float> y])
    options:
        - MYFUNC_TYPE
        - MYFUNC_SAMPLER_FNC()
    */

```

* Prevents **name collisions** by using the following pattern where `FNC_` is followed with the function name:

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

* **Function Overloading**. Arguments are arranged in such a way that optional elements are at the end. When possible sort them according their memory size (except textures that remain at the top). Ex.: `SAMPLER_TYPE, mat4, mat3, mat2, vec4, vec3, vec2, float, ivec4, ivec3, ivec2, int, bool`

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
