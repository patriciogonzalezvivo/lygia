![](https://cms.guggenheim-bilbao.eus/uploads/2020/02/01312.jpg)

# LYGIA: a multi-language shader library

Tired of searching for the same functions over and over? or to port and reimplementing them between platforms and shader languages? LYGIA is shader library of reusable functions that can be include easily on your projects. Doesn't matter the shader language, if they run local or on the cloud. LYGIA is very granular, designed for reusability, performance and flexibility. 

<p float="center">
    <a href="https://github.com/patriciogonzalezvivo/lygia_unity_examples"><img src="https://lygia.xyz/imgs/unity.png" width="100" /></a>
    <a href="https://github.com/patriciogonzalezvivo/lygia_threejs_examples"><img src="https://lygia.xyz/imgs/threejs.png" width="100" /></a>
    <a href="https://github.com/patriciogonzalezvivo/lygia_p5_examples"><img src="https://lygia.xyz/imgs/p5.png" width="100" /></a>
    <a href="https://editor.p5js.org/patriciogonzalezvivo/sketches"><img src="https://lygia.xyz/imgs/p5js.png" width="100" /></a>
</p>

## How does it work?

In your shader `#include` the functions you need:

```glsl

    uniform vec2    u_resolution;
    uniform float   u_time;

    #include "space/ratio.glsl"
    #include "math/decimation.glsl"
    #include "draw/circle.glsl"

    void main(void) {
        vec3 color = vec3(0.0);
        vec2 st = gl_FragCoord.xy/u_resolution.xy;
        st = ratio(st, u_resolution);
        
        color = vec3(st.x,st.y,abs(sin(u_time)));
        color = decimation(color, 20.);
        color += circle(st, .5, .1);
        
        gl_FragColor = vec4(color, 1.0);
    }
```

If you are working **locally** on an ecosystem that can resolve `#include` dependencies, just clone LYGIA in your project relative to the shader you are loading:

```bash
    git clone https://github.com/patriciogonzalezvivo/lygia.git
```

or as a submodule:

```bash
    git submodule add https://github.com/patriciogonzalezvivo/lygia.git
```

If you are working on a **cloud platform** probably you want to resolve the dependencies without needing to install anything. Just add a link to `https://lygia.xyz/resolve.js`: 

```html
<script src="https://lygia.xyz/resolve.js"></script>
```

To then resolve the dependencies by passing a `string` or `strings[]` to `resolveLygia()`:

```js
    vertSource = resolveLygia(vertSource);
    fragSource = resolveLygia(fragSource);
    shdr = createShader(vertSource, fragSource);
```

Learn more about how to use it from this **examples**:

* [2D examples for Processing (GLSL)](https://github.com/patriciogonzalezvivo/lygia_p5_examples)
* [2D/3D examples for P5.js (GLSL)](https://editor.p5js.org/patriciogonzalezvivo/sketches)
* [2D examples for Three.js (GLSL)](https://github.com/patriciogonzalezvivo/lygia_threejs_examples) 
* [2D/3D examples for Unity3D (HLSL)](https://github.com/patriciogonzalezvivo/lygia_unity_examples)
* [2D/3D examples on GlslViewer (GLSL)](https://github.com/patriciogonzalezvivo/lygia_examples)

For more information, guidance or feedback about using LYGIA, join [#Lygia channel on shader.zone discord](https://shader.zone/).


## How is it organized?

The functions are divided in different categories:

* `math/`: general math functions and constants like `PI`, `SqrtLength()`, etc. 
* `space/`: general spatial operations like `scale()`, `rotate()`, etc.
* `color/`: general color operations like `luma()`, `saturation()`, blend modes, palettes, color space conversion and tonemaps.
* `animation/`: animation operations, like easing
* `generative/`: generative functions like `random()`, `noise()`, etc. 
* `sdf/`: signed distance field functions.
* `draw/`: drawing functions like `digits()`, `stroke()`, `fill`, etc/.
* `sample/`: sample operations
* `filters/`: typical filter operations like different kind of blurs, mean and median filters.
* `distort/`: distort sampling operations
* `simulate/`: simulate sampling operations
* `lighting/`: different lighting models and functions for foward/deferred/raymarching rendering


## Flexible how?

There are some functions whose behaviour can be changed using the `#defines` keyword before including it. For example, [gaussian blurs](filter/gaussianBlur.glsl) are usually are done in two passes. By default, these are performed on their 1D version, but in the case you are interested on using a 2D kernel, all in the same pass, you will need to add the `GAUSSIANBLUR_2D` keyword this way:

```glsl

    #define GAUSSIANBLUR_2D
    #include "filter/gaussianBlur.glsl"

    void main(void) {
        ...
        
        vec2 pixel = 1./u_resolution;
        color = gaussianBlur(u_tex0, uv, pixel, 9);
        ...
    }

```

# Design Principles

This library:

* Relies on `#include "path/to/file.*lsl"` which is defined by Khronos GLSL standard and suported by most engines and enviroments ( like Unity3D, [OpenFrameworks](https://github.com/openframeworks/openFrameworks), [glslViewer](https://github.com/patriciogonzalezvivo/glslViewer/wiki/Compiling), [glsl-canvas VS Code pluging](https://marketplace.visualstudio.com/items?itemName=circledev.glsl-canvas), etc. ). It requires a tipical C-like pre-compiler MACRO which is easy to implement with just basic string operations to resolve dependencies. Here you can find some implementations on different languages:
    * C++:
        - [VERA's routines](https://github.com/patriciogonzalezvivo/vera/blob/main/src/ops/fs.cpp#L110-L171) for resolving GLSL dependencies.
    * Python
        - [Small and simple routing to resolve includes](https://gist.github.com/patriciogonzalezvivo/9a50569c2ef9b08058706443a39d838e)
    * JavaScript: 
        - [vanilla JS online resolver](https://lygia.xyz/resolve.js) This small file brings `resolveLygia()` which takes a `string` or `string[]` and parse it solving all the `#include` dependencies into a single `string` you can load on your shaders.
        - [vite glsl plugin](https://github.com/UstymUkhman/vite-plugin-glsl) by Ustym Ukhman. Imports `.glsl` local dependencies, or load inline shaders through vite
        - [esbuild glsl plugin](https://github.com/ricardomatias/esbuild-plugin-glsl-include) by Ricardo Matias. Import local `.glsl` dependencies through esbuild.
        - [webpack glsl plugin](https://github.com/grieve/webpack-glsl-loader) by Ryan Grieve. Import local `.glsl` dependencies through webpack.
        - [observable](https://observablehq.com/d/e4e8a96f64a6bf81) by Radamés Ajna. It's an series of examples on how to load LYGIA inside [Observable](https://observablehq.com).

* It's **very granular**. One function per file. The file and the function share the same name, namely: `myFunc.glsl` contains `myFunct()`. There are some files that just include a collection of files inside a folder with the same name. For example:

```
    color/blend.glsl
    // which includes
    color/blend/*.glsl

```

* It's **multi language**. Right now most of is GLSL (`*.glsl`) and HLSL (`*.hlsl`), but there are plans to extend it to Metal (`*.metal`).

```
    math/mirror.glsl
    math/mirror.hlsl
```

* **Self documented**. Each file contains a structured comment (in YAML) at the top of the file. This one contains the name of the original author, description, use and `#define` options

```glsl

    /*
    original_author: <FULL NAME>
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
    #define MYFUNC_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV)
    #endif

    #ifndef FNC_MYFUNC
    #define FNC_MYFUNC
    MYFUNC_TYPE myFunc(sampler2D tex, vec2 st) {
        return MYFUNC_SAMPLER_FNC(st);
    }
    #endif

```

* **Function Overloading**. Arguments are arranged in such a way that optional elements are at the back. When possible sort them according their memory size (except textures that reamin at the top). Ex.: `sampler2D, mat4, mat3, mat2, vec4, vec3, vec2, float, ivec4, ivec3, ivec2, int, bool`

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

# Contributions

LYGIA have a long way to go. Your support will be appreciated and rewarded. It can take many forms like: fixing bugs, expanding the crosscompatibility between GLSL/HLSL/Metal/SPIR-V, contributing new lygia functions or examples and integrations for new enviroments like TouchDesigner, GoDot, ISF, etc.

Another way to support it, is through [GitHub Sponsorships](https://github.com/sponsors/patriciogonzalezvivo). 

# Acknowledgements

This library has been built over years, and in many cases on top of the work of brillant, generous people like: [Inigo Quiles](https://www.iquilezles.org/), [Morgan McGuire](https://casual-effects.com/), [Hugh Kennedy](https://github.com/hughsk) and [Matt DesLauriers](https://www.mattdesl.com/).

# License 

LYGIA is dual-licensed under [the Prosperity License](https://prosperitylicense.com/versions/3.0.0) and the [Patron License](https://lygia.xyz/license) for [sponsors](https://github.com/sponsors/patriciogonzalezvivo) and contributors.

Sponsors and contributors shown in the [Patron License](https://lygia.xyz/license) can ignore any noncommercial or copyleft rules in all of [the Prosperity Licensed](https://prosperitylicense.com/versions/3.0.0) software. 
