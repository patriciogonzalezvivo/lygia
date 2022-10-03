![](https://artpil.com/wp-content/uploads/Lygia-Clark-banner.jpg)

# LYGIA: a multi-language shader library

Tired of reimplementing and searching for the same functions over and over, I started compiling and building a shader library of reusable assets (mostly functions) that can be include over and over. It's very granular, designed for reusability, performance and flexibility. 

Learn how to use it with this examples for:

* [Unity3D through HLSL](https://github.com/patriciogonzalezvivo/lygia_unity_examples)
* [In pure GLSL you can run in GlslViewer](https://github.com/patriciogonzalezvivo/lygia_examples)

Join [#Lygia channel on shader.zone discord](https://shader.zone/) to learn how to use it, share work or get help.

## How does it work?

1. Clone this repository in your project, where your shaders are.

```bash
git clone https://github.com/patriciogonzalezvivo/lygia.git
```

2. In your shader `#include` the functions you need:

```glsl
#ifdef GL_ES
precision mediump float;
#endif

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

## How is it organized?

The functions are divided in different categories:

* `math/`: general math functions and constants. 
* `space/`: general spatial operations like `scale()`, `rotate()`, etc.
* `color/`: general color operations like `luma()`, `saturation()`, blend modes, palettes, color space conversion and tonemaps.
* `animation/`: animation operations, like easing
* `generative/`: generative functions like `random()`, `noise()`, etc. 
* `sdf/`: signed distance field generation functions. Most of them ported from [PixelSpiritDeck](https://patriciogonzalezvivo.github.io/PixelSpiritDeck/)
* `draw/`: functions that draw shapes, numbers, lines, etc. Most of them ported from [PixelSpiritDeck](https://patriciogonzalezvivo.github.io/PixelSpiritDeck/)
* `sample/`: sample operations
* `filters/`: typical filter operations like different kind of blurs, mean and median filters.
* `distort/`: distort sampling operations
* `simulate/`: simulate sampling operations
* `lighting/`: different foward/deferred/raymarching lighting models and functions

## Flexible how?

There are some functions that are "templated" using `#defines`. You can change how it behaves by defining a keyword before including it. For examples, [gaussian blurs](filter/gaussianBlur.glsl) are usually done in two passes (and it defaults), but let's say you are in a rush you can specify to use 

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

# Acknowledgements

This library has been built over years, and in many cases on top of the work of brillant generous people like: [Inigo Quiles](https://www.iquilezles.org/), [Morgan McGuire](https://casual-effects.com/), [Hugh Kennedy](https://github.com/hughsk) and [Matt DesLauriers](https://www.mattdesl.com/).

# License 

LYGIA is dual-licensed under [the Prosperity License](https://prosperitylicense.com/versions/3.0.0) and the [Patron License](https://patronlicense.com/versions/1.0.0.html) for individual cases.

A [Patron License](https://patronlicense.com/versions/1.0.0.html) can be obtained by making regular payments through [GitHub Sponsorships](https://github.com/sponsors/patriciogonzalezvivo), in amounts qualifying for a tier of rewards that includes “patron licenses”. A Patron License grants qualifying patrons permission to ignore any noncommercial or copyleft rules in all of my License Zero Prosperity licensed software.

By becoming a Sponsor, you'll be helping to ensure I can spend the time not just fixing bugs, adding features, releasing new versions, but also keeping projects afloat and growing. Think of investing in me not just for the output of my code but my continued role in the open-source ecosystem. If open software producers and maintainers like me aren't supported then communities won't grow and there may not be a fresh package when you need a bug fixed.

Keeping LYGIA in growing healthy require works and dedication, I really appreciate your support improving it. That could be adding new functions, testing it, fixing bugs, translating the GLSL files to HLSL and Metal or just [sponsoring through GitHub](https://github.com/sponsors/patriciogonzalezvivo).


# Design Principles

This library:

* Relays on `#include "file"` which is defined by Khronos GLSL standard and suported by most engines and enviroments ( like [glslViewer](https://github.com/patriciogonzalezvivo/glslViewer/wiki/Compiling), [glsl-canvas VS Code pluging](https://marketplace.visualstudio.com/items?itemName=circledev.glsl-canvas), Unity, etc. ). It requires a tipical C-like pre-compiler MACRO which is easy to implement with just basic string operations to resolve dependencies. Here you can find some implementations on different languages:
    * C++: https://github.com/patriciogonzalezvivo/ada/blob/main/src/fs.cpp#L88-L171
    * Python: https://gist.github.com/patriciogonzalezvivo/9a50569c2ef9b08058706443a39d838e
    * JavaScript: 
        - vite: https://github.com/UstymUkhman/vite-plugin-glsl
        - esbuild: https://github.com/ricardomatias/esbuild-plugin-glsl-include
        - webpack: https://github.com/grieve/webpack-glsl-loader
        - observable: https://observablehq.com/d/e4e8a96f64a6bf81
        - vanilla: https://github.com/actarian/vscode-glsl-canvas/blob/91ff09bf6cec35e73d1b64e50b56ef3299d2fe6b/src/glsl/export.ts#L351

* it's very granular. One file, one function. Ex: `myFunc.glsl` contains `myFunct()`.

* There are some files that just include a collection of files inside a folder with the same name. For example:

```glsl
color/blend.glsl
// which includes
color/blend/*.glsl
```

* It's multi language. Right now most of it is on GLSL (`*.glsl`) but the goal is to have duplicate files for HLSL (`*.hlsl`) and Metal (`*.metal`).

```glsl
math/mix.glsl
math/mix.hlsl
```

* Check for name collisions using the following pattern where `FNC_` is followed with the function name:

```glsl
#ifndef FNC_MYFUNC
#define FNC_MYFUNC

float myFunc(float in) {
    return in;
}

#endif
```

* Have some templeting capabilities also through `#defines` probably the most frequent one is templating the sampling function for reusability. The `#define` options start with the name of the function, in this example `MYFUNC_`. They are added as `options:` in the header.
 
```glsl
/*
original_author: <FULL NAME>
description: [DESCRIPTION + URL]
use: myFunc(<sampler2D> texture, <vec2> st)
options: |
   MYFUNC_TYPE: return type
   MYFUNC_SAMPLER_FNC: function use to texture sample 
*/

#ifndef FNC_MYFUNC
#define FNC_MYFUNC

#ifndef MYFUNC_TYPE
#define MYFUNC_TYPE vec4
#endif

#ifndef MYFUNC_SAMPLER_FNC
#define MYFUNC_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV)
#endif

MYFUNC_TYPE myFunc(sampler2D tex, vec2 st) {
    return MYFUNC_SAMPLER_FNC(st);
}

#endif
```

* Utilize function overloading. For that please sort your arguments accordingly so it defaults gracefully. When possible sort them in the following order: `sampler2D, mat4, mat3, mat2, vec4, vec3, vec2, float, ivec4, ivec3, ivec2, int, bool`

```glsl
/*
original_author: <FULL NAME>
description: [DESCRIPTION + URL]
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
