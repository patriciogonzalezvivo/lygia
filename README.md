![](https://artpil.com/wp-content/uploads/Lygia-Clark-banner.jpg)

# LYGIA: a multi-language shader library

Tired of reimplementing and searching for the same functions over and over, I started compiling and building a shader library of reusable assets (mostly functions) that can be include over and over. It's very granular, designed for reusability, performance and flexibility. 

## How does it work?

1. Clone this repository in your project, where your shaders are.

```bash
git clone https://github.com/patriciogonzalezvivo/lygia.git
```

2. In your shader `#include` the functions you need:

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


Learn more about how to use it on this repositories with **examples**:

* [2D examples for Processing (GLSL)](https://github.com/patriciogonzalezvivo/lygia_p5_examples)
* [2D examples for P5.js](https://editor.p5js.org/patriciogonzalezvivo/sketches/XCkTzoyB3)
* [2D examples for Three.js (GLSL)](https://github.com/patriciogonzalezvivo/lygia_threejs_examples) 
* [2D/3D examples for Unity3D (HLSL)](https://github.com/patriciogonzalezvivo/lygia_unity_examples)
* [2D/3D examples on GlslViewer (GLSL)](https://github.com/patriciogonzalezvivo/lygia_examples)

Join [#Lygia channel on shader.zone discord](https://shader.zone/) to learn how to use it, share work or get help.


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
* `lighting/`: different foward/deferred/raymarching lighting models and functions


## Flexible how?

There are some functions which behaviour can be change using `#defines` keyword before including it. For examples, [gaussian blurs](filter/gaussianBlur.glsl) are usually are done in two passes, so by default perform only 1D kernerls, but in the case you are interested on performing a 2D kernel all in the same pass you will need to add the `GAUSSIANBLUR_2D` keyword in the following way. 

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

* Relays on `#include "path/to/file.*lsl"` which is defined by Khronos GLSL standard and suported by most engines and enviroments ( like [glslViewer](https://github.com/patriciogonzalezvivo/glslViewer/wiki/Compiling), [glsl-canvas VS Code pluging](https://marketplace.visualstudio.com/items?itemName=circledev.glsl-canvas), Unity, etc. ). It requires a tipical C-like pre-compiler MACRO which is easy to implement with just basic string operations to resolve dependencies. Here you can find some implementations on different languages:
    * C++: https://github.com/patriciogonzalezvivo/ada/blob/main/src/fs.cpp#L88-L171
    * Python: https://gist.github.com/patriciogonzalezvivo/9a50569c2ef9b08058706443a39d838e
    * JavaScript: 
        - vite: https://github.com/UstymUkhman/vite-plugin-glsl
        - esbuild: https://github.com/ricardomatias/esbuild-plugin-glsl-include
        - webpack: https://github.com/grieve/webpack-glsl-loader
        - observable: https://observablehq.com/d/e4e8a96f64a6bf81
        - vanilla JS include dependency resolver: https://github.com/actarian/vscode-glsl-canvas/blob/91ff09bf6cec35e73d1b64e50b56ef3299d2fe6b/src/glsl/export.ts#L351

* it's **very granular**. One function per file. Where the file and the function share the same name. Ex: `myFunc.glsl` contains `myFunct()`. There are some files that just include a collection of files inside a folder with the same name. For example:

```glsl
color/blend.glsl
// which includes
color/blend/*.glsl
```

* It's **multi language**. Right now most of it is on GLSL (`*.glsl`) and HLSL (`*.hlsl`), but there is plans to extend it to Metal (`*.metal`).

```glsl
math/mix.glsl
math/mix.hlsl
```

* **Self documentation** each file contain a structured comment (in YAML) at the top of the file. This one contain the name of the original author, description, use and `#define` options

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

* **Templeting capabilities through `#defines`**,  probably the most frequent one is templating the sampling function for reusability. The `#define` options start with the name of the function, in this example `MYFUNC_`. They are added as `options:` in the header.
 
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

* Utilize **function overloading**. where arguments are arrange in such a way that optional elements are at the back. When possible sort them according their memory size (except textures that reamin at the top). Ex.: `sampler2D, mat4, mat3, mat2, vec4, vec3, vec2, float, ivec4, ivec3, ivec2, int, bool`

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

# Acknowledgements

This library has been built over years, and in many cases on top of the work of brillant generous people like: [Inigo Quiles](https://www.iquilezles.org/), [Morgan McGuire](https://casual-effects.com/), [Hugh Kennedy](https://github.com/hughsk) and [Matt DesLauriers](https://www.mattdesl.com/).

# License 

LYGIA is dual-licensed under [the Prosperity License](https://prosperitylicense.com/versions/3.0.0) and the [Patron License](https://patronlicense.com/versions/1.0.0.html) for individual cases.

A [Patron License](https://patronlicense.com/versions/1.0.0.html) can be obtained by making regular payments through [GitHub Sponsorships](https://github.com/sponsors/patriciogonzalezvivo), in amounts qualifying for a tier of rewards that includes “patron licenses”. A Patron License grants qualifying patrons permission to ignore any noncommercial or copyleft rules in all of [the Prosperity Licensed](https://prosperitylicense.com/versions/3.0.0) software.

Keeping LYGIA healthy require works and dedication, I will really appreciate your support. That could by contributing new code (functions or examples in new enviroments like Processing, TouchDesigner, Three.js, OpenFrameworks, etc), or fixing bugs and translating the GLSL/HLSL to Metal. 

Another way to support is by [sponsoring through GitHub](https://github.com/sponsors/patriciogonzalezvivo). By becoming a Sponsor, you'll be helping to ensure I can spend more time fixing bugs, adding features, releasing new versions, and making more examples and expanding the support for new frameworks.
