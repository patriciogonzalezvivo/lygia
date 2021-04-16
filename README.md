![](https://artpil.com/wp-content/uploads/Lygia-Clark-banner.jpg)

# Lygia: multi-language shader library

Tire of reimplementing and searching for the same functions over and over, started compiling and building a shader library of reusable assets (mostly functions) that can be include over and over. It's very granular, designed for reusability, performance and flexibility. 

## How it works?

You just include the functions you need:

```
#include "lygia/space/ratio.glsl"
#include "lygia/draw/fill.glsl"
#include "lygia/draw/stroke.glsl"
#include "lygia/sdf/circleSDF.glsl"
#include "lygia/sdf/triSDF.glsl"
#include "lygia/sdf/flowerSDF.glsl"

void main(void) {
    vec3 color = vec3(0.0);
    vec2 st = gl_FragCoord.xy/u_resolution.xy;
    st = ratio(st, u_resolution);
    
    color += stroke(circleSDF(st),.9,.1);
    color += fill(flowerSDF(st.yx,3),.2);
    color -= fill(triSDF(vec2(st.x,.98-st.y)),.15);
    
    gl_FragColor = vec4(color, 1.0);
}
```

The functions are divided in different categories:

* `math/`: general math functions and constants. 
* `space/`: general spatial operations like `scale()`, `rotate()`, etc.
* `color/`: general color operations like `luma()`, `saturation()`, etc.
    * `blend/`: typical blend photoshop operations
    * `space/`: color space conversions 
* `animation/`: animation operations
    * `easing/`: easing functions
* `generative/`: generative functions like `random()`, `noise()`, etc. 
* `sdf/`: signed distance field generation functions. Most of them ported from [PixelSpiritDeck](https://patriciogonzalezvivo.github.io/PixelSpiritDeck/)
* `draw/`: functions that draw shapes, numbers, lines, etc. Most of them ported from [PixelSpiritDeck](https://patriciogonzalezvivo.github.io/PixelSpiritDeck/)
* `filters/`: typical filter operations like different kind of blurs, mean and median filters.


There are some functions that are "templeted" using `#defines`. You can change how it behaves by defining a keyword before including it. For examples, [gaussian blurs](filter/gaussianBlur.glsl) usually are done in two passes (and it defaults), but let's say you are in a rush you can specify to use 

```
#define GAUSSIANBLUR_2D
#include "filter/gaussianBlur.glsl"

void main(void) {

    ...

    color = gaussianBlur(u_tex0, uv, 1./u_resolution, 9);

    ...
}
```

# Acknowledgements

This library have been build over years, and most often than not on top of the work of brillant generous people like: [Inigo Quiles](https://www.iquilezles.org/), [Morgan McGuire](https://casual-effects.com/), [Hugh Kennedy](https://github.com/hughsk), [Matt DesLauriers](https://www.mattdesl.com/).
I have tried to give according credits and correct license to each file. It's not perfect but it could be with your help! Please if you see something, say somthing.

Lygia is open sourced under the terms of the [BSD license](LICENSE). You are free to use it, extend it and redistribute without charge, but I really appreciate if you can support improving it. That could be adding new functions, testing it, fixing bugs, translating the GLSL files to HLSL and Metal or just [sponsoring me through GitHub](https://github.com/sponsors/patriciogonzalezvivo) to do it for you.  


# Design Principles

This library:

* Relays on `#include "file"` which is defined by Khornos GLSL standard and suported by most engines and enviroments. It follows a tipical C-like pre-compiler MACRO which is easy to implement with simple string operations to resolve dependencies. Probably the most important thing to solve while implementing is avoiding dependency loops, and if it's possible prevent duplication. If you need some example code of how to resolve this in:
    * C++: https://github.com/patriciogonzalezvivo/glslViewer/blob/master/src/io/fs.cpp#L104
    * Python: https://gist.github.com/patriciogonzalezvivo/9a50569c2ef9b08058706443a39d838e
    * JS: https://github.com/actarian/vscode-glsl-canvas/blob/91ff09bf6cec35e73d1b64e50b56ef3299d2fe6b/src/glsl/export.ts#L351

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

* Have embedded description, use, authorship and license in YAML style so eventually could be parsed easily. If you add new functions please use this template:

```glsl
/*
author: <FULL NAME>
description: [DESCRIPTION + URL]
use: myFunc(<float> input)
options: none
license: |
  This software is released under the MIT license:
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/
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

* If the function have dependencies to other files, add them on the first lines of the file, before the authorship/description/license header and outside the `#ifndef` flag check. So once pre-compiled things are use/description/license are cristal clear and nicelly separated.

```glsl
#include "../this/other/function.glsl"

/*
author: <FULL NAME>
...

#ifndef FNC_MYFUNC
#define FNC_MYFUNC
...
```

* Have some templeting capabilities also through `#defines` probably the most frequent one is templeting the sampling function for reusability. The `#define` options start with the name of the function, in this example `MYFUNC_`. They are added as `options:` in the header.
 
```glsl
/*
author: <FULL NAME>
description: [DESCRIPTION + URL]
use: myFunc(<sampler2D> texture, <vec2> st)
options: |
   MYFUNC_TYPE: return type
   MYFUNC_SAMPLER_FNC: function use to texture sample 
license: ...
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

* Utilize function overloading. For that please sort your arguments accordingly so it default gracefully. When it's possible sort them in the following order: `sampler2D, mat4, mat3, mat2, vec4, vec3, vec2, float, ivec4, ivec3, ivec2, int, bool`

```glsl
/*
author: <FULL NAME>
description: [DESCRIPTION + URL]
use: myFunc(<vec2> st, <vec2|float> x[, <float> y])
license: ...
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
