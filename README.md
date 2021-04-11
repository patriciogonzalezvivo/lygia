![](https://artpil.com/wp-content/uploads/Lygia-Clark-banner.jpg)

# Lygia: multi-language shader library

Tire of reimplementing and searching for the same functions over and over, started compiling and building a shader library. It's very granular, with interdependencies, designed for reusability, performance and flexibility. 

This library have build over years, most often than not on top of the work of smarter people. Tried to give according credits and correct license to each file. It's not perfect but it could be with your help! Please if you see something odd or missing sumit a PR.

## Principles

This library:

* Relays on `#include "file"` which is defined by Khornos GLSL standard and suported by most engines and enviroments. It follows a tipical C-like pre-compiler MACRO which is easy to implement with simple string operations to resolve dependencies. Probably the most important thing to solve while implementing is avoiding dependency loops, and if it's possible prevent duplication. If you need some example code of how to resolve this in:
    * C++: https://github.com/patriciogonzalezvivo/glslViewer/blob/master/src/io/fs.cpp#L104
    * Python: https://gist.github.com/patriciogonzalezvivo/9a50569c2ef9b08058706443a39d838e

* it's very granular. One file, one function. Ex: `myFunc.glsl` contains `myFunct()`.

* There are some files that just include a collection of files inside a folder with the same name. For example:

```
color/blend.glsl
// which includes
color/blend/*.glsl
```

* It's multi language. Right now most of it is on GLSL (`*.glsl`) but the goal is to have duplicate files for HLSL (`*.hlsl`) and Metal (`*.metal`).

```
math/mix.glsl
math/mix.hlsl
```

* Have embedded description, use, authorship and license in YAML style so eventually could be parsed easily. If you add new functions please use this template:

```
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

```
#ifndef FNC_MYFUNC
#define FNC_MYFUNC

float myFunc(float in) {
    return in;
}

#endif
```

* If the function have dependencies to other files, add them on the first lines of the file, before the authorship/description/license header and outside the `#ifndef` flag check. So once pre-compiled things are use/description/license are cristal clear and nicelly separated.

```
#include "../this/other/function.glsl"

/*
author: <FULL NAME>
...

#ifndef FNC_MYFUNC
#define FNC_MYFUNC
...
```

* Have some templeting capabilities also through `#defines` probably the most frequent one is templeting the sampling function for reusability. The `#define` options start with the name of the function, in this example `MYFUNC_`. They are added as `options:` in the header.
 
```
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

```
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
