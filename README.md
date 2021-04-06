# SLIB: a flexible shader library of common operations 

Tire of reimplementing or researching the same shader functions I start building my own shader library. This library have build over the years of my profesional carreer and is building on top of the work people smarter than me. I have tried to give according credits and correct license to each file. It's not perfect but it could be, please if you see something odd let me know.

## Principles

* this library relays on `#include "file"` which is defined by Khornos GLSL standard and suported by most engines like Unity. It follows a tipical C-like pre-compiler MACRO which is easy to implement in most enviroments simple string operations to resolve dependencies. Probably the most important thing to solve while implementing is avoiding dependency loops. If you need some example code of how to resolve this in:
    * C++: https://github.com/patriciogonzalezvivo/glslViewer/blob/master/src/io/fs.cpp#L104

* have embebed name collisions so you don't need to worry about calling a function twice. It use the following pattern:

```
#ifndef FNC_NAME
#define FNC_NAME
...
#endif
```

* it have some templeting capabilities also through `#defines` probably the most frequent one is templeting the sampling function for reusability:

```
#ifndef SAMPLER_FNC
#define SAMPLER_FNC(POS_UV) texture(tex,POS_UV).r
#endif
```

* it's very granular. One file one function. There are some files that just include a collection of files with the same name. For example:

```
color/blend.glsl
# includes
color/blend/*.glsl
```

* it meant to be multi language. Right now mostly is on GLSL (`*.glsl`) but the goal is to have duplicate files on HLSL (`*.hlsl`) and Metal (`*.metal`)



