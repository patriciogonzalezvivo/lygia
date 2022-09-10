#include "../math/saturate.glsl"

/*
author: Patricio Gonzalez Vivo
description: Simple Ripple Propagation
use: <vec3> ripple(<sampler2D> tex, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - RIPPLE_SAMPLER_FNC(UV)
license: |
    Copyright (c) 2022 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/
#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef RIPPLE_SAMPLER_FNC
#define RIPPLE_SAMPLER_FNC(UV) SAMPLER_FNC(tex, UV)
#endif

#ifndef FNC_RIPPLE
#define FNC_RIPPLE
vec3 ripple(sampler2D tex, vec2 st, vec2 pixel) {
    vec3 rta = RIPPLE_SAMPLER_FNC(st).rgb;
   	float s0 = rta.y;
    float s1 = RIPPLE_SAMPLER_FNC(st + vec2(0.0,   -pixel.y)).r;    //     s1
    float s2 = RIPPLE_SAMPLER_FNC(st + vec2(-pixel.x,   0.0)).r;    //  s2 s0 s3
    float s3 = RIPPLE_SAMPLER_FNC(st + vec2( pixel.x,   0.0)).r;    //     s4
    float s4 = RIPPLE_SAMPLER_FNC(st + vec2(0.0,    pixel.y)).r;
    float d = -(s0 - .5) * 2. + (s1 + s2 + s3 + s4 - 2.);
    d *= 0.99;
    d = saturate(d * 0.5 + 0.5);
    return vec3(d, rta.x, 0.0);
}
#endif