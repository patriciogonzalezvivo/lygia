/*
author: Patricio Gonzalez Vivo
description: expands mix to linearly mix more than two values
use: mix(<float|vec2|vec3|vec4> a, <float|vec2|vec3|vec4> b, <float|vec2|vec3|vec4> c [, <float|vec2|vec3|vec4> d], <float> pct)
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_MIX
#define FNC_MIX

float   mmix(in float a, in float b, in float c) { return mix(a, b, c); }
vec2    mmix(in vec2 a, in vec2 b, in float c) { return mix(a, b, c); }
vec2    mmix(in vec2 a, in vec2 b, in vec2 c) { return mix(a, b, c); }
vec3    mmix(in vec3 a, in vec3 b, in float c) { return mix(a, b, c); }
vec3    mmix(in vec3 a, in vec3 b, in vec3 c) { return mix(a, b, c); }
vec4    mmix(in vec4 a, in vec4 b, in float c) { return mix(a, b, c); }
vec4    mmix(in vec4 a, in vec4 b, in vec4 c) { return mix(a, b, c); }

float mmix(float a , float b, float c, float pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec2 mmix(vec2 a , vec2 b, vec2 c, float pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec2 mmix(vec2 a , vec2 b, vec2 c, vec2 pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec3 mmix(vec3 a , vec3 b, vec3 c, float pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec3 mmix(vec3 a , vec3 b, vec3 c, vec3 pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec4 mmix(vec4 a , vec4 b, vec4 c, float pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec4 mmix(vec4 a , vec4 b, vec4 c, vec4 pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

float mmix(in float a , in float b, in float c, in float d, in float pct) {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

vec2 mmix(in vec2 a , in vec2 b, in vec2 c, in vec2 d, in float pct) {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

vec2 mmix(in vec2 a , in vec2 b, in vec2 c, in vec2 d, in vec2 pct) {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

vec3 mmix(in vec3 a , in vec3 b, in vec3 c, in vec3 d, in float pct) {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

vec3 mmix(in vec3 a , in vec3 b, in vec3 c, in vec3 d, in vec3 pct) {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

vec4 mmix(in vec4 a , in vec4 b, in vec4 c, in vec4 d, in float pct) {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

vec4 mmix(in vec4 a , in vec4 b, in vec4 c, in vec4 d, in vec4 pct) {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}
#endif
