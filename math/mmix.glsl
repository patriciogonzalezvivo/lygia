/*
contributors: Patricio Gonzalez Vivo
description: expands mix to linearly mix more than two values
use: <float|vec2|vec3|vec4> mmix(<float|vec2|vec3|vec4> a, <float|vec2|vec3|vec4> b, <float|vec2|vec3|vec4> c [, <float|vec2|vec3|vec4> d], <float> pct)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MIX
#define FNC_MIX

float   mmix(const in float a, const in float b, const in float c) { return mix(a, b, c); }
vec2    mmix(const in vec2 a, const in vec2 b, const in float c) { return mix(a, b, c); }
vec2    mmix(const in vec2 a, const in vec2 b, const in vec2 c) { return mix(a, b, c); }
vec3    mmix(const in vec3 a, const in vec3 b, const in float c) { return mix(a, b, c); }
vec3    mmix(const in vec3 a, const in vec3 b, const in vec3 c) { return mix(a, b, c); }
vec4    mmix(const in vec4 a, const in vec4 b, const in float c) { return mix(a, b, c); }
vec4    mmix(const in vec4 a, const in vec4 b, const in vec4 c) { return mix(a, b, c); }

float mmix(const in float a, const in float b, const in float c, const in float pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec2 mmix(vec2 a, vec2 b, vec2 c, float pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec2 mmix(vec2 a, vec2 b, vec2 c, vec2 pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec3 mmix(vec3 a, vec3 b, vec3 c, float pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec3 mmix(vec3 a, vec3 b, vec3 c, vec3 pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec4 mmix(vec4 a, vec4 b, vec4 c, float pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec4 mmix(vec4 a, vec4 b, vec4 c, vec4 pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

float mmix(in float a, in float b, in float c, in float d, in float pct) {
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

vec2 mmix(in vec2 a, in vec2 b, in vec2 c, in vec2 d, in float pct) {
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

vec2 mmix(in vec2 a, in vec2 b, in vec2 c, in vec2 d, in vec2 pct) {
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

vec3 mmix(in vec3 a, in vec3 b, in vec3 c, in vec3 d, in float pct) {
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

vec3 mmix(in vec3 a, in vec3 b, in vec3 c, in vec3 d, in vec3 pct) {
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

vec4 mmix(in vec4 a, in vec4 b, in vec4 c, in vec4 d, in float pct) {
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

vec4 mmix(in vec4 a, in vec4 b, in vec4 c, in vec4 d, in vec4 pct) {
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
