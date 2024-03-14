/*
contributors: Ian Heisters
description: Transforms the input signal into a triangle wave. For instance, if x goes between 0 and 2, the returned value will go from 0 to 1, and then 1 to 0 in a triangle shape.
use: <float|vec2> mirror(<float|vec2> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
*/

#ifndef FNC_MIRROR
#define FNC_MIRROR
float mirror(in float x) {
    float f = fract(x);
    float m = floor(mod(x, 2.));
    float fm = f * m;
    return f + m - fm * 2.;
}

vec2 mirror(in vec2 v) {
    vec2 f = fract(v);
    vec2 m = floor(mod(v, 2.));
    vec2 fm = f * m;
    return f + m - fm * 2.;
}
#endif
