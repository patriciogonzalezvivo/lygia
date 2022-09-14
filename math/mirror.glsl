/*
original_author: Ian Heisters
description: Transforms the input signal into a triangle wave. For instance, if x goes between 0 and 2, the returned value will go from 0 to 1, and then 1 to 0 in a triangle shape.
use: mirror(<vec2|float> x)
*/

#ifndef FNC_MIRROR
#define FNC_MIRROR
float mirror(in float x) {
    float f = fract(x);
    float m = floor(mod(x, 2.));
    float fm = f * m;
    return f + m - fm * 2.;
}

vec2 mirror(in vec2 xy) {
    vec2 f = fract(xy);
    vec2 m = floor(mod(xy, 2.));
    vec2 fm = f * m;
    return f + m - fm * 2.;
}
#endif
