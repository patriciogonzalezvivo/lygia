#include "../../math/saturate.glsl"

/*
contributors: Narkowicz 2015
description: ACES Filmic Tone Mapping Curve. https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
use: <vec3|vec4> tonemapACES(<vec3|vec4> x)
*/

#ifndef FNC_TONEMAPACES
#define FNC_TONEMAPACES
vec3 tonemapACES(vec3 v) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((v*(a*v+b))/(v*(c*v+d)+e));
}

vec4 tonemapACES(in vec4 v) {
    return vec4(tonemapACES(v.rgb), v.a);
}
#endif