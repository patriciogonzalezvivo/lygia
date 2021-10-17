#include "../../math/saturate.glsl"

/*
Author: Narkowicz 2015
description: ACES Filmic Tone Mapping Curve. https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
use: tonemapACES(<vec3> x)
*/

#ifndef FNC_TONEMAPACES
#define FNC_TONEMAPACES

vec3 tonemapACES(in vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate(x * (a * x + b)) / (x * (c * x + d) + e);
}

#endif