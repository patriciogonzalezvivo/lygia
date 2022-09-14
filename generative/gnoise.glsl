#include "random.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: gradient Noise 
use: gnoise(<float> x)
*/

#ifndef FNC_GNOISE
#define FNC_GNOISE

float gnoise(float x) {
    float i = floor(x);  // integer
    float f = fract(x);  // fraction
    return mix(random(i), random(i + 1.0), smoothstep(0.,1.,f)); 
}

#endif