#include "rgb2hue.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Converts a RGB rainbow pattern back to a single float value
use: <float> rgb2heat(<vec3|vec4> color)
*/

#ifndef FNC_RGB2HEAT
#define FNC_RGB2HEAT
float rgb2heat(vec3 c) {
    return 1.0 - rgb2hue(c) * 1.538461538;
}

float rgb2heat(vec4 c) { return rgb2heat(c.rgb); }
#endif