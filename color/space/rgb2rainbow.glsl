#include "rgb2hue.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Converts a RGB rainbow pattern back to a single float value
use: <float> rgb2rainbow(<vec3|vec4> color)
*/

#ifndef FNC_RGB2RAINBOW
#define FNC_RGB2RAINBOW
float rgb2rainbow(vec3 c) {
    return 1.025 - rgb2hue(c) * 1.538461538;
}

float rgb2rainbow(vec4 c) { return rgb2rainbow(c.rgb); }
#endif