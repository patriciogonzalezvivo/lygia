#include "rgb2hue.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB rainbow pattern back to a single float value
use: <float> rgb2heat(<vec3|vec4> color)
*/

#ifndef FNC_RGB2HEAT
#define FNC_RGB2HEAT
float rgb2heat(const in vec3 c) { return 1.025 - rgb2hue(c) * 1.538461538; }
float rgb2heat(const in vec4 c) { return rgb2heat(c.rgb); }
#endif