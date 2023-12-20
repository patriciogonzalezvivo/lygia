#include "rgb2hue.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB rainbow pattern back to a single float value
use: <float> rgb2heat(<float3|float4> color)
*/

#ifndef FNC_RGB2HEAT
#define FNC_RGB2HEAT
float rgb2heat(float3 c) { return 1.025 - rgb2hue(c) * 1.538461538; }
float rgb2heat(float4 c) { return rgb2heat(c.rgb); }
#endif