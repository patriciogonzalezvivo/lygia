#include "../color/space/w2rgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: wavelength to RGB
use: <float3> wavelength(<float> wavelength)
*/

#ifndef FNC_WAVELENGTH
#define FNC_WAVELENGTH
vec3 wavelength(float w) { return w2rgb(w); }
#endif
