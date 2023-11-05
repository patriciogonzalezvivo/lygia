#include "../color/space/k2rgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: blackbody in kelvin to RGB. Range between 0.0 and 40000.0 Kelvin
use: <vec3> blackbody(<float> wavelength)
*/

#ifndef FNC_BLACKBODY
#define FNC_BLACKBODY
vec3 blackbody(const in float k) { return k2rgb(k); }
#endif