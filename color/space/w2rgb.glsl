#include "gamma2linear.glsl"
#include "../../math/saturate.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: wavelength to RGB
use: <vec3> w2rgb(<float> wavelength)
*/

#ifndef FNC_W2RGB
#define FNC_W2RGB
vec3 w2rgb(float w) {
    float x = saturate((w - 400.0)/ 300.0);
    const vec3 c1 = vec3(3.54585104, 2.93225262, 2.41593945);
    const vec3 x1 = vec3(0.69549072, 0.49228336, 0.27699880);
    const vec3 y1 = vec3(0.02312639, 0.15225084, 0.52607955);
    const vec3 c2 = vec3(3.90307140, 3.21182957, 3.96587128);
    const vec3 x2 = vec3(0.11748627, 0.86755042, 0.66077860);
    const vec3 y2 = vec3(0.84897130, 0.88445281, 0.73949448);
    return gamma2linear( bump(c1 * (x - x1), y1) + bump(c2 * (x - x2), y2) );
}
#endif