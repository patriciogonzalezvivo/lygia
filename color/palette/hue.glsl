#include "../../math/cubic.glsl" 
/*
original_author: Patricio Gonzalez Vivo
description: |
    Physical Hue. 
    
    Ratio: 

    * 1/3 = neon
    * 1/4 = refracted
    * 1/5+ = approximate white

use: <vec3> hue(<float> hue[, <float> ratio])
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/color/palette/hue.frag
*/

#ifndef FNC_PALETTE_HUE
#define FNC_PALETTE_HUE

vec3 hue(float x, float ratio) {
    // return smoothstep(  vec3(0.9059, 0.8745, 0.8745), vec3(1.0), 
    return cubic(
                        abs( mod(x + vec3(0.0,1.0,2.0) * ratio, 1.0) * 2.0 - 1.0));
}

vec3 hue(float x) { return hue(x, 0.33333); }

#endif