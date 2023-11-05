#include "../../math/saturate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simpler water color ramp 
use: <vec3> water(<float> value)
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/color/palette/water.frag
*/

#ifndef FNC_WATER
#define FNC_WATER

vec3 water(float x) {
    return pow(vec3(.1, .7, .8), vec3(4.* saturate(1.0-x) ));
}

#endif