#include "../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: turns alpha to zero if it's outside the texture normalize coordinates
use: <vec4> sampleZero(<SAMPLER_TYPE> tex, <vec2> st);
options:
    - SAMPLER_FNC(TEX, UV)
*/

#ifndef FNC_SAMPLEZERO
#define FNC_SAMPLEZERO
vec4 sampleZero(SAMPLER_TYPE tex, vec2 st) { 
    return SAMPLER_FNC( tex, st ) * vec4(1.0,1.0,1.0, (st.x <= 0.0 || st.x >= 1.0 || st.y <= 0.0 || st.y >= 1.0)? 0.0 : 1.0); 
}

vec4 sampleZero(SAMPLER_TYPE tex, vec2 st, float edge) { 
    return SAMPLER_FNC( tex, st ) * vec4(1.0,1.0,1.0, (st.x <= edge || st.x >= 1.0-edge || st.y <= edge || st.y >= 1.0-edge)? 0.0 : 1.0); 
}
#endif