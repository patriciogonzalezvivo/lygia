#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: turns alpha to zero if it's outside the texture normalize coordinates
use: <vec4> sampleZero(<SAMPLER_TYPE> tex, <vec2> st);
options:
    - SAMPLER_FNC(TEX, UV)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLEZERO
#define FNC_SAMPLEZERO
vec4 sampleZero(SAMPLER_TYPE tex, vec2 st) { 
    return SAMPLER_FNC( tex, st ) * vec4(1.0,1.0,1.0, (st.x <= 0.0 || st.x >= 1.0 || st.y <= 0.0 || st.y >= 1.0)? 0.0 : 1.0); 
}

vec4 sampleZero(SAMPLER_TYPE tex, vec2 st, float pad) { 
    return SAMPLER_FNC( tex, st ) * vec4(1.0,1.0,1.0, (st.x <= pad || st.x >= 1.0-pad || st.y <= pad || st.y >= 1.0 - pad)? 0.0 : 1.0); 
}

vec4 sampleZero(SAMPLER_TYPE tex, vec2 st, vec2 pad) { 
    return SAMPLER_FNC( tex, st ) * vec4(1.0,1.0,1.0, (st.x <= pad.x || st.x >= 1.0-pad.x || st.y <= pad.y || st.y >= 1.0 - pad.y)? 0.0 : 1.0); 
}

vec4 sampleZero(SAMPLER_TYPE tex, vec2 st, vec4 pad) { 
    return SAMPLER_FNC( tex, st ) * vec4(1.0,1.0,1.0, (st.x <= pad.x || st.x >= pad.z || st.y <= pad.y || st.y >= pad.w)? 0.0 : 1.0); 
}

#ifdef STR_AABB
vec4 sampleZero(SAMPLER_TYPE tex, vec2 st, AABB pad) { 
    return SAMPLER_FNC( tex, st ) * vec4(1.0,1.0,1.0, (st.x <= pad.min.x || st.x >= pad.max.x || st.y <= pad.min.y || st.y >= pad.max.y)? 0.0 : 1.0); 
}
#endif

#endif