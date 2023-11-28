#include "../../math/saturate.glsl"

/*
contributors: Inigo Quiles
description: |
    Visual approximation on Penner's paper on preintegrated SSS, Siggraph 2011. https://www.shadertoy.com/view/llXBWn
use: <vec3> penner( <float> NoL, <float> ir )
license: MIT License (MIT) Copyright 2017 Inigo Quilez
*/


#ifndef FNC_PENNER
#define FNC_PENNER

vec3 penner( float NoL, float ir ) {
    float pndl = saturate( NoL);
    float nndl = saturate(-NoL);

    return vec3(pndl) + 
#if defined(PLATFORM_RPI)
           vec3(1.0,0.1,0.01) * 0.7 * pow(saturate(ir*0.75-nndl),2.0);
#else
        //    vec3(1.0,0.1,0.01) * 0.2 * (1.0-pndl)*(1.0-pndl) * pow(1.0-nndl, 3.0/(ir+0.001)) * saturate(ir-0.04);
           vec3(1.0,0.2,0.05) * 0.250 * (1.0-pndl)*(1.0-pndl) * pow(1.0-nndl,3.0/(ir+0.001)) * saturate(ir-0.04);
#endif
}

#endif