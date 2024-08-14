#include "../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: triplanar mapping
use: <float4> sample2DCube(in <SAMPLER_TYPE> lut, in <float3> xyz)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLETRIPLANAR_TYPE: optional depending the target version of GLSL (vec4 or float4)
    - SAMPLETRIPLANAR_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


#ifndef SAMPLETRIPLANAR_TYPE
#define SAMPLETRIPLANAR_TYPE float4
#endif

#ifndef SAMPLETRIPLANAR_FNC
#define SAMPLETRIPLANAR_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_SAMPLETRIPLANAR
#define FNC_SAMPLETRIPLANAR
SAMPLETRIPLANAR_TYPE sampleTriplanar(SAMPLER_TYPE tex, in float3 d) {
    SAMPLETRIPLANAR_TYPE colx = SAMPLETRIPLANAR_FNC(tex, d.yz);
    SAMPLETRIPLANAR_TYPE coly = SAMPLETRIPLANAR_FNC(tex, d.zx);
    SAMPLETRIPLANAR_TYPE colz = SAMPLETRIPLANAR_FNC(tex, d.xy);
    
    float3 n = d*d;
    return (colx*n.x + coly*n.y + colz*n.z)/(n.x+n.y+n.z);
}

// iq's cubemap function
SAMPLETRIPLANAR_TYPE sampleTriplanar(SAMPLER_TYPE tex, in float3 d, in float s) {
    SAMPLETRIPLANAR_TYPE colx = SAMPLETRIPLANAR_FNC(tex, 0.5 + s*d.yz/d.x);
    SAMPLETRIPLANAR_TYPE coly = SAMPLETRIPLANAR_FNC(tex, 0.5 + s*d.zx/d.y);
    SAMPLETRIPLANAR_TYPE colz = SAMPLETRIPLANAR_FNC(tex, 0.5 + s*d.xy/d.z);
    
    float3 n = d * d;
    return (colx*n.x + coly*n.y + colz*n.z)/(n.x+n.y+n.z);
}
#endif