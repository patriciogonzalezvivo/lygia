#include "2DCube.hlsl"
#include "../space/scale.hlsl"
#include "../sdf/boxSDF.hlsl"
#include "../sdf/opIntersection.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Use a 2D texture as to encode a 3D SDF (Signed Distance Field) function
use: <vecSAMPLE3DSDF_TYPE4> sample3DSdf(in <SAMPLER_TYPE> lut, in <float3> xyz)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLE_2DCUBE_CELL_SIZE
    - SAMPLE_2DCUBE_CELLS_PER_SIDE: default 8
    - SAMPLE_2DCUBE_FNC
    - SAMPLE3DSDF_TYPE: defaults to float
    - SAMPLE3DSDF_FNC(TEX, POS): defaults to sample2DCube(TEX, POS).r
examples:
    - /shaders/sample_3Dsdf.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SAMPLE3DSDF_TYPE
#define SAMPLE3DSDF_TYPE float
#endif

#ifndef SAMPLE3DSDF_FNC
#define SAMPLE3DSDF_FNC(TEX, POS) sample2DCube(TEX, POS).r
#endif

#ifndef FNC_SAMPLE3DSDF
#define FNC_SAMPLE3DSDF
SAMPLE3DSDF_TYPE sample3DSdf(SAMPLER_TYPE tex, float3 pos) {
    pos += 0.5;
    pos = scale(pos, 0.5);
    SAMPLE3DSDF_TYPE sdf = (SAMPLE3DSDF_FNC(tex, pos) * 2.0 - 1.0) * 1.5;
    return opIntersection( boxSDF(pos - 0.5, float3(0.5, 0.5, 0.5)), sdf);
}
#endif