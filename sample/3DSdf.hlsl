#include "2DCube.hlsl"
#include "../space/scale.hlsl"
#include "../sdf/boxSDF.hlsl"
#include "../sdf/opIntersection.hlsl"

#ifndef FNC_SAMPLE3DSDF
#define FNC_SAMPLE3DSDF
float sample3DSdf(SAMPLER_TYPE tex, float3 pos) {
    pos += .5;
    pos = scale(pos, .5);
    float sdf = (sample2DCube(tex, pos).r * 2.0 - 1.0) * 1.5;
    return opIntersection( boxSDF(pos - 0.5, float3(.5, .5, .5)), sdf);
}
#endif