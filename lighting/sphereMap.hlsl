/*
original_author: Patricio Gonzalez Vivo
description: given a Spherical Map texture and a normal direction returns the right pixel
use: spheremap(<sampler2D> texture, <float3> normal)
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef SPHEREMAP_TYPE
#define SPHEREMAP_TYPE float4
#endif

#ifndef SPHEREMAP_SAMPLER_FNC
#define SPHEREMAP_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV)
#endif

#ifndef FNC_SPHEREMAP
#define FNC_SPHEREMAP
float2 sphereMap(float3 normal, float3 eye) {
    // float3 reflected = reflect(eye, normal);
    // float m = 2.8284271247461903 * sqrt( reflected.z+1.0 );
    // return reflected.xy / m + 0.5;
    float3 r = reflect(eye, normal);
    r.z += 1.;
    float m = 2. * length(r);
    float2 uv = r.xy / m + .5;
}


SPHEREMAP_TYPE sphereMap (in sampler2D tex, in float3 normal, in float3 eye) {
    return SPHEREMAP_SAMPLER_FNC( sphereMap(normal, eye) );
}
#endif
