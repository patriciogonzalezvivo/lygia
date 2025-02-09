#include "../space/lookAt.hlsl"
#include "../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Displace UV space into a XYZ space using an heightmap
use: <float3> displace(<SAMPLER_TYPE> tex, <float3> ro, <float3|float2> rd)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef DISPLACE_DEPTH
#define DISPLACE_DEPTH 1.
#endif

#ifndef DISPLACE_PRECISION
#define DISPLACE_PRECISION 0.01
#endif

#ifndef DISPLACE_SAMPLER
#define DISPLACE_SAMPLER(UV) tex2D(tex, UV).r
#endif

#ifndef DISPLACE_MAX_ITERATIONS
#define DISPLACE_MAX_ITERATIONS 120
#endif

#ifndef FNC_DISPLACE
#define FNC_DISPLACE
float3 displace(SAMPLER_TYPE tex, float3 ro, float3 rd) {

    // the z length of the target vector
    float dz = ro.z - DISPLACE_DEPTH;
    float t = dz / rd.z;

    // the intersection point between the ray and the highest point on the plane
    float3 prev = float3(
        ro.x - rd.x * t,
        ro.y - rd.y * t,
        ro.z - rd.z * t
    );
    
    float3 curr = prev;
    float lastD = prev.z;
    float hmap = 0.;
    float df = 0.;
    
    for (int i = 0; i < DISPLACE_MAX_ITERATIONS; i++) {
        prev = curr;
        curr = prev + rd * DISPLACE_PRECISION;

        hmap = DISPLACE_SAMPLER( curr.xy - 0.5 );
        // distance to the displaced surface
        float df = curr.z - hmap * DISPLACE_DEPTH;
        
        // if we have an intersection
        if (df < 0.0) {
            // linear interpolation to find more precise df
            float t = lastD / (abs(df)+lastD);
            return (prev + t * (curr - prev)) + float3(0.5, 0.5, 0.0);
        } 
        else
            lastD = df;
    }
    
    return float3(0.0, 0.0, 1.0);
}

float3 displace(SAMPLER_TYPE tex, float3 ro, float2 uv) {
    float3 rd = lookAt(-ro) * normalize(float3(uv - 0.5, 1.0));
    return displace(u_tex0Depth, ro, rd);
}
#endif