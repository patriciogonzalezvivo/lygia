/*
contributors: [Stefan Gustavson, Ian McEwan]
description: grad4, used for snoise(float4 v)
use: <float4> grad4(<float> j, <float4> ip)
*/

#ifndef FNC_GRAD4
#define FNC_GRAD4

float4 grad4(float j, float4 ip) {
    const float4 ones = float4(1.0, 1.0, 1.0, -1.0);
    float4 p, s;

    p.xyz = floor( frac (float3(j, j, j) * ip.xyz) * 7.0) * ip.z - 1.0;
    p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
    // GLSL: s = float4(lessThan(p, float4(0.0)));
    s = float4(1 - step(float4(0, 0, 0, 0), p));
    p.xyz = p.xyz + (s.xyz * 2.0 - 1.0) * s.www;

    return p;
}

#endif
