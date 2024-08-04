/*
contributors: Patricio Gonzalez Vivo
description: create a look at matrix
use:
    - <float3x3> lookAt(<float3> forward, <float3> up)
    - <float3x3> lookAt(<float3> target, <float3> eye, <float3> up)
    - <float3x3> lookAt(<float3> target, <float3> eye, <float> roll)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LOOKAT
#define FNC_LOOKAT

float3x3 lookAt(float3 forward, float3 up)
{
    float3 zaxis = forward;
#if defined (LOOK_AT_RIGHT_HANDED)
    float3 xaxis = normalize(cross(zaxis, up));
    float3 yaxis = normalize(cross(xaxis, zaxis));
#else
    float3 xaxis = normalize(cross(up, zaxis));
    float3 yaxis = normalize(cross(zaxis, xaxis));
#endif
    float3x3 m;
    m._m00_m10_m20 = xaxis;
    m._m01_m11_m21 = yaxis;
    m._m02_m12_m22 = zaxis;
    return m;
}

float3x3 lookAt(float3 eye, float3 target, float3 up)
{
    float3 forward = normalize(target - eye);
    return lookAt(forward, up);
}

float3x3 lookAt(float3 eye, float3 target, float roll)
{
    float3 up = float3(sin(roll), cos(roll), 0.0);
    return lookAt(eye, target, up);
}

float3x3 lookAt(float3 forward)
{
    return lookAt(forward, float3(0.0, 1.0, 0.0));
}

#endif