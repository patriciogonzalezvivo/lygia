/*
contributors: Patricio Gonzalez Vivo
description: create a look up matrix 
use: 
    - <float3x3> lookAt(<float3> forward, <float3> up)
    - <float3x3> lookAt(<float3> target, <float3> eye, <float3> up)
    - <float3x3> lookAt(<float3> target, <float3> eye, <float> rolle)
*/

#ifndef FNC_LOOKAT
#define FNC_LOOKAT

float3x3 lookAt(float3 forward, float3 up) {
    float3 xaxis = normalize(cross(forward, up));
    float3 yaxis = up;
    float3 zaxis = forward;
    return float3x3(xaxis, yaxis, zaxis);
}

float3x3 lookAt(float3 eye, float3 target, float3 up) {
    float3 zaxis = normalize(target - eye);
    float3 xaxis = normalize(cross(zaxis, up));
    float3 yaxis = cross(zaxis, xaxis);
    return float3x3(xaxis, yaxis, zaxis);
}

float3x3 lookAt(float3 eye, float3 target, float roll) {
    float3 up = float3(sin(roll), cos(roll), 0.0);
    float3 zaxis = normalize(target - eye);
    float3 xaxis = normalize(cross(zaxis, up));
    float3 yaxis = normalize(cross(xaxis, zaxis));
    return float3x3(xaxis, yaxis, zaxis);
}

#endif