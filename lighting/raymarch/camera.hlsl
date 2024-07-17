/*
contributors:  Inigo Quiles
description: Set a camera for raymarching 
use: <float3x3> raymarchCamera(in <float3> ro, [in <float3> ta [, in <float3> up] ])
*/

#ifndef FNC_RAYMARCHCAMERA
#define FNC_RAYMARCHCAMERA

float3x3 raymarchCamera( in float3 ro, in float3 ta, in float3 up ) {
    float3 cw = normalize(ta-ro);
    float3 cu = normalize( cross(cw,up) );
    float3 cv = normalize( cross(cu,cw) );
    // glsl is row-major anh hlsl column major.
    // float3x3 is initialised with row vectors, but cu, cv and cw are columns vector
    // thus, we need to transpose after initialization
    return transpose(float3x3(cu, cv, cw));
}

float3x3 raymarchCamera( in float3 ro, in float3 ta, float cr ) {
    float3 cw = normalize(ta-ro);
    float3 cp = float3(sin(cr), cos(cr),0.0);
    float3 cu = normalize( cross(cw,cp) );
    float3 cv =          ( cross(cu,cw) );
    return transpose(float3x3(cu, cv, cw));
}

float3x3 raymarchCamera( in float3 ro, in float3 ta ) {
    return raymarchCamera( ro, ta, float3(0.0, 1.0, 0.0) );
}

float3x3 raymarchCamera( in float3 ro ) {
    return raymarchCamera( ro, float3(0.0, 0.0, 0.0), float3(0.0, 1.0, 0.0) );
}

#endif