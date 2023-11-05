/*
contributors: Patricio Gonzalez Vivo
description: get screen coordinates from view position 
use: <float2> view2screenPosition(<float3> viewPosition )
options:
    - CAMERA_PROJECTION_MATRIX: float4x4 matrix with camera projection
*/

#ifndef CAMERA_PROJECTION_MATRIX
#define CAMERA_PROJECTION_MATRIX u_projectionMatrix
#endif

#ifndef FNC_VIEW2SCREENPOSITION
#define FNC_VIEW2SCREENPOSITION
float2 view2screenPosition(float3 viewPosition){
    float4 clip = mul(CAMERA_PROJECTION_MATRIX, float4(viewPosition, 1.0));
    return (clip.xy / clip.w + 1.0) * 0.5;
}
#endif