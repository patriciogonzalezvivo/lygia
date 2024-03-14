/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: <float3x3> rotate3dX(<float> radians)
*/

#ifndef FNC_ROTATE3DX
#define FNC_ROTATE3DX
float3x3 rotate3dX(const in float r){
    return float3x3(float3(1.0,0.0,0.0),
                    float3(0.0,cos(r),-sin(r)),
                    float3(0.0,sin(r),cos(r)));
}
#endif
