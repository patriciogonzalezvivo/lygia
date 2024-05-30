/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: <float3x3> rotate3dZ(<float> radians)
*/

#ifndef FNC_ROTATE3DZ
#define FNC_ROTATE3DZ
float3x3 rotate3dZ(const in float r){
    return float3x3(float3(cos(r),-sin(r),0.),
                    float3(sin(r),cos(r),0.),
                    float3(0.,0.,1.));
}
#endif
