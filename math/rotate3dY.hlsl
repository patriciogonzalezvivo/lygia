/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: <float3x3> rotate3dY(<float> radians)
*/

#ifndef FNC_ROTATE3DY
#define FNC_ROTATE3DY
float3x3 rotate3dY(const in float theta){
    return float3x3(float3(cos(theta),0.,-sin(theta)),
                    float3(0.,1.,0.),
                    float3(sin(theta),0.,cos(theta)));
}
#endif
