/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
use: <float4x4> rotate4dX(<float> radians)
*/

#ifndef FNC_ROTATE4DX
#define FNC_ROTATE4DX
float4x4 rotate4dX(const in float phi){
    return float4x4(float4(1.,0.,0.,0),
                    float4(0.,cos(phi),-sin(phi),0.),
                    float4(0.,sin(phi),cos(phi),0.),
                    float4(0.,0.,0.,1.));
}
#endif
