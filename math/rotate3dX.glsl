/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: <mat3> rotate3dX(<float> radians)
*/

#ifndef FNC_ROTATE3DX
#define FNC_ROTATE3DX
mat3 rotate3dX(const in float r){
    return mat3(vec3(1.0,0.0,0.0),
                vec3(0.0,cos(r),-sin(r)),
                vec3(0.0,sin(r),cos(r)));
}
#endif
