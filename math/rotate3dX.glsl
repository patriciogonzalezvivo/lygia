/*
original_author: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: rotate3dX(<float> radians)
*/

#ifndef FNC_ROTATE3DX
#define FNC_ROTATE3DX
mat3 rotate3dX(in float phi){
    return mat3(vec3(1.0,0.0,0.0),
                vec3(0.0,cos(phi),-sin(phi)),
                vec3(0.0,sin(phi),cos(phi)));
}
#endif
