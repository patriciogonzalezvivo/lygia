/*
original_author: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: rotate3dZ(<float> radians)
*/

#ifndef FNC_ROTATE3DZ
#define FNC_ROTATE3DZ
mat3 rotate3dZ(in float psi){
    return mat3(vec3(cos(psi),-sin(psi),0.),
                vec3(sin(psi),cos(psi),0.),
                vec3(0.,0.,1.));
}
#endif
