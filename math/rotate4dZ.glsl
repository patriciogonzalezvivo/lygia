/*
original_author: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
use: rotate4dZ(<float> radians)
*/

#ifndef FNC_ROTATE4DZ
#define FNC_ROTATE4DZ
mat4 rotate4dZ(in float psi){
    return mat4(
        vec4(cos(psi),-sin(psi),0.,0),
        vec4(sin(psi),cos(psi),0.,0.),
        vec4(0.,0.,1.,0.),
        vec4(0.,0.,0.,1.));
}
#endif
