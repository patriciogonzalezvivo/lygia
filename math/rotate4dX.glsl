/*
original_author: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
use: rotate4dX(<float> radians)
*/

#ifndef FNC_ROTATE4DX
#define FNC_ROTATE4DX
mat4 rotate4dX(in float phi){
    return mat4(
        vec4(1.,0.,0.,0),
        vec4(0.,cos(phi),-sin(phi),0.),
        vec4(0.,sin(phi),cos(phi),0.),
        vec4(0.,0.,0.,1.));
}
#endif
