/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: <mat3> rotate3dY(<float> radians)
*/

#ifndef FNC_ROTATE3DY
#define FNC_ROTATE3DY
mat3 rotate3dY(const in float theta){
    return mat3(vec3(cos(theta),0.,-sin(theta)),
                vec3(0.,1.,0.),
                vec3(sin(theta),0.,cos(theta)));
}
#endif
