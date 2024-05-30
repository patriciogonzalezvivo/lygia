/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
use: <mat4> rotate4dY(<float> radians)
*/

#ifndef FNC_ROTATE4DY
#define FNC_ROTATE4DY
mat4 rotate4dY(in float r){
    return mat4(vec4(cos(r),0.,-sin(r),0),
                vec4(0.,1.,0.,0.),
                vec4(sin(r),0.,cos(r),0.),
                vec4(0.,0.,0.,1.));
}
#endif
