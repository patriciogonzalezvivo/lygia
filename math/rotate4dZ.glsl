/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
use: <mat4> rotate4dZ(<float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE4DZ
#define FNC_ROTATE4DZ
mat4 rotate4dZ(in float r){
    float c = cos(r);
    float s = sin(r);
    return mat4(vec4(cos(r),sin(r),0.,0),
                vec4(-sin(r),cos(r),0.,0.),
                vec4(0.,0.,1.,0.),
                vec4(0.,0.,0.,1.));
}
#endif
