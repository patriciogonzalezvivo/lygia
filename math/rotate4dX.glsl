/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
use: <mat4> rotate4dX(<float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE4DX
#define FNC_ROTATE4DX
mat4 rotate4dX(const in float r){
    float c = cos(r);
    float s = sin(r);
    return mat4(vec4(1.,0.,0.,0),
                vec4(0.,c,s,0.),
                vec4(0.,-s,c,0.),
                vec4(0.,0.,0.,1.));
}
#endif
