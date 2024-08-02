/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
use: <mat4> rotate4dY(<float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE4DY
#define FNC_ROTATE4DY
mat4 rotate4dY(in float r){
    float c = cos(r);
    float s = sin(r);
    return mat4(vec4(c,0.,-s,0),
                vec4(0.,1.,0.,0.),
                vec4(s,0.,c,0.),
                vec4(0.,0.,0.,1.));
}
#endif
