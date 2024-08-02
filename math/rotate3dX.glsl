/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: <mat3> rotate3dX(<float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE3DX
#define FNC_ROTATE3DX
mat3 rotate3dX(const in float r){
    float c = cos(r);
    float s = sin(r);
    return mat3(vec3(1.0,0.0,0.0),
                vec3(0.0,c,s),
                vec3(0.0,-s,c));
}
#endif
