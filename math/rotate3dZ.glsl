/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: <mat3> rotate3dZ(<float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE3DZ
#define FNC_ROTATE3DZ
mat3 rotate3dZ(const in float r){
    return mat3(vec3(cos(r),-sin(r),0.),
                vec3(sin(r),cos(r),0.),
                vec3(0.,0.,1.));
}
#endif
