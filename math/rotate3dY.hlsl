/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: <float3x3> rotate3dY(<float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE3DY
#define FNC_ROTATE3DY
float3x3 rotate3dY(const in float r){
    float c = cos(r);
    float s = sin(r);
    return float3x3(float3(c,0.,s),
                    float3(0.,1.,0.),
                    float3(-s,0.,c));
}
#endif
