/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: <float3x3> rotate3dZ(<float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE3DZ
#define FNC_ROTATE3DZ
float3x3 rotate3dZ(const in float r){
    float c = cos(r);
    float s = sin(r);
    return float3x3(float3(c,-s,0.),
                    float3(s,c,0.),
                    float3(0.,0.,1.));
}
#endif
