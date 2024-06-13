/*
contributors: Patricio Gonzalez Vivo
description: given a 3x3 returns a 4x4
use: <float4x4> toMat4(<float4x4> m)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_TOMAT4
#define FNC_TOMAT4
float4x4 toMat4(float3x3 m) {
    return float4x4(float4(m[0], 0.0), 
                    float4(m[1], 0.0), 
                    float4(m[2], 0.0), 
                    float4(0.0, 0.0, 0.0, 1.0) );
}
#endif