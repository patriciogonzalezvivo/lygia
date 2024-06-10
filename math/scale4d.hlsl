/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 scale matrix
use:
    - <float4x4> scale4d(<float|float3|float4> radians)
    - <float4x4> scale4d(<float> x, <float> y, <float> z [, <float> w])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SCALE4D
float4x4 scale4d(float s) {
    return float4x4(s, 0.0, 0.0, 0.0,
                0.0, s, 0.0, 0.0,
                0.0, 0.0, s, 0.0,
                0.0, 0.0, 0.0, 1.0 );
}

float4x4 scale4d(float x, float y, float z) {
    return float4x4( x, 0.0, 0.0, 0.0,
                0.0,  y, 0.0, 0.0,
                0.0, 0.0,  z, 0.0,
                0.0, 0.0, 0.0, 1.0);
}

float4x4 scale4d(float x, float y, float z, float w) {
    return float4x4( x, 0.0, 0.0, 0.0,
                0.0,  y, 0.0, 0.0,
                0.0, 0.0,  z, 0.0,
                0.0, 0.0, 0.0,  w );
}

float4x4 scale4d(float3 s) {
    return float4x4(s.x, 0.0, 0.0, 0.0,
                0.0, s.y, 0.0, 0.0,
                0.0, 0.0, s.z, 0.0,
                0.0, 0.0, 0.0, 1.0 );
}

float4x4 scale4d(float4 s) {
    return float4x4(s.x, 0.0, 0.0, 0.0,
                0.0, s.y, 0.0, 0.0,
                0.0, 0.0, s.z, 0.0,
                0.0, 0.0, 0.0, s.w );
}
#endif
