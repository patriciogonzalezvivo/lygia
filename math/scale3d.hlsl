/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 scale matrix
use:
    - <float3x3> scale3d(<float|float3> radians)
    - <float3x3> scale3d(<float> x, <float> y, <float> z)
*/

#ifndef FNC_SCALE3D
float3x3 scale3d(float s) {
    return float3x3(s, 0.0, 0.0,
                    0.0, s, 0.0,
                    0.0, 0.0, s );
}

float3x3 scale3d(float x, float y, float z) {
    return float3x3(  x, 0.0, 0.0,
                    0.0,  y, 0.0,
                    0.0, 0.0,  z );
}

float3x3 scale3d(float3 s) {
    return float3x3(s.x, 0.0, 0.0,
                    0.0, s.y, 0.0,
                    0.0, 0.0, s.z );
}
#endif
