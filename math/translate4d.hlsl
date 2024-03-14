/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 translate matrix
use: 
    - <float4x4> translate4d(<float3> t)
    - <float4x4> translate4d(<float> x, <float> y, <float> z)
*/

#ifndef FNC_TRANSLATE4D
float4x4 translate4d(float3 t) {
    return float4x4(1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    t.x, t.y, t.z, 1.0  );
}

float4x4 translate4d(float x, float y, float z) {
    return float4x4(1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    x,   y,   z, 1.0 );
}
#endif
