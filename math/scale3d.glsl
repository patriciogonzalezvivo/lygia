/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 scale matrix
use:
    - <mat3> scale3d(<float|vec3> radians)
    - <mat3> scale3d(<float> x, <float> y, <float> z)
*/

#ifndef FNC_SCALE3D
mat3 scale3d(float s) {
    return mat3(s, 0.0, 0.0,
                0.0, s, 0.0,
                0.0, 0.0, s );
}

mat3 scale3d(float x, float y, float z) {
    return mat3(  x, 0.0, 0.0,
                0.0,  y, 0.0,
                0.0, 0.0,  z );
}

mat3 scale3d(vec3 s) {
    return mat3(s.x, 0.0, 0.0,
                0.0, s.y, 0.0,
                0.0, 0.0, s.z );
}
#endif
