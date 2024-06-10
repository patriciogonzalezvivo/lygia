/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 translate matrix
use:
    - <mat4> translate4d(<vec3> t)
    - <mat4> translate4d(<float> x, <float> y, <float> z)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_TRANSLATE4D
mat4 translate4d(vec3 t) {
    return mat4(1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                t.x, t.y, t.z, 1.0 );
}

mat4 translate4d(float x, float y, float z) {
    return mat4(1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                  x,   y,   z, 1.0 );
}
#endif
