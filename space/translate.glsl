/*
contributors:  Shadi El Hajj
description: Add a translation component to a transform matrix
use: <mat4> translate(in <mat3> matrix, in <vec3> tranaslation)
*/

#ifndef FNC_TRANSLATE
#define FNC_TRANSLATE

 mat4 translate(mat3 m, vec3 translation) {
    mat4 m4 = mat4(m);
    m4[0][3] = translation.x;
    m4[1][3] = translation.y;
    m4[2][3] = translation.z;
    m4[3][3] = 1.0;
    return m4;
}

#endif
