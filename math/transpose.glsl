/*
contributors: Mikola Lysenko
description: transpose matrixes
use: <mat3|mat4> transpose(in <mat3|mat4> m)
*/

#if !defined(FNC_TRANSPOSE) && (__VERSION__ < 120)
#define FNC_TRANSPOSE
mat3 transpose(in mat3 m) {
    return mat3(    m[0][0], m[1][0], m[2][0],
                    m[0][1], m[1][1], m[2][1],
                    m[0][2], m[1][2], m[2][2] );
}

mat4 transpose(in mat4 m) {
    return mat4(    vec4(m[0][0], m[1][0], m[2][0], m[3][0]),
                    vec4(m[0][1], m[1][1], m[2][1], m[3][1]),
                    vec4(m[0][2], m[1][2], m[2][2], m[3][2]),
                    vec4(m[0][3], m[1][3], m[2][3], m[3][3])    );
}
#endif