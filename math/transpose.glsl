/*
original_author: Mikola Lysenko
description: transpose matrixes
use: <mat3|mat4> transpose(in <mat3|mat4> m)
*/

#if !defined(FNC_TRANSPOSE) && (__VERSION__ < 120)
#define FNC_TRANSPOSE

mat3 transpose(in mat3 m) {
    vec3 i0 = m[0];
    vec3 i1 = m[1];
    vec3 i2 = m[2];

    return mat3(
                    vec3(i0.x, i1.x, i2.x),
                    vec3(i0.y, i1.y, i2.y),
                    vec3(i0.z, i1.z, i2.z)
                );
}

mat4 transpose(in mat4 m) {
    vec4 i0 = m[0];
    vec4 i1 = m[1];
    vec4 i2 = m[2];
    vec4 i3 = m[3];

    return mat4(
                    vec4(i0.x, i1.x, i2.x, i3.x),
                    vec4(i0.y, i1.y, i2.y, i3.y),
                    vec4(i0.z, i1.z, i2.z, i3.z),
                    vec4(i0.w, i1.w, i2.w, i3.w)
                );
}

#endif