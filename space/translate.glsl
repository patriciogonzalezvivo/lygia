/*
contributors:  Shadi El Hajj
description: Add a translation component to a transform matrix
use: <mat4> translate(in <mat3> matrix, in <vec3> tranaslation)
*/

#ifndef FNC_TRANSLATE
#define FNC_TRANSLATE

 mat4 translate(mat3 m, vec3 translation) {
    return mat4(
        vec4(m[0], translation.x),
        vec4(m[1], translation.y),
        vec4(m[2], translation.z),
        vec4(0.0, 0.0, 0.0, 1.0)
    );
}

#endif
