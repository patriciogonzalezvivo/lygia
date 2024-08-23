/*
contributors:  Shadi El Hajj
description: Add a translation component to a transform matrix
use: <mat4> translate(in <mat3> matrix, in <vec3> tranaslation)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef FNC_TRANSLATE
#define FNC_TRANSLATE

 mat4 translate(mat3 m, vec3 translation) {
    return mat4(
        vec4(m[0], 0.0),
        vec4(m[1], 0.0),
        vec4(m[2], 0.0),
        vec4(translation, 1.0)
    );
}

#endif
