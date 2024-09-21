/*
contributors:  Shadi El Hajj
description: Tangent-Bitangent-Normal Matrix
use: mat3 tbn(vec3 t, vec3 b, vec3 n)
license: MIT License (MIT) Copyright (c) 2024 Shadi El Hajj
*/

#ifndef FNC_TBN
#define FNC_TBN

mat3 tbn(vec3 t, vec3 b, vec3 n) {
    return mat3(t, b, n);
}

mat3 tbn(vec3 n, vec3 up) {
    vec3 t = normalize(cross(up, n));
    vec3 b = cross(n, t);
    return tbn(t, b, n);
}

#endif