/*
contributors:  Shadi El Hajj
description: Tangent-Bitangent-Normal Matrix
use: mat3 tbn(vec3 t, vec3 b, vec3 n)
license: MIT License (MIT) Copyright (c) 2024 Shadi El Hajj
*/

fn tbn(t: vec3f, b: vec3f, n: vec3f) -> mat3x3f {
    return mat3x3f(t, b, n);
}

fn tbn_up(n: vec3f, up: vec3f) -> mat3x3f {
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return tbn(t, b, n);
}
