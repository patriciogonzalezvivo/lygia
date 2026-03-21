#include "../medium.wgsl"

/*
contributors: Shadi El Hajj
description: |
    Medium Constructor.
use:
    - void mediumNew(out <medium> _mat)
    - <medium> mediumNew()
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

fn mediumNew(_mat: Medium) {
    _mat.scattering   = vec3f(1.0, 1.0, 1.0);
    _mat.absorption   = vec3f(1.0, 1.0, 1.0);
    _mat.sdf     = RAYMARCH_MAX_DIST;

}

Medium mediumNew() {
    Medium mat;
    mediumNew(mat);
    return mat;
}

Medium mediumNew(vec3 scattering, vec3 absorption, float sdf) {
    Medium mat = mediumNew();
    mat.scattering = scattering;
    mat.absorption = absorption;
    mat.sdf = sdf;
    return mat;
}

Medium mediumNew(vec3 scattering, float sdf) {
    return mediumNew(scattering, vec3f(0.0, 0.0, 0.0), sdf);
}
