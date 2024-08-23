#include "../medium.hlsl"

/*
contributors: Shadi El Hajj
description: |
    Medium Constructor.
use:
    - void mediumNew(out <medium> _mat)
    - <medium> mediumNew()
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef FNC_MEDIUM_NEW
#define FNC_MEDIUM_NEW

void mediumNew(out Medium _mat) {
    _mat.scattering = float3(1.0, 1.0, 1.0);
    _mat.absorption = float3(1.0, 1.0, 1.0);
    _mat.sdf        = RAYMARCH_MAX_DIST;

}

Medium mediumNew() {
    Medium mat;
    mediumNew(mat);
    return mat;
}

Medium mediumNew(float3 scattering, float3 absorption, float sdf) {
    Medium mat = mediumNew();
    mat.scattering = scattering;
    mat.absorption = absorption;
    mat.sdf = sdf;
    return mat;
}

Medium mediumNew(float3 scattering, float sdf) {
    return mediumNew(scattering, float3(0.0, 0.0, 0.0), sdf);
}

#endif
