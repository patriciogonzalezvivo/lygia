#include "../volume.glsl"

/*
contributors: Shadi El Hajj
description: |
    Volume Constructor.
use:
    - void volumeNew(out <volume> _mat)
    - <volume> volumeNew()
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef FNC_VOLUME_NEW
#define FNC_VOLUME_NEW

void volumeNew(out Volume _mat) {
    _mat.scattering   = vec3(1.0, 1.0, 1.0);
    _mat.absorption   = vec3(1.0, 1.0, 1.0);
    _mat.sdf     = RAYMARCH_MAX_DIST;

}

Volume volumeNew() {
    Volume mat;
    volumeNew(mat);
    return mat;
}

Volume volumeNew(vec3 scattering, vec3 absorption, float sdf) {
    Volume mat = volumeNew();
    mat.scattering = scattering;
    mat.absorption = absorption;
    mat.sdf = sdf;
    return mat;
}

Volume volumeNew(vec3 scattering, float sdf) {
    return volumeNew(scattering, vec3(0.0, 0.0, 0.0), sdf);
}

#endif
