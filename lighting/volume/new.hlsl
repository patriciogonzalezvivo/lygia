#include "../volume.hlsl"

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
    _mat.scattering = float3(1.0, 1.0, 1.0);
    _mat.absorption = float3(1.0, 1.0, 1.0);
    _mat.sdf        = RAYMARCH_MAX_DIST;

}

Volume volumeNew() {
    Volume mat;
    volumeNew(mat);
    return mat;
}

Volume volumeNew(float3 scattering, float3 absorption, float sdf) {
    Volume mat = volumeNew();
    mat.scattering = scattering;
    mat.absorption = absorption;
    mat.sdf = sdf;
    return mat;
}

Volume volumeNew(float3 scattering, float sdf) {
    return volumeNew(scattering, float3(0.0, 0.0, 0.0), sdf);
}

#endif
