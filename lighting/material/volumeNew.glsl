#include "../volumeMaterial.glsl"

/*
contributors: Shadi El Hajj
description: |
    Volume Material Constructor.
use:
    - void volumeMaterialNew(out <volumeMaterial> _mat)
    - <volumeMaterial> volumeMaterialNew()
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef FNC_VOLUME_MATERIAL_NEW
#define FNC_VOLUME_MATERIAL_NEW

void volumeMaterialNew(out VolumeMaterial _mat) {
    _mat.absorption   = vec3(1.0, 1.0, 1.0);
    _mat.scattering   = vec3(1.0, 1.0, 1.0);
    _mat.sdf     = RAYMARCH_MAX_DIST;

}

VolumeMaterial volumeMaterialNew() {
    VolumeMaterial mat;
    volumeMaterialNew(mat);
    return mat;
}

VolumeMaterial volumeMaterialNew(vec3 absorption, vec3 scattering, float sdf) {
    VolumeMaterial mat = volumeMaterialNew();
    mat.absorption = absorption;
    mat.scattering = scattering;
    mat.sdf = sdf;
    return mat;
}

#endif
