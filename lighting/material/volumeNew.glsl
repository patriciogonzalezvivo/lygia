#include "../volumeMaterial.glsl"

/*
contributors: Shadi El Hajj
description: |
    Volume Material Constructor.
use:
    - void volumeMaterialNew(out <volumeMaterial> _mat)
    - <volumeMaterial> volumeMaterialNew()
*/

#ifndef FNC_VOLUME_MATERIAL_NEW
#define FNC_VOLUME_MATERIAL_NEW

void volumeMaterialNew(out VolumeMaterial _mat) {
    _mat.color   = vec3(1.0, 1.0, 1.0);
    _mat.density = 1.0;
    _mat.sdf     = RAYMARCH_MAX_DIST;

}

VolumeMaterial volumeMaterialNew() {
    VolumeMaterial mat;
    volumeMaterialNew(mat);
    return mat;
}

VolumeMaterial volumeMaterialNew(vec3 color, float sdf) {
    VolumeMaterial mat = volumeMaterialNew();
    mat.color.rgb = color;
    mat.sdf = sdf;
    return mat;
}

VolumeMaterial volumeMaterialNew(vec3 color, float density, float sdf) {
    VolumeMaterial mat = volumeMaterialNew();
    mat.color.rgb = color;
    mat.density = density;
    mat.sdf = sdf;
    return mat;
}

#endif
