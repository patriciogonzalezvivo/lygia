/*
contributors: Patricio Gonzalez Vivo
description: |
    Material Constructor with all the values set to zero.
use:
    - void materialZero(out <material> _mat)
    - <material> materialZero()
options:
    - SURFACE_POSITION
    - SCENE_BACK_SURFACE
    - SHADING_MODEL_CLEAR_COAT
    - MATERIAL_HAS_CLEAR_COAT_NORMAL
    - SHADING_MODEL_IRIDESCENCE
    - SHADING_MODEL_SUBSURFACE
    - SHADING_MODEL_CLOTH
    - SHADING_MODEL_SPECULAR_GLOSSINESS
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MATERIAL_ZERO
#define FNC_MATERIAL_ZERO

void materialZero(out Material _mat) {
    _mat.albedo = vec4(0.0);
    _mat.emissive = vec3(0.0);
    _mat.position = vec3(0.0);
    _mat.normal = vec3(0.0);
#if defined(RENDER_RAYMARCHING)
    _mat.sdf = 0.0;
    _mat.valid = true;
#endif

#if defined(SCENE_BACK_SURFACE)
    _mat.normal_back = vec3(0.0);
#endif
    _mat.ior = vec3(0.0);
    _mat.roughness = 0.0;
    _mat.metallic = 0.0;
    _mat.reflectance = 0.0;
    _mat.ambientOcclusion = 0.0;

#if defined (SHADING_MODEL_CLEAR_COAT)
    _mat.clearCoat = 0.0;
    _mat.clearCoatRoughness = 0.0;
    #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    _mat.clearCoatNormal = vec3(0.0);
    #endif
#endif

#if defined(SHADING_MODEL_IRIDESCENCE)
    _mat.thickness          = 0.0;
#endif

#if defined(SHADING_MODEL_SUBSURFACE)
    _mat.subsurfaceColor    = vec3(0.0);
    _mat.subsurfacePower    = 0.0;
    _mat.subsurfaceThickness = 0.0;
#endif
}

Material materialZero() {
    Material mat;
    materialZero(mat);
    return mat;
}

#endif
