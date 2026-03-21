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

fn materialZero(_mat: Material) {
    _mat.albedo = vec4f(0.0);
    _mat.emissive = vec3f(0.0);
    _mat.position = vec3f(0.0);
    _mat.normal = vec3f(0.0);
    _mat.sdf = 0.0;
    _mat.valid = true;

    _mat.normal_back = vec3f(0.0);
    _mat.ior = vec3f(0.0);
    _mat.roughness = 0.0;
    _mat.metallic = 0.0;
    _mat.reflectance = 0.0;
    _mat.ambientOcclusion = 0.0;

    _mat.clearCoat = 0.0;
    _mat.clearCoatRoughness = 0.0;
    _mat.clearCoatNormal = vec3f(0.0);

    _mat.thickness          = 0.0;

    _mat.subsurfaceColor    = vec3f(0.0);
    _mat.subsurfacePower    = 0.0;
    _mat.subsurfaceThickness = 0.0;
}

Material materialZero() {
    Material mat;
    materialZero(mat);
    return mat;
}
