#include "../toShininess.hlsl"
#include "roughness.hlsl"
#include "metallic.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Get material shininess property from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines
use: float4 materialShininess()
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MATERIAL_SHININESS
#define FNC_MATERIAL_SHININESS

float materialShininess() {
    float shininess = 15.0;

#ifdef MATERIAL_SHININESS
    shininess = MATERIAL_SHININESS;

#elif defined(MATERIAL_METALLIC) && defined(MATERIAL_ROUGHNESS)
    float roughness = materialRoughness();
    float metallic = materialMetallic();

    shininess = toShininess(roughness, metallic);
#endif
    return shininess;
}

#endif