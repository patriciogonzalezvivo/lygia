#include "../toShininess.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: get material shininess property from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines 
use: vec4 materialShininess()
*/

#ifndef FNC_MATERIAL_SHININESS
#define FNC_MATERIAL_SHININESS

float materialShininess() {
    float shininess = 15.0;

#ifdef MATERIAL_SHININESS
    shininess = MATERIAL_SHININESS;

#elif defined(FNC_MATERIAL_METALLIC) && defined(MATERIAL_METALLIC) && defined(FNC_MATERIAL_ROUGHNESS) && defined(MATERIAL_ROUGHNESS)
    float roughness = materialRoughness();
    float metallic = materialMetallic();
    shininess = toShininess(roughness, metallic);
#endif

    return shininess;
}

#endif