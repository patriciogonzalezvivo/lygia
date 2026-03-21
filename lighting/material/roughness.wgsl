#include "../../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Get material roughness property from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines
use: vec4 materialRoughness()
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

uniform SAMPLER_TYPE MATERIAL_ROUGHNESSMAP;

// #define MATERIAL_ROUGHNESSMETALLICMAP_UNIFORM
uniform SAMPLER_TYPE MATERIAL_ROUGHNESSMETALLICMAP;

// #define MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP_UNIFORM
uniform SAMPLER_TYPE MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP;

fn materialRoughness() -> f32 {
    let roughness = 0.05;

    let uv = v_texcoord.xy;
    uv += (MATERIAL_ROUGHNESSMAP_OFFSET).xy;
    uv *= (MATERIAL_ROUGHNESSMAP_SCALE).xy;
    roughness = max(roughness, SAMPLER_FNC(MATERIAL_ROUGHNESSMAP, uv).g);

    let uv = v_texcoord.xy;
    roughness = max(roughness, SAMPLER_FNC(MATERIAL_ROUGHNESSMETALLICMAP, uv).g);

    let uv = v_texcoord.xy;
    roughness = max(roughness, SAMPLER_FNC(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP, uv).g);

    roughness = MATERIAL_ROUGHNESS;

    return roughness;
}
