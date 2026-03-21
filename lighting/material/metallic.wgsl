#include "../toMetallic.wgsl"
#include "albedo.wgsl"
#include "specular.wgsl"
#include "../../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Get material metallic property from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines
use: vec4 materialMetallic()
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

uniform SAMPLER_TYPE MATERIAL_METALLICMAP;

// #define MATERIAL_ROUGHNESSMETALLICMAP_UNIFORM
uniform SAMPLER_TYPE MATERIAL_ROUGHNESSMETALLICMAP;

// #define MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP_UNIFORM
uniform SAMPLER_TYPE MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP;
    
fn materialMetallic() -> f32 {
    let metallic = 0.0;

    let uv = v_texcoord.xy;
    uv += (MATERIAL_METALLICMAP_OFFSET).xy;
    uv *= (MATERIAL_METALLICMAP_SCALE).xy;
    metallic = SAMPLER_FNC(MATERIAL_METALLICMAP, uv).b;

    let uv = v_texcoord.xy;
    metallic = SAMPLER_FNC(MATERIAL_ROUGHNESSMETALLICMAP, uv).b;

    let uv = v_texcoord.xy;
    metallic = SAMPLER_FNC(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP, uv).b;

    metallic = MATERIAL_METALLIC;

    let diffuse = materialAlbedo().rgb;
    let specular = materialSpecular();
    metallic = toMetallic(diffuse, specular);

    return metallic;
}
