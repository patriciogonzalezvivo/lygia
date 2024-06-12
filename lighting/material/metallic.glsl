#include "../toMetallic.glsl"
#include "albedo.glsl"
#include "specular.glsl"
#include "../../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Get material metalic property from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines
use: vec4 materialMetallic()
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MATERIAL_METALLIC
#define FNC_MATERIAL_METALLIC

#ifdef MATERIAL_METALLICMAP
uniform SAMPLER_TYPE MATERIAL_METALLICMAP;
#endif

#if defined(MATERIAL_ROUGHNESSMETALLICMAP) && !defined(MATERIAL_ROUGHNESSMETALLICMAP_UNIFORM)
#define MATERIAL_ROUGHNESSMETALLICMAP_UNIFORM
uniform SAMPLER_TYPE MATERIAL_ROUGHNESSMETALLICMAP;
#endif

#if defined(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP) && !defined(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP_UNIFORM)
#define MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP_UNIFORM
uniform SAMPLER_TYPE MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP;
#endif
    
float materialMetallic() {
    float metallic = 0.0;

#if defined(MATERIAL_METALLICMAP) && defined(MODEL_VERTEX_TEXCOORD)
    vec2 uv = v_texcoord.xy;
    #if defined(MATERIAL_METALLICMAP_OFFSET)
    uv += (MATERIAL_METALLICMAP_OFFSET).xy;
    #endif
    #if defined(MATERIAL_METALLICMAP_SCALE)
    uv *= (MATERIAL_METALLICMAP_SCALE).xy;
    #endif
    metallic = SAMPLER_FNC(MATERIAL_METALLICMAP, uv).b;

#elif defined(MATERIAL_ROUGHNESSMETALLICMAP) && defined(MODEL_VERTEX_TEXCOORD)
    vec2 uv = v_texcoord.xy;
    metallic = SAMPLER_FNC(MATERIAL_ROUGHNESSMETALLICMAP, uv).b;

#elif defined(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP) && defined(MODEL_VERTEX_TEXCOORD)
    vec2 uv = v_texcoord.xy;
    metallic = SAMPLER_FNC(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP, uv).b;

#elif defined(MATERIAL_METALLIC)
    metallic = MATERIAL_METALLIC;

#else
    vec3 diffuse = materialAlbedo().rgb;
    vec3 specular = materialSpecular();
    metallic = toMetallic(diffuse, specular);
#endif

    return metallic;
}

#endif