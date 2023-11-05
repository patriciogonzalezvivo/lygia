#include "../../sample.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: get material normal property from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines 
use: float4 materialOcclusion()
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
*/

#ifndef FNC_MATERIAL_OCCLUSION
#define FNC_MATERIAL_OCCLUSION

#ifdef MATERIAL_OCCLUSIONMAP
uniform SAMPLER_TYPE MATERIAL_OCCLUSIONMAP;
#endif

#if defined(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP) && !defined(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP_UNIFORM)
#define MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP_UNIFORM
uniform SAMPLER_TYPE MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP;
#endif

float materialOcclusion() {
    float occlusion = 1.0;

#if defined(MATERIAL_OCCLUSIONMAP) && defined(MODEL_VERTEX_TEXCOORD)
    float2 uv = v_texcoord.xy;
    occlusion = SAMPLER_FNC(MATERIAL_OCCLUSIONMAP, uv).r;
#elif defined(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP) && defined(MODEL_VERTEX_TEXCOORD)
    float2 uv = v_texcoord.xy;
    occlusion = SAMPLER_FNC(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP, uv).r;
#endif

#if defined(MATERIAL_OCCLUSIONMAP_STRENGTH)
    occlusion *= MATERIAL_OCCLUSIONMAP_STRENGTH;
#endif

    return occlusion;
}

#endif