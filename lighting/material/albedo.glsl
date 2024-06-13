#include "../../color/space/gamma2linear.glsl"
#include "../../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Get material BaseColor from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines
use: vec4 materialAlbedo()
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MATERIAL_ALBEDO
#define FNC_MATERIAL_ALBEDO

#ifdef MATERIAL_BASECOLORMAP
uniform SAMPLER_TYPE MATERIAL_BASECOLORMAP;
#endif

#ifdef MATERIAL_ALBEDOMAP
uniform SAMPLER_TYPE MATERIAL_ALBEDOMAP;
#endif

vec4 materialAlbedo() {
    vec4 albedo = vec4(0.5, 0.5, 0.5, 1.0);
    
#if defined(MATERIAL_BASECOLORMAP) && defined(MODEL_VERTEX_TEXCOORD)
    vec2 uv = v_texcoord.xy;
    #if defined(MATERIAL_BASECOLORMAP_OFFSET)
    uv += (MATERIAL_BASECOLORMAP_OFFSET).xy;
    #endif
    #if defined(MATERIAL_BASECOLORMAP_SCALE)
    uv *= (MATERIAL_BASECOLORMAP_SCALE).xy;
    #endif
    albedo = gamma2linear( SAMPLER_FNC(MATERIAL_BASECOLORMAP, uv) );

#elif defined(MATERIAL_ALBEDOMAP) && defined(MODEL_VERTEX_TEXCOORD)
    vec2 uv = v_texcoord.xy;
    #if defined(MATERIAL_ALBEDOMAP_OFFSET)
    uv += (MATERIAL_ALBEDOMAP_OFFSET).xy;
    #endif
    #if defined(MATERIAL_ALBEDOMAP_SCALE)
    uv *= (MATERIAL_ALBEDOMAP_SCALE).xy;
    #endif
    albedo = gamma2linear( SAMPLER_FNC(MATERIAL_ALBEDOMAP, uv) );

#elif defined(MATERIAL_BASECOLOR)
    albedo = MATERIAL_BASECOLOR;

#elif defined(MATERIAL_ALBEDO)
    albedo = MATERIAL_ALBEDO;

#endif

#if defined(MODEL_VERTEX_COLOR)
    albedo *= v_color;
#endif

    return albedo;
}

#endif