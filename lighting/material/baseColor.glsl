#include "../../color/space/gamma2linear.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: get material BaseColor from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines 
use: vec4 materialBaseColor()
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef FNC_MATERIAL_BASECOLOR
#define FNC_MATERIAL_BASECOLOR

#ifdef MATERIAL_BASECOLORMAP
uniform sampler2D MATERIAL_BASECOLORMAP;
#endif

vec4 materialBaseColor() {
    vec4 base = vec4(1.0);
    
#if defined(MATERIAL_BASECOLORMAP) && defined(MODEL_VERTEX_TEXCOORD)
    vec2 uv = v_texcoord.xy;
    #if defined(MATERIAL_BASECOLORMAP_OFFSET)
    uv += (MATERIAL_BASECOLORMAP_OFFSET).xy;
    #endif
    #if defined(MATERIAL_BASECOLORMAP_SCALE)
    uv *= (MATERIAL_BASECOLORMAP_SCALE).xy;
    #endif
    base = gamma2linear( SAMPLER_FNC(MATERIAL_BASECOLORMAP, uv) );
    
#elif defined(MATERIAL_BASECOLOR)
    base = MATERIAL_BASECOLOR;

#endif

#if defined(MODEL_VERTEX_COLOR)
    base *= v_color;
#endif

    return base;
}

#endif