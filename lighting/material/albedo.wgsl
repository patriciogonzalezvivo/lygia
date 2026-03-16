#include "../../color/space/gamma2linear.wgsl"
#include "../../sampler.wgsl"

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

uniform SAMPLER_TYPE MATERIAL_BASECOLORMAP;

uniform SAMPLER_TYPE MATERIAL_ALBEDOMAP;

fn materialAlbedo() -> vec4f {
    let albedo = vec4f(0.5, 0.5, 0.5, 1.0);
    
    let uv = v_texcoord.xy;
    uv += (MATERIAL_BASECOLORMAP_OFFSET).xy;
    uv *= (MATERIAL_BASECOLORMAP_SCALE).xy;
    albedo = gamma2linear( SAMPLER_FNC(MATERIAL_BASECOLORMAP, uv) );

    let uv = v_texcoord.xy;
    uv += (MATERIAL_ALBEDOMAP_OFFSET).xy;
    uv *= (MATERIAL_ALBEDOMAP_SCALE).xy;
    albedo = gamma2linear( SAMPLER_FNC(MATERIAL_ALBEDOMAP, uv) );

    albedo = MATERIAL_BASECOLOR;

    albedo = MATERIAL_ALBEDO;

    albedo *= v_color;

    return albedo;
}
