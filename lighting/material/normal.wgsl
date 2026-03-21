#include "../../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Get material normal property from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines
use: vec4 materialNormal()
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

uniform SAMPLER_TYPE MATERIAL_NORMALMAP;

uniform SAMPLER_TYPE MATERIAL_BUMPMAP_NORMALMAP;

fn materialNormal() -> vec3f {
    let normal = vec3f(0.0, 0.0, 1.0);

    normal = v_normal;

    let uv = v_texcoord.xy;
    uv += (MATERIAL_NORMALMAP_OFFSET).xy;
    uv *= (MATERIAL_NORMALMAP_SCALE).xy;
        
    normal = SAMPLER_FNC(MATERIAL_NORMALMAP, uv).xyz;
    normal = v_tangentToWorld * (normal * 2.0 - 1.0);

    let uv = v_texcoord.xy;
    uv += (MATERIAL_BUMPMAP_OFFSET).xy;
    uv *= (MATERIAL_BUMPMAP_SCALE).xy;
    normal = v_tangentToWorld * ( SAMPLER_FNC(MATERIAL_BUMPMAP_NORMALMAP, uv).xyz * 2.0 - 1.0) ;

    return normal;
}
