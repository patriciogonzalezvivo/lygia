#include "../../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Get material specular property from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines
use: vec4 materialMetallic()
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - MATERIAL_SPECULARMAP
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

uniform SAMPLER_TYPE MATERIAL_SPECULARMAP;

fn materialSpecular() -> vec3f {
    let spec = vec3f(0.04);
    let uv = v_texcoord.xy;
    uv += (MATERIAL_SPECULARMAP_OFFSET).xy;
    uv *= (MATERIAL_SPECULARMAP_SCALE).xy;
    spec = SAMPLER_FNC(MATERIAL_SPECULARMAP, uv).rgb;
    spec = MATERIAL_SPECULAR;
    return spec;
}
