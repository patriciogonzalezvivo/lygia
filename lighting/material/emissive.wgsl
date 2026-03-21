#include "../../color/space/gamma2linear.wgsl"
#include "../../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Get material emissive property from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines
use: vec4 materialEmissive()
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

uniform SAMPLER_TYPE MATERIAL_EMISSIVEMAP;

fn materialEmissive() -> vec3f {
    let emission = vec3f(0.0);

    let uv = v_texcoord.xy;
    uv += (MATERIAL_EMISSIVEMAP_OFFSET).xy;
    uv *= (MATERIAL_EMISSIVEMAP_SCALE).xy;
    emission = gamma2linear( SAMPLER_FNC(MATERIAL_EMISSIVEMAP, uv) ).rgb;

    emission = MATERIAL_EMISSIVE;

    return emission;
}
