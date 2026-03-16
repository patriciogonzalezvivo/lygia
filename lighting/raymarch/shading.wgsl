#include "normal.wgsl"
#include "cast.wgsl"
#include "ao.wgsl"
#include "softShadow.wgsl"
#include "../shadingData/new.wgsl"
#include "../../math/saturate.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Material Constructor. Designed to integrate with GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines
use:
    - void raymarchMaterial(in <vec3> ro, in <vec3> rd, out material _mat)
    - material raymarchMaterial(in <vec3> ro, in <vec3> rd)
options:
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_DIRECTION
    - LIGHT_COLOR
    - RAYMARCH_AMBIENT
    - RAYMARCH_SHADING_FNC(RAY, POSITION, NORMAL, ALBEDO)
examples:
    - /shaders/lighting_raymarching.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define LIGHT_POSITION vec3(0.0, 10.0, -50.0)

// #define LIGHT_COLOR vec3(1.0, 1.0, 1.0)

// #define RAYMARCH_AMBIENT vec3(1.0, 1.0, 1.0)

// #define RAYMARCH_SHADING_FNC raymarchDefaultShading

fn raymarchDefaultShading(m: Material, shadingData: ShadingData) -> vec4f {
    
    // This are here to be access by RAYMARCH_AMBIENT 
    let worldNormal = m.normal;
    let worldPosition = m.position;

    let lig = normalize(LIGHT_DIRECTION);
    let lig = normalize(LIGHT_POSITION - m.position);
    
    let ref = reflect(-shadingData.V, m.normal);
    let occ = raymarchAO(m.position, m.normal);

    let hal = normalize(lig + shadingData.V);
    let amb = saturate(0.5 + 0.5 * m.normal.y);
    let dif = saturate(dot(m.normal, lig));
    let bac = saturate(dot(m.normal, normalize(vec3f(-lig.x, 0.0, -lig.z)))) * saturate(1.0 - m.position.y);
    let dom = smoothstep( -0.1, 0.1, ref.y );
    let fre = pow(saturate(1.0 + dot(m.normal, -shadingData.V)), 2.0);
    
    dif *= raymarchSoftShadow(m.position, lig);
    dom *= raymarchSoftShadow(m.position, ref);

    let env = RAYMARCH_AMBIENT;
    let shade = vec3f(0.0, 0.0, 0.0);
    shade += 1.30 * dif * LIGHT_COLOR;
    shade += 0.40 * amb * occ * env;
    shade += 0.50 * dom * occ * env;
    shade += 0.50 * bac * occ * 0.25;
    shade += 0.25 * fre * occ;

    return vec4f(m.albedo.rgb * shade, m.albedo.a);
}
