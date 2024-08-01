#include "normal.glsl"
#include "cast.glsl"
#include "ao.glsl"
#include "softShadow.glsl"
#include "../../math/saturate.glsl"

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
    - RAYMARCH_MATERIAL_FNC(RAY, POSITION, NORMAL, ALBEDO)
examples:
    - /shaders/lighting_raymarching.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef LIGHT_POSITION
#define LIGHT_POSITION vec3(0.0, 10.0, -50.0)
#endif

#ifndef LIGHT_COLOR
#define LIGHT_COLOR vec3(1.0, 1.0, 1.0)
#endif

#ifndef RAYMARCH_AMBIENT
#define RAYMARCH_AMBIENT float3(1.0, 1.0, 1.0)
#endif

#ifndef RAYMARCH_MATERIAL_FNC
#define RAYMARCH_MATERIAL_FNC raymarchDefaultMaterial
#endif

#ifndef FNC_RAYMARCHMATERIAL
#define FNC_RAYMARCHMATERIAL

vec4 raymarchDefaultMaterial(vec3 ray, vec3 position, vec3 normal, RAYMARCH_MAP_MATERIAL_TYPE color) {
    vec3  env = RAYMARCH_AMBIENT;

    vec3 ref = reflect(-m.V, m.normal);
    float occ = raymarchAO(m.position, m.normal);

    #if defined(LIGHT_DIRECTION)
    vec3 lig = normalize( LIGHT_DIRECTION );
    #else
    vec3 lig = normalize(LIGHT_POSITION - m.position);
    #endif
    
    vec3 hal = normalize(lig + m.V);
    float amb = saturate(0.5 + 0.5 * m.normal.y);
    float dif = saturate(dot(m.normal, lig));
    float bac = saturate(dot(m.normal, normalize(vec3(-lig.x, 0.0, -lig.z)))) * saturate(1.0 - m.position.y);
    float dom = smoothstep( -0.1, 0.1, ref.y );
    float fre = pow(saturate(1.0 + dot(m.normal, -m.V)), 2.0);
    
    dif *= raymarchSoftShadow(m.position, lig);
    dom *= raymarchSoftShadow(m.position, ref);

    vec3 light = vec3(0.0, 0.0, 0.0);
    light += 1.30 * dif * LIGHT_COLOR;
    light += 0.40 * amb * occ * env;
    light += 0.50 * dom * occ * env;
    light += 0.50 * bac * occ * 0.25;
    light += 0.25 * fre * occ;

    return vec4(m.albedo.rgb * light, m.albedo.a);
}

#endif