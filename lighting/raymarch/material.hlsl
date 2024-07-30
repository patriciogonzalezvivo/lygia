#include "normal.hlsl"
#include "cast.hlsl"
#include "ao.hlsl"
#include "softShadow.hlsl"
#include "../../math/sum.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Material Constructor. Designed to integrate with GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines
use:
    - void raymarchMaterial(in <float3> ro, in <float3> rd, out material _mat)
    - material raymarchMaterial(in <float3> ro, in <float3> rd)
options:
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_DIRECTION
    - LIGHT_COLOR
    - RAYMARCH_AMBIENT
    - RAYMARCH_MATERIAL_FNC(RAY, POSITION, NORMAL, ALBEDO)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef LIGHT_POSITION
#define LIGHT_POSITION float3(0.0, 10.0, -50.0)
#endif

#ifndef LIGHT_COLOR
#define LIGHT_COLOR float3(1.0, 1.0, 1.0)
#endif

#ifndef RAYMARCH_AMBIENT
#define RAYMARCH_AMBIENT float3(0.0, 0.0, 0.0)
#endif

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND float3(0.0, 0.0, 0.0)
#endif

#ifndef RAYMARCH_MATERIAL_FNC
#define RAYMARCH_MATERIAL_FNC raymarchDefaultMaterial
#endif

#ifndef FNC_RAYMARCHMATERIAL
#define FNC_RAYMARCHMATERIAL

Material raymarchDefaultMaterial(float3 ray, float3 position, float3 normal, Material material) {
    float3  env = RAYMARCH_AMBIENT;

    if (!material.valid)
    {
        material.albedo = float4(RAYMARCH_BACKGROUND, 0.0);
        return material;
    }

    float3 ref = reflect( ray, normal );
    float occ = raymarchAO( position, normal );

    #if defined(LIGHT_DIRECTION)
    float3  lig = normalize( LIGHT_DIRECTION );
    #else
    float3  lig = normalize( LIGHT_POSITION - position);
    #endif
    
    float3  hal = normalize( lig-ray );
    float amb = saturate( 0.5+0.5*normal.y );
    float dif = saturate( dot( normal, lig ) );
    float bac = saturate( dot( normal, normalize(float3(-lig.x, 0.0,-lig.z))) ) * saturate( 1.0-position.y );
    float dom = smoothstep( -0.1, 0.1, ref.y );
    float fre = pow( saturate(1.0+dot(normal,ray) ), 2.0 );
    
    dif *= raymarchSoftShadow( position, lig );
    dom *= raymarchSoftShadow( position, ref );

    float3 light = float3(0.0, 0.0, 0.0);
    light += 1.30 * dif * LIGHT_COLOR;
    light += 0.40 * amb * occ * env;
    light += 0.50 * dom * occ * env;
    light += 0.50 * bac * occ * 0.25;
    light += 0.25 * fre * occ;

    material.albedo.rgb *= light;
    
    return material;
}

#endif