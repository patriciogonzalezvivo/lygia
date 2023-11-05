#include "normal.hlsl"
#include "cast.hlsl"
#include "ao.hlsl"
#include "softShadow.hlsl"
#include "../../math/sum.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Material Constructor. Designed to integrate with GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines 
use: 
    - void raymarchMaterial(in <float3> ro, in <float3> rd, out material _mat)
    - material raymarchMaterial(in <float3> ro, in <float3> rd)
options:
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_DIRECTION
    - LIGHT_COLOR
    - RAYMARCH_AMBIENT
    - RAYMARCH_MATERIAL_FNC(RAY, POSITION, NORMAL, ALBEDO)
*/

#ifndef LIGHT_POSITION
#define LIGHT_POSITION float3(0.0, 10.0, -50.0)
#endif

#ifndef LIGHT_COLOR
#define LIGHT_COLOR float3(0.5, 0.5, 0.5)
#endif

#ifndef RAYMARCH_AMBIENT
#define RAYMARCH_AMBIENT float3(1.0, 1.0, 1.0)
#endif

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND float3(0.0, 1.0, 1.0)
#endif

#ifndef RAYMARCH_MAP_MATERIAL_TYPE
#define RAYMARCH_MAP_MATERIAL_TYPE float3
#endif

#ifndef RAYMARCH_MAP_MATERIAL_FNC
#define RAYMARCH_MAP_MATERIAL_FNC 
#endif

#ifndef RAYMARCH_MATERIAL_FNC
#define RAYMARCH_MATERIAL_FNC(RAY, POSITION, NORMAL, ALBEDO) raymarchDefaultMaterial(RAY, POSITION, NORMAL, ALBEDO)
#endif

#ifndef FNC_RAYMARCHMATERIAL
#define FNC_RAYMARCHMATERIAL

float3 raymarchDefaultMaterial(float3 ray, float3 position, float3 normal, RAYMARCH_MAP_MATERIAL_TYPE color) {
    float3  env = RAYMARCH_AMBIENT;

    if ( sum(color) <= 0.0 ) 
        return RAYMARCH_BACKGROUND;

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
    
    dif *= raymarchSoftShadow( position, lig, 0.02, 2.5 );
    dom *= raymarchSoftShadow( position, ref, 0.02, 2.5 );

    float3 light = float3(0.0, 0.0, 0.0);
    light += 1.30 * dif * LIGHT_COLOR;
    light += 0.40 * amb * occ * env;
    light += 0.50 * dom * occ * env;
    light += 0.50 * bac * occ * 0.25;
    light += 0.25 * fre * occ;

    return RAYMARCH_MAP_MATERIAL_FNC(color) * light;
}

float3 raymarchMaterial(float3 ray, float3 position, float3 normal, float3 albedo) {
    return RAYMARCH_MATERIAL_FNC(ray, position, normal, albedo);
}

#if defined(STR_MATERIAL)
void raymarchMaterial( in float3 ro, in float3 rd, inout Material mat) { 
    RAYMARCH_MAP_TYPE res = raymarchCast(ro, rd);

    float3 col = float3(0.0, 0.0, 0.0);
    float3 m = res.rgb;
    float t = res.a;

    float3 pos = ro + t * rd;
    float3 nor = raymarchNormal( pos );
    float occ = raymarchAO( pos, nor );

    mat.albedo = res;
    mat.normal = nor;
    mat.ambientOcclusion = occ;

    #if defined(SHADING_SHADOWS)
    float3 ref = reflect( rd, nor );
    float3 lig = normalize( LIGHT_POSITION );

    float3  hal = normalize( lig-rd );
    float amb = saturate( 0.5+0.5*nor.y);
    float dif = saturate( dot( nor, lig ));
    float bac = saturate( dot( nor, normalize(float3(-lig.x,0.0,-lig.z))))*saturate( 1.0-pos.y);
    float dom = smoothstep( -0.1, 0.1, ref.y );
    float fre = pow( saturate(1.0+dot(nor,rd) ), 2.0 );

    dif *= raymarchSoftShadow( pos, lig, 0.02, 2.5 );
    dom *= raymarchSoftShadow( pos, ref, 0.02, 2.5 );

    mat.shadows = 1.30 * dif;
    mat.shadows += 0.40 * amb * occ;
    mat.shadows += 0.50 * dom * occ;
    mat.shadows += 0.50 * bac * occ;
    mat.shadows += 0.25 * fre * occ * 0.25;
    #endif
}
#endif

#endif