#include "cast.hlsl"
#include "ao.hlsl"
#include "normal.hlsl"
#include "softShadow.hlsl"
#include "../../math/saturate.hlsl"

/*
original_author:  Inigo Quiles
description: default raymarching renderer
use: <float4> raymarchDefaultRender( in <float3> ro, in <float3> rd ) 
options:
    - LIGHT_COLOR: float3(0.5) or u_lightColor in GlslViewer
    - LIGHT_POSITION: float3(0.0, 10.0, -50.0) or u_light in GlslViewer
    - RAYMARCH_BACKGROUND: float3(0.0)
    - RAYMARCH_AMBIENT: float3(1.0)
    - RAYMARCH_MATERIAL_FNC raymarchDefaultMaterial
*/

#ifndef LIGHT_POSITION
#if defined(GLSLVIEWER)
#define LIGHT_POSITION u_light
#else
#define LIGHT_POSITION float3(0.0, 10.0, -50.0)
#endif
#endif

#ifndef LIGHT_COLOR
#if defined(GLSLVIEWER)
#define LIGHT_COLOR u_lightColor
#else
#define LIGHT_COLOR float3(0.5)
#endif
#endif

#ifndef RAYMARCH_AMBIENT
#define RAYMARCH_AMBIENT float3(1.0)
#endif

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND float3(0.0)
#endif

#ifndef RAYMARCH_MATERIAL_FNC
#define RAYMARCH_MATERIAL_FNC raymarchDefaultMaterial
#endif

#ifndef FNC_RAYMARCHDEFAULT
#define FNC_RAYMARCHDEFAULT

float3 raymarchDefaultMaterial(float3 ray, float3 pos, float3 nor, float3 alb) {
    if ( alb.r + alb.g + alb.b <= 0.0 ) 
        return RAYMARCH_BACKGROUND;

    float3 color = alb;
    float3 ref = reflect( ray, nor );
    float occ = raymarchAO( pos, nor );
    float3  lig = normalize( LIGHT_POSITION );
    float3  hal = normalize( lig-ray );
    float amb = saturate( 0.5+0.5*nor.y );
    float dif = saturate( dot( nor, lig ) );
    float bac = saturate( dot( nor, normalize(float3(-lig.x,0.0,-lig.z))) ) * saturate( 1.0-pos.y );
    float dom = smoothstep( -0.1, 0.1, ref.y );
    float fre = pow( saturate(1.0+dot(nor,ray) ), 2.0 );
    
    dif *= raymarchSoftShadow( pos, lig, 0.02, 2.5 );
    dom *= raymarchSoftShadow( pos, ref, 0.02, 2.5 );

    float3 lin = float3(0.0);
    lin += 1.30 * dif * LIGHT_COLOR;
    lin += 0.40 * amb * occ * RAYMARCH_AMBIENT;
    lin += 0.50 * dom * occ * RAYMARCH_AMBIENT;
    lin += 0.50 * bac * occ * 0.25;
    lin += 0.25 * fre * occ;

    return color * lin;
}

float4 raymarchDefaultRender( in float3 ray_origin, in float3 ray_direction ) { 
    float3 col = float3(0.0);
    
    float4 res = raymarchCast(ray_origin, ray_direction);
    float t = res.a;

    float3 pos = ray_origin + t * ray_direction;
    float3 nor = raymarchNormal( pos );
    col = RAYMARCH_MATERIAL_FNC(ray_direction, pos, nor, res.rgb);

    return float4( saturate(col), t );
}

#endif