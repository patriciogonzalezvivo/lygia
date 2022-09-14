#include "cast.glsl"
#include "ao.glsl"
#include "normal.glsl"
#include "softShadow.glsl"
#include "../../math/saturate.glsl"

/*
original_author:  Inigo Quiles
description: default raymarching renderer
use: <vec4> raymarchDefaultRender( in <vec3> ro, in <vec3> rd ) 
options:
    - LIGHT_COLOR: vec3(0.5) or u_lightColor in GlslViewer
    - LIGHT_POSITION: vec3(0.0, 10.0, -50.0) or u_light in GlslViewer
    - RAYMARCH_BACKGROUND: vec3(0.0)
    - RAYMARCH_AMBIENT: vec3(1.0)
    - RAYMARCH_MATERIAL_FNC raymarchDefaultMaterial
*/

#ifndef LIGHT_POSITION
#if defined(GLSLVIEWER)
#define LIGHT_POSITION u_light
#else
#define LIGHT_POSITION vec3(0.0, 10.0, -50.0)
#endif
#endif

#ifndef LIGHT_COLOR
#if defined(GLSLVIEWER)
#define LIGHT_COLOR u_lightColor
#else
#define LIGHT_COLOR vec3(0.5)
#endif
#endif

#ifndef RAYMARCH_AMBIENT
#define RAYMARCH_AMBIENT vec3(1.0)
#endif

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND vec3(0.0)
#endif

#ifndef RAYMARCH_MATERIAL_FNC
#define RAYMARCH_MATERIAL_FNC raymarchDefaultMaterial
#endif

#ifndef FNC_RAYMARCHDEFAULT
#define FNC_RAYMARCHDEFAULT

vec3 raymarchDefaultMaterial(vec3 ray, vec3 pos, vec3 nor, vec3 alb) {
    if ( alb.r + alb.g + alb.b <= 0.0 ) 
        return RAYMARCH_BACKGROUND;

    vec3 color = alb;
    vec3 ref = reflect( ray, nor );
    float occ = raymarchAO( pos, nor );
    vec3  lig = normalize( LIGHT_POSITION );
    vec3  hal = normalize( lig-ray );
    float amb = saturate( 0.5+0.5*nor.y );
    float dif = saturate( dot( nor, lig ) );
    float bac = saturate( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))) ) * saturate( 1.0-pos.y );
    float dom = smoothstep( -0.1, 0.1, ref.y );
    float fre = pow( saturate(1.0+dot(nor,ray) ), 2.0 );
    
    dif *= raymarchSoftShadow( pos, lig, 0.02, 2.5 );
    dom *= raymarchSoftShadow( pos, ref, 0.02, 2.5 );

    vec3 lin = vec3(0.0);
    lin += 1.30 * dif * LIGHT_COLOR;
    lin += 0.40 * amb * occ * RAYMARCH_AMBIENT;
    lin += 0.50 * dom * occ * RAYMARCH_AMBIENT;
    lin += 0.50 * bac * occ * 0.25;
    lin += 0.25 * fre * occ;

    return color * lin;
}

vec4 raymarchDefaultRender( in vec3 ray_origin, in vec3 ray_direction ) { 
    vec3 col = vec3(0.0);
    
    vec4 res = raymarchCast(ray_origin, ray_direction);
    float t = res.a;

    vec3 pos = ray_origin + t * ray_direction;
    vec3 nor = raymarchNormal( pos );
    col = RAYMARCH_MATERIAL_FNC(ray_direction, pos, nor, res.rgb);

    return vec4( saturate(col), t );
}

#endif