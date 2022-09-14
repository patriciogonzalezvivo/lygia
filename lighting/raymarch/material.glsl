#include "../material.glsl"
#include "ao.glsl"
#include "normal.glsl"
#include "softShadow.glsl"
#include "cast.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Material Constructor. Designed to integrate with GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines 
use: 
    - void raymarchMaterial(in <vec3> ro, in <vec3> rd, out material _mat)
    - material raymarchMaterial(in <vec3> ro, in <vec3> rd)
    - LIGHT_POSITION: in glslViewer is u_light
*/

#ifndef LIGHT_POSITION
#if defined(GLSLVIEWER)
#define LIGHT_POSITION u_light
#else
#define LIGHT_POSITION vec3(0.0, 10.0, -50.0)
#endif
#endif

#ifndef FNC_RAYMARCHMATERIAL
#define FNC_RAYMARCHMATERIAL

void raymarchMaterial( in vec3 ro, in vec3 rd, inout Material mat) { 
    vec4 res = raymarchCast(ro, rd);

    vec3 col = vec3(0.0);
    vec3 m = res.rgb;
    float t = res.a;

    vec3 pos = ro + t * rd;
    vec3 nor = raymarchNormal( pos );
    float occ = raymarchAO( pos, nor );

    mat.baseColor = res;
    mat.normal = nor;
    mat.ambientOcclusion = occ;

    #if defined(SHADING_SHADOWS)
    vec3 ref = reflect( rd, nor );
    vec3 lig = normalize( LIGHT_POSITION );

    vec3  hal = normalize( lig-rd );
    float amb = clamp( 0.5+0.5*nor.y, 0.0, 1.0 );
    float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
    float bac = clamp( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
    float dom = smoothstep( -0.1, 0.1, ref.y );
    float fre = pow( clamp(1.0+dot(nor,rd),0.0,1.0), 2.0 );

    dif *= raymarchSoftShadow( pos, lig, 0.02, 2.5 );
    dom *= raymarchSoftShadow( pos, ref, 0.02, 2.5 );

    mat.shadows = 1.30 * dif;
    mat.shadows += 0.40 * amb * occ;
    mat.shadows += 0.50 * dom * occ;
    mat.shadows += 0.50 * bac * occ;
    mat.shadows += 0.25 * fre * occ * 0.25;
    #endif
}

Material raymarchMaterial( in vec3 ro, in vec3 rd) { 
    Material mat;
    raymarchMaterial( ro, rd, mat);
    return mat;
}

#endif