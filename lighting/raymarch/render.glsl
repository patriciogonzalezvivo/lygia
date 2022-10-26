#include "cast.glsl"
#include "ao.glsl"
#include "normal.glsl"
#include "softShadow.glsl"
#include "material.glsl"
#include "../../math/saturate.glsl"

/*
original_author:  Inigo Quiles
description: default raymarching renderer
use: <vec4> raymarchDefaultRender( in <vec3> ro, in <vec3> rd ) 
options:
    - LIGHT_COLOR: vec3(0.5) or u_lightColor in GlslViewer
    - LIGHT_POSITION: vec3(0.0, 10.0, -50.0) or u_light in GlslViewer
    - LIGHT_DIRECTION;
    - RAYMARCH_BACKGROUND: vec3(0.0)
    - RAYMARCH_AMBIENT: vec3(1.0)
    - RAYMARCH_MATERIAL_FNC raymarchDefaultMaterial
*/

#ifndef FNC_RAYMARCHDEFAULT
#define FNC_RAYMARCHDEFAULT

vec4 raymarchDefaultRender( in vec3 ray_origin, in vec3 ray_direction ) { 
    vec3 col = vec3(0.0);
    
    vec4 res = raymarchCast(ray_origin, ray_direction);
    float t = res.a;

    vec3 pos = ray_origin + t * ray_direction;
    vec3 nor = raymarchNormal( pos );
    col = raymarchMaterial(ray_direction, pos, nor, res.rgb);

    return vec4( saturate(col), t );
}

#endif