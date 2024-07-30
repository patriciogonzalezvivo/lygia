#include "cast.glsl"
#include "ao.glsl"
#include "normal.glsl"
#include "softShadow.glsl"
#include "material.glsl"
#include "fog.glsl"
#include "../../math/saturate.glsl"

/*
contributors:  Inigo Quiles
description: Default raymarching renderer
use: <vec4> raymarchDefaultRender( in <vec3> rayOriging, in <vec3> rayDirection, in <vec3> cameraForward,
    out <vec3> eyeDepth, out <vec3> worldPosition, out <vec3> worldNormal ) 
options:
    - LIGHT_COLOR: vec3(0.5) or u_lightColor in GlslViewer
    - LIGHT_POSITION: vec3(0.0, 10.0, -50.0) or u_light in GlslViewer
    - LIGHT_DIRECTION;
    - RAYMARCH_BACKGROUND: vec3(0.0)
    - RAYMARCH_AMBIENT: vec3(1.0)
    - RAYMARCH_MATERIAL_FNC raymarchDefaultMaterial
examples:
    - /shaders/lighting_raymarching.frag
*/

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE a
#endif

#ifndef RAYMARCH_MAP_MATERIAL
#define RAYMARCH_MAP_MATERIAL rgb
#endif

#ifndef RAYMARCH_MAP_MATERIAL_TYPE
#define RAYMARCH_MAP_MATERIAL_TYPE vec3
#endif

#ifndef RAYMARCHCAST_TYPE
#define RAYMARCHCAST_TYPE vec4
#endif

#ifndef FNC_RAYMARCHDEFAULT
#define FNC_RAYMARCHDEFAULT

vec4 raymarchDefaultRender(
    in vec3 rayOrigin, in vec3 rayDirection, vec3 cameraForward,
    out float eyeDepth, out vec3 worldPos, out vec3 worldNormal ) { 

    vec4 color = vec4(0.0);
    
    RAYMARCHCAST_TYPE res = raymarchCast(rayOrigin, rayDirection);
    float t = res.RAYMARCH_MAP_DISTANCE;

    worldPos = rayOrigin + t * rayDirection;
    worldNormal = raymarchNormal( worldPos );
    color = raymarchMaterial(rayDirection, worldPos, worldNormal, res.RAYMARCH_MAP_MATERIAL);
    color.rgb = raymarchFog(color.rgb, t, rayOrigin, rayDirection);

    // Eye-space depth. See https://www.shadertoy.com/view/4tByz3
    eyeDepth = t * dot(rayDirection, cameraForward);

    return color;
}

#endif