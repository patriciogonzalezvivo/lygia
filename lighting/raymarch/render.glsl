#include "cast.glsl"
#include "ao.glsl"
#include "normal.glsl"
#include "shading.glsl"
#include "fog.glsl"

/*
contributors:  Inigo Quiles
description: Default raymarching renderer
use:
    - <vec4> raymarchDefaultRender( in <vec3> rayOriging, in <vec3> rayDirection, in <vec3> cameraForward)
    - <vec4> raymarchDefaultRender( in <vec3> rayOriging, in <vec3> rayDirection, in <vec3> cameraForward, out <float> eyeDepth )
    - <vec4> raymarchDefaultRender( in <vec3> rayOriging, in <vec3> rayDirection, in <vec3> cameraForward, out <float> eyeDepth, out <Material> res )
    - <vec4> raymarchDefaultRender( in <vec3> rayOriging, in <vec3> rayDirection, in <vec3> cameraForward, out <vec3> eyeDepth, out <vec3> worldPosition, out <vec3> worldNormal ) 
options:
    - RAYMARCH_BACKGROUND: vec3(0.0)
    - RAYMARCH_AOV: return AOVs in a Material structure
examples:
    - /shaders/lighting_raymarching.frag
*/

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND vec3(0.0, 0.0, 0.0)
#endif

#ifndef FNC_RAYMARCH_DEFAULT
#define FNC_RAYMARCH_DEFAULT

vec4 raymarchDefaultRender( in vec3 rayOrigin, in vec3 rayDirection, vec3 cameraForward,
                            out float dist ,out float eyeDepth, out Material res) { 

    res = raymarchCast(rayOrigin, rayDirection);
    float t = res.sdf;
    vec3 worldPos = rayOrigin + t * rayDirection;
    vec3 worldNormal = raymarchNormal( worldPos );

    vec4 color = vec4(RAYMARCH_BACKGROUND, 0.0);
    if (res.valid) {
        res.position = worldPos;
        res.normal = worldNormal;
        res.ambientOcclusion = raymarchAO(res.position, res.normal);
        ShadingData shadingData = shadingDataNew();
        shadingData.V = -rayDirection;
        color = RAYMARCH_SHADING_FNC(res, shadingData);
        dist = t;
    } else {
        dist = RAYMARCH_MAX_DIST;
    }
    
    color.rgb = raymarchFog(color.rgb, t, rayOrigin, rayDirection);

    #if defined(RAYMARCH_AOV)
    // Eye-space depth. See https://www.shadertoy.com/view/4tByz3
    eyeDepth = t * dot(rayDirection, cameraForward);
    #else
    eyeDepth = 0.0;
    #endif

    return color;
}

#endif
