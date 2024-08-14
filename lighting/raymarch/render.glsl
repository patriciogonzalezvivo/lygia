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
    - RAYMARCH_RETURN:  0. nothing (default), 1. depth;  2. depth and material; 3. depth: world position and normal
examples:
    - /shaders/lighting_raymarching.frag
*/

#ifndef RAYMARCH_RETURN 
#define RAYMARCH_RETURN 0
#endif

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND vec3(0.0, 0.0, 0.0)
#endif

#ifndef FNC_RAYMARCH_DEFAULT
#define FNC_RAYMARCH_DEFAULT

vec4 raymarchDefaultRender( in vec3 rayOrigin, in vec3 rayDirection, vec3 cameraForward
#if RAYMARCH_RETURN != 0
                            ,out float eyeDepth
#endif
#if RAYMARCH_RETURN == 2
                            ,out Material res
#elif RAYMARCH_RETURN == 3
                            ,out vec3 worldPos, out vec3 worldNormal 
#endif
    ) { 

#if RAYMARCH_RETURN != 2
    Material res;
#endif
#if RAYMARCH_RETURN != 3
    vec3 worldPos, worldNormal;
#endif

    res = raymarchCast(rayOrigin, rayDirection);
    float t = res.sdf;

    worldPos = rayOrigin + t * rayDirection;
    worldNormal = raymarchNormal( worldPos );

    vec4 color = vec4(RAYMARCH_BACKGROUND, 0.0);
    if (res.valid) {
        res.position = worldPos;
        res.normal = worldNormal;
        // res.ambientOcclusion = raymarchAO(res.position, res.normal);
        res.V = -rayDirection;
        color = RAYMARCH_SHADING_FNC(res);
    }
    
    color.rgb = raymarchFog(color.rgb, t, rayOrigin, rayDirection);

    #if RAYMARCH_RETURN != 0
    eyeDepth = t;
    #endif

    return color;
}

#endif
