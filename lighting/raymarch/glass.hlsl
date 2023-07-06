#include "../envMap.hlsl"
#include "../fresnel.hlsl"
#include "../specular.hlsl"
#include "../reflection.hlsl"

/*
original_author:  The Art Of Code
description: |
    Raymarching for glass render. For more info, see the video below link:
    Tutorial 1:https://youtu.be/NCpaaLkmXI8
    Tutorial 2:https://youtu.be/0RWaR7zApEo
use: <float3> raymarchGlass( in <float3> ray, in <float3> pos, in <float> ior, in <float> roughness ) 
options:
    - RAYMARCH_GLASS_DENSITY: 0.                        [Density of the ray going through the glass]
    - RAYMARCH_GLASS_COLOR: float3(1.0, 1.0, 1.0)       [Color of the glass]
    - RAYMARCH_GLASS_WAVELENGTH                         [Define this option]
    - RAYMARCH_GLASS_ENABLE_FRESNEL                     [Define this option to enable fresnel on edges]
    - RAYMARCH_GLASS_FRESNEL_STRENGTH 5.                [The lower the value, more sharp it gets]
    - RAYMARCH_GLASS_CHROMATIC_ABBERATION .01           [Chromatic Abberation Effects on environment map]
    - RAYMARCH_GLASS_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness)
examples: |
    - /shaders/lighting_glass_raymarching_refraction.frag
*/

#ifndef RAYMARCH_GLASS_DENSITY
#define RAYMARCH_GLASS_DENSITY 0.
#endif
#ifndef RAYMARCH_GLASS_COLOR
#define RAYMARCH_GLASS_COLOR float3(1.,1.,1.)
#endif

#ifdef RAYMARCH_GLASS_WAVELENGTH
    #if !defined(RAYMARCH_GLASS_FRESNEL_STRENGTH) && defined(RAYMARCH_GLASS_ENABLE_FRESNEL)
    #define RAYMARCH_GLASS_FRESNEL_STRENGTH 5.
    #endif
#endif

#ifdef RAYMARCH_GLASS_MAP_FNC
#define RAYMARCH_GLASS_WAVELENGTH_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness) RAYMARCH_GLASS_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness)
#endif

#ifndef RAYMARCH_GLASS_CHROMATIC_ABBERATION
#define RAYMARCH_GLASS_CHROMATIC_ABBERATION .01
#endif

#ifndef RAYMARCH_GLASS_SAMPLES
#define RAYMARCH_GLASS_SAMPLES 50
#endif

#ifndef RAYMARCH_GLASS_MIN_DIST
#define RAYMARCH_GLASS_MIN_DIST 0.
#endif

#ifndef RAYMARCH_GLASS_MAX_DIST
#define RAYMARCH_GLASS_MAX_DIST 100.
#endif

#ifndef RAYMARCH_GLASS_MIN_HIT_DIST
#define RAYMARCH_GLASS_MIN_HIT_DIST .0001
#endif

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE a
#endif

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)
#endif

#ifndef RAYMARCH_MAP_TYPE
#define RAYMARCH_MAP_TYPE float4
#endif

#ifndef RAYMARCH_MAP_MATERIAL_TYPE
#define RAYMARCH_MAP_MATERIAL_TYPE float3
#endif

#ifndef RAYMARCH_GLASS_MAP_MATERIAL
#define RAYMARCH_GLASS_MAP_MATERIAL rgb
#endif

#ifndef FNC_RAYMARCHGLASS
#define FNC_RAYMARCHGLASS
RAYMARCH_MAP_TYPE raymarchGlassMarching(in float3 ro, in float3 rd) {
    float tmin = RAYMARCH_GLASS_MIN_DIST;
    float tmax = RAYMARCH_GLASS_MAX_DIST;

    float t = tmin;
    RAYMARCH_MAP_MATERIAL_TYPE m;

    // Because when the ray is inside the surface,the distance becomes negative.
    float side = sign(RAYMARCH_MAP_FNC(ro).RAYMARCH_MAP_DISTANCE);

    for (int i = 0; i < RAYMARCH_GLASS_SAMPLES; i++) {
        float3 pos = ro + rd * t;
        RAYMARCH_MAP_TYPE sideDirection = RAYMARCH_MAP_FNC(pos);
        t += sideDirection.RAYMARCH_MAP_DISTANCE * side;
        m = sideDirection.RAYMARCH_GLASS_MAP_MATERIAL;
        if(t > tmax || abs(t) < RAYMARCH_GLASS_MIN_HIT_DIST)
            break;

    }
    return RAYMARCH_MAP_TYPE(m, t);
}
#endif

#ifndef FNC_RAYMARCH_DEFAULT_GLASS
#define FNC_RAYMARCH_DEFAULT_GLASS
float3 raymarchGlass(in float3 ray, in float3 pos, in float ior, in float roughness) {
    float3 color = envMap(ray, roughness).rgb;

    RAYMARCH_MAP_TYPE marchOutside = raymarchGlassMarching(pos,ray); // Outside of the object
    if(marchOutside.RAYMARCH_MAP_DISTANCE < RAYMARCH_MAX_DIST) {
        float3 newPos = pos + ray * marchOutside.RAYMARCH_MAP_DISTANCE;
        float3 nEnter, nExit;

    #ifdef RAYMARCH_GLASS_EDGE_SHARPNESS
        nEnter = raymarchNormal(newPos, RAYMARCH_GLASS_EDGE_SHARPNESS);
    #else
        nEnter = raymarchNormal(newPos);
    #endif
        float3 newReflect = reflect(ray, nEnter);

        float3 rdIn = refract(ray, nEnter, 1./ior);
        float3 pEnter = newPos - nEnter * RAYMARCH_GLASS_MIN_HIT_DIST * 3.;
        
        RAYMARCH_MAP_TYPE marchInside = raymarchGlassMarching(pEnter, rdIn); // Inside the object
        
        float3 pExit = pEnter + rdIn * marchInside.RAYMARCH_MAP_DISTANCE;

    #ifdef RAYMARCH_GLASS_EDGE_SHARPNESS
        nExit = -raymarchNormal(pExit, RAYMARCH_GLASS_EDGE_SHARPNESS);
    #else
        nExit = -raymarchNormal(pExit);
    #endif

        float3 rdOut, res;
    #ifdef RAYMARCH_GLASS_WAVELENGTH
        float NoV = dot(ray, nEnter);

        #ifdef RAYMARCH_GLASS_WAVELENGTH_MAP_FNC
            RAYMARCH_GLASS_WAVELENGTH_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness);
        #else
            // Red
            rdOut = refract(rdIn, nExit, ior - RAYMARCH_GLASS_CHROMATIC_ABBERATION);

            if(dot(rdOut, rdOut) == 0.)
                rdOut = reflect(rdIn, nExit);

            res.r = envMap(rdOut, roughness).r;

            // Green
            rdOut = refract(rdIn, nExit, ior);

            if(dot(rdOut, rdOut) == 0.)
                rdOut = reflect(rdIn, nExit);

            res.g = envMap(rdOut, roughness).g;

            // Blue
            rdOut = refract(rdIn, nExit, ior + RAYMARCH_GLASS_CHROMATIC_ABBERATION);

            if(dot(rdOut, rdOut) == 0.)
                rdOut = reflect(rdIn, nExit);

            res.b = envMap(rdOut, roughness).b;
        #endif

        float optDist = exp(-marchInside.RAYMARCH_MAP_DISTANCE * RAYMARCH_GLASS_DENSITY);

        res *= optDist * RAYMARCH_GLASS_COLOR;
    #ifdef RAYMARCH_GLASS_ENABLE_FRESNEL
        float fresnelVal = pow(1.+dot(ray, nEnter), RAYMARCH_GLASS_FRESNEL_STRENGTH);
        return mix(res, color, saturate(fresnelVal));
    #else
        return res;
    #endif
    #else
        rdOut = refract(rdIn, nExit, ior);

        if(dot(rdOut, rdOut) == 0.)
            rdOut = reflect(rdIn, nExit);

        float optDist = exp(-marchInside.RAYMARCH_MAP_DISTANCE * RAYMARCH_GLASS_DENSITY);

        res = envMap(rdOut, roughness).rgb;

        res *= optDist * RAYMARCH_GLASS_COLOR;

        return res;
    #endif
    } else {
        return envMap(ray, 0.).rgb;
    }
}
#endif