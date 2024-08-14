#include "../envMap.glsl"

/*
contributors:  The Art Of Code
description: |
    Raymarching for glass render. For more info, see the video link:
    Tutorial 1: https://youtu.be/NCpaaLkmXI8
    Tutorial 2: https://youtu.be/0RWaR7zApEo
use: <vec3> raymarchGlass( in <vec3> ray, in <vec3> pos, in <float> ior, in <float> roughness ) 
options:
    - RAYMARCH_GLASS_DENSITY: 0.                        [Density of the ray going through the glass]
    - RAYMARCH_GLASS_COLOR: vec3(1.0, 1.0, 1.0)       [Color of the glass]
    - RAYMARCH_GLASS_WAVELENGTH                         [Define this option to enable chromatic abberation effects]
    - RAYMARCH_GLASS_ENABLE_REFLECTION                  [Define this option to enable reflection]
    - RAYMARCH_GLASS_REFLECTION_EFFECT 5.               [The higher the value, the less reflections area from surface view]
    - RAYMARCH_GLASS_CHROMATIC_ABBERATION .01           [Chromatic Abberation Effects value on environment map]
    - RAYMARCH_GLASS_EDGE_SHARPNESS                     [Optional, to determine the edge sharpness]
    - RAYMARCH_GLASS_FNC_MANUAL                         [Optional, enable this to set glass params manually without using defines]
    - RAYMARCH_GLASS_FNC(RAY,POSITION,IOR,ROUGHNESS)
    - RAYMARCH_GLASS_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness)
examples:
    - /shaders/lighting_raymarching_glass_refraction.frag
*/

#ifndef RAYMARCH_GLASS_DENSITY
#define RAYMARCH_GLASS_DENSITY 0.
#endif

#ifndef RAYMARCH_GLASS_COLOR
#define RAYMARCH_GLASS_COLOR vec3(1.,1.,1.)
#endif

#if !defined(RAYMARCH_GLASS_REFLECTION_EFFECT) && defined(RAYMARCH_GLASS_ENABLE_REFLECTION)
#define RAYMARCH_GLASS_REFLECTION_EFFECT 5.
#endif

#ifdef RAYMARCH_GLASS_MAP_FNC
#define RAYMARCH_GLASS_WAVELENGTH_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness) RAYMARCH_GLASS_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness)
#endif

#ifndef RAYMARCH_GLASS_FNC
#define RAYMARCH_GLASS_FNC(RAY, POSITION, IOR, ROUGHNESS) raymarchDefaultGlass(RAY, POSITION, IOR, ROUGHNESS)
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
#define RAYMARCH_MAP_TYPE vec4
#endif

#ifndef RAYMARCH_MAP_MATERIAL_TYPE
#define RAYMARCH_MAP_MATERIAL_TYPE vec3
#endif

#ifndef RAYMARCH_GLASS_MAP_MATERIAL
#define RAYMARCH_GLASS_MAP_MATERIAL rgb
#endif

#ifndef FNC_RAYMARCH_GLASS
#define FNC_RAYMARCH_GLASS
RAYMARCH_MAP_TYPE raymarchGlassMarching(in vec3 ro, in vec3 rd) {
    float tmin = RAYMARCH_GLASS_MIN_DIST;
    float tmax = RAYMARCH_GLASS_MAX_DIST;

    float t = tmin;
    RAYMARCH_MAP_MATERIAL_TYPE m;

    // Because when the ray is inside the surface,the distance becomes negative.
    float side = sign(RAYMARCH_MAP_FNC(ro).RAYMARCH_MAP_DISTANCE);

    for (int i = 0; i < RAYMARCH_GLASS_SAMPLES; i++) {
        vec3 pos = ro + rd * t;
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

// For overwriting the parameters rendering that can be set manually
#ifdef RAYMARCH_GLASS_FNC_MANUAL
vec3 raymarchDefaultGlass(in vec3 ray, in vec3 pos, in float ior, in float roughness, in float glassSharpness, in float chromatic, in float density, in bool enableReflection, in float reflection, in vec3 colorGlass) {
    vec3 color = vec3(0.);

    RAYMARCH_MAP_TYPE marchOutside = raymarchGlassMarching(pos,ray); // Outside of the object
    if(marchOutside.RAYMARCH_MAP_DISTANCE < RAYMARCH_MAX_DIST) {
        vec3 newPos = pos + ray * marchOutside.RAYMARCH_MAP_DISTANCE;
        vec3 nEnter, nExit;

        nEnter = raymarchNormal(newPos, glassSharpness);

        vec3 newReflect = reflect(ray, nEnter);

        color = envMap(newReflect, roughness).rgb;

        vec3 rdIn = refract(ray, nEnter, 1./ior);
        vec3 pEnter = newPos - nEnter * RAYMARCH_GLASS_MIN_HIT_DIST * 3.;
        
        RAYMARCH_MAP_TYPE marchInside = raymarchGlassMarching(pEnter, rdIn); // Inside the object
        
        vec3 pExit = pEnter + rdIn * marchInside.RAYMARCH_MAP_DISTANCE;

        nExit = -raymarchNormal(pExit, glassSharpness);

        vec3 rdOut, res;
        if(chromatic != 0.) {

            #ifdef RAYMARCH_GLASS_WAVELENGTH_MAP_FNC
                RAYMARCH_GLASS_WAVELENGTH_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness);
            #else
                // Red
                rdOut = refract(rdIn, nExit, ior - chromatic);

                if(dot(rdOut, rdOut) == 0.)
                    rdOut = reflect(rdIn, nExit);

                res.r = envMap(rdOut, roughness).r;

                // Green
                rdOut = refract(rdIn, nExit, ior);

                if(dot(rdOut, rdOut) == 0.)
                    rdOut = reflect(rdIn, nExit);

                res.g = envMap(rdOut, roughness).g;

                // Blue
                rdOut = refract(rdIn, nExit, ior + chromatic);

                if(dot(rdOut, rdOut) == 0.)
                    rdOut = reflect(rdIn, nExit);

                res.b = envMap(rdOut, roughness).b;
            #endif

            float optDist = exp(-marchInside.RAYMARCH_MAP_DISTANCE * density);

            res *= optDist * colorGlass;
            
            if (enableReflection) {
                float fresnelVal = pow(1.+dot(ray, nEnter), reflection);
                return vec4(mix(res, color, saturate(fresnelVal)), 1.0);
            } else {
                return vec4(res, 1.0);
            }
        } else {
            rdOut = refract(rdIn, nExit, ior);

            if(dot(rdOut, rdOut) == 0.)
                rdOut = reflect(rdIn, nExit);

            float optDist = exp(-marchInside.RAYMARCH_MAP_DISTANCE * density);

            res = envMap(rdOut, roughness).rgb;

            res *= optDist * colorGlass;

            if (enableReflection) {
                float fresnelVal = pow(1.+dot(ray, nEnter), reflection);
                return vec4(mix(res, color, saturate(fresnelVal)), 1.0);
            } else {
                return vec4(res, 1.0);
            }
        }
    } else {
        return vec4(envMap(ray, 0.).rgb, 1.0);
    }
}
#endif

vec4 raymarchDefaultGlass(in vec3 ray, in vec3 pos, in float ior, in float roughness) {
    vec3 color = vec3(0.);

    RAYMARCH_MAP_TYPE marchOutside = raymarchGlassMarching(pos,ray); // Outside of the object
    if(marchOutside.RAYMARCH_MAP_DISTANCE < RAYMARCH_MAX_DIST) {
        vec3 newPos = pos + ray * marchOutside.RAYMARCH_MAP_DISTANCE;
        vec3 nEnter, nExit;

    #ifdef RAYMARCH_GLASS_EDGE_SHARPNESS
        nEnter = raymarchNormal(newPos, RAYMARCH_GLASS_EDGE_SHARPNESS);
    #else
        nEnter = raymarchNormal(newPos);
    #endif
        vec3 newReflect = reflect(ray, nEnter);

        color = envMap(newReflect, roughness).rgb;

        vec3 rdIn = refract(ray, nEnter, 1./ior);
        vec3 pEnter = newPos - nEnter * RAYMARCH_GLASS_MIN_HIT_DIST * 3.;
        
        RAYMARCH_MAP_TYPE marchInside = raymarchGlassMarching(pEnter, rdIn); // Inside the object
        
        vec3 pExit = pEnter + rdIn * marchInside.RAYMARCH_MAP_DISTANCE;

    #ifdef RAYMARCH_GLASS_EDGE_SHARPNESS
        nExit = -raymarchNormal(pExit, RAYMARCH_GLASS_EDGE_SHARPNESS);
    #else
        nExit = -raymarchNormal(pExit);
    #endif

        vec3 rdOut, res;
    #ifdef RAYMARCH_GLASS_WAVELENGTH

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
        
        #ifdef RAYMARCH_GLASS_ENABLE_REFLECTION
            float fresnelVal = pow(1.+dot(ray, nEnter), RAYMARCH_GLASS_REFLECTION_EFFECT);
            return vec4(mix(res, color, saturate(fresnelVal)), 1.);
        #else
            return vec4(res, 1.0);
        #endif
    #else
        rdOut = refract(rdIn, nExit, ior);

        if(dot(rdOut, rdOut) == 0.)
            rdOut = reflect(rdIn, nExit);

        float optDist = exp(-marchInside.RAYMARCH_MAP_DISTANCE * RAYMARCH_GLASS_DENSITY);

        res = envMap(rdOut, roughness).rgb;

        res *= optDist * RAYMARCH_GLASS_COLOR;

        #ifdef RAYMARCH_GLASS_ENABLE_REFLECTION
            float fresnelVal = pow(1.+dot(ray, nEnter), RAYMARCH_GLASS_REFLECTION_EFFECT);
            return vec4(mix(res, color, saturate(fresnelVal)), 1.0);
        #else
            return vec4(res, 1.0);
        #endif
    #endif
    } else {
        return vec4(envMap(ray, 0.).rgb, 1.0);
    }
}

vec4 raymarchGlass(in vec3 ray, in vec3 pos, in float ior, in float roughness) {
    return RAYMARCH_GLASS_FNC(ray, pos, ior, roughness);
}
#endif