#include "../envMap.wgsl"

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
    - RAYMARCH_GLASS_WAVELENGTH                         [Define this option to enable chromatic aberration effects]
    - RAYMARCH_GLASS_ENABLE_REFLECTION                  [Define this option to enable reflection]
    - RAYMARCH_GLASS_REFLECTION_EFFECT 5.               [The higher the value, the less reflections area from surface view]
    - RAYMARCH_GLASS_CHROMATIC_ABERRATION .01           [Chromatic Aberration Effects value on environment map]
    - RAYMARCH_GLASS_EDGE_SHARPNESS                     [Optional, to determine the edge sharpness]
    - RAYMARCH_GLASS_FNC_MANUAL                         [Optional, enable this to set glass params manually without using defines]
    - RAYMARCH_GLASS_FNC(RAY,POSITION,IOR,ROUGHNESS)
    - RAYMARCH_GLASS_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness)
examples:
    - /shaders/lighting_raymarching_glass_refraction.frag
*/

const RAYMARCH_GLASS_DENSITY: f32 = 0.;

// #define RAYMARCH_GLASS_COLOR vec3(1.,1.,1.)

const RAYMARCH_GLASS_REFLECTION_EFFECT: f32 = 5.;

// #define RAYMARCH_GLASS_WAVELENGTH_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness) RAYMARCH_GLASS_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness)

// #define RAYMARCH_GLASS_FNC(RAY, POSITION, IOR, ROUGHNESS) raymarchDefaultGlass(RAY, POSITION, IOR, ROUGHNESS)

// #define RAYMARCH_GLASS_CHROMATIC_ABERRATION .01

const RAYMARCH_GLASS_SAMPLES: f32 = 50;

const RAYMARCH_GLASS_MIN_DIST: f32 = 0.;

const RAYMARCH_GLASS_MAX_DIST: f32 = 100.;

// #define RAYMARCH_GLASS_MIN_HIT_DIST .0001

// #define RAYMARCH_MAP_DISTANCE a

// #define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)

// #define RAYMARCH_MAP_TYPE vec4

// #define RAYMARCH_MAP_MATERIAL_TYPE vec3

// #define RAYMARCH_GLASS_MAP_MATERIAL rgb

RAYMARCH_MAP_TYPE raymarchGlassMarching(in vec3 ro, in vec3 rd) {
    let tmin = RAYMARCH_GLASS_MIN_DIST;
    let tmax = RAYMARCH_GLASS_MAX_DIST;

    let t = tmin;
    RAYMARCH_MAP_MATERIAL_TYPE m;

    // Because when the ray is inside the surface,the distance becomes negative.
    let side = sign(RAYMARCH_MAP_FNC(ro).RAYMARCH_MAP_DISTANCE);

    for (int i = 0; i < RAYMARCH_GLASS_SAMPLES; i++) {
        let pos = ro + rd * t;
        RAYMARCH_MAP_TYPE sideDirection = RAYMARCH_MAP_FNC(pos);
        t += sideDirection.RAYMARCH_MAP_DISTANCE * side;
        m = sideDirection.RAYMARCH_GLASS_MAP_MATERIAL;
        if(t > tmax || abs(t) < RAYMARCH_GLASS_MIN_HIT_DIST)
            break;

    }
    return RAYMARCH_MAP_TYPE(m, t);
}

// For overwriting the parameters rendering that can be set manually
fn raymarchDefaultGlass3(ray: vec3f, pos: vec3f, ior: f32, roughness: f32, glassSharpness: f32, chromatic: f32, density: f32, enableReflection: bool, reflection: f32, colorGlass: vec3f) -> vec3f {
    let color = vec3f(0.);

    RAYMARCH_MAP_TYPE marchOutside = raymarchGlassMarching(pos,ray); // Outside of the object
    if(marchOutside.RAYMARCH_MAP_DISTANCE < RAYMARCH_MAX_DIST) {
        let newPos = pos + ray * marchOutside.RAYMARCH_MAP_DISTANCE;
        vec3 nEnter, nExit;

        nEnter = raymarchNormal(newPos, glassSharpness);

        let newReflect = reflect(ray, nEnter);

        color = envMap(newReflect, roughness).rgb;

        let rdIn = refract(ray, nEnter, 1./ior);
        let pEnter = newPos - nEnter * RAYMARCH_GLASS_MIN_HIT_DIST * 3.;
        
        RAYMARCH_MAP_TYPE marchInside = raymarchGlassMarching(pEnter, rdIn); // Inside the object
        
        let pExit = pEnter + rdIn * marchInside.RAYMARCH_MAP_DISTANCE;

        nExit = -raymarchNormal(pExit, glassSharpness);

        vec3 rdOut, res;
        if(chromatic != 0.) {

                RAYMARCH_GLASS_WAVELENGTH_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness);
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

            let optDist = exp(-marchInside.RAYMARCH_MAP_DISTANCE * density);

            res *= optDist * colorGlass;
            
            if (enableReflection) {
                let fresnelVal = pow(1.+dot(ray, nEnter), reflection);
                return vec4f(mix(res, color, saturate(fresnelVal)), 1.0);
            } else {
                return vec4f(res, 1.0);
            }
        } else {
            rdOut = refract(rdIn, nExit, ior);

            if(dot(rdOut, rdOut) == 0.)
                rdOut = reflect(rdIn, nExit);

            let optDist = exp(-marchInside.RAYMARCH_MAP_DISTANCE * density);

            res = envMap(rdOut, roughness).rgb;

            res *= optDist * colorGlass;

            if (enableReflection) {
                let fresnelVal = pow(1.+dot(ray, nEnter), reflection);
                return vec4f(mix(res, color, saturate(fresnelVal)), 1.0);
            } else {
                return vec4f(res, 1.0);
            }
        }
    } else {
        return vec4f(envMap(ray, 0.).rgb, 1.0);
    }
}

fn raymarchDefaultGlass3a(ray: vec3f, pos: vec3f, ior: f32, roughness: f32) -> vec4f {
    let color = vec3f(0.);

    RAYMARCH_MAP_TYPE marchOutside = raymarchGlassMarching(pos,ray); // Outside of the object
    if(marchOutside.RAYMARCH_MAP_DISTANCE < RAYMARCH_MAX_DIST) {
        let newPos = pos + ray * marchOutside.RAYMARCH_MAP_DISTANCE;
        vec3 nEnter, nExit;

        nEnter = raymarchNormal(newPos, RAYMARCH_GLASS_EDGE_SHARPNESS);
        nEnter = raymarchNormal(newPos);
        let newReflect = reflect(ray, nEnter);

        color = envMap(newReflect, roughness).rgb;

        let rdIn = refract(ray, nEnter, 1./ior);
        let pEnter = newPos - nEnter * RAYMARCH_GLASS_MIN_HIT_DIST * 3.;
        
        RAYMARCH_MAP_TYPE marchInside = raymarchGlassMarching(pEnter, rdIn); // Inside the object
        
        let pExit = pEnter + rdIn * marchInside.RAYMARCH_MAP_DISTANCE;

        nExit = -raymarchNormal(pExit, RAYMARCH_GLASS_EDGE_SHARPNESS);
        nExit = -raymarchNormal(pExit);

        vec3 rdOut, res;

            RAYMARCH_GLASS_WAVELENGTH_MAP_FNC(res, rdIn, rdOut, pEnter, pExit, nEnter, nExit, ior, roughness);
            // Red
            rdOut = refract(rdIn, nExit, ior - RAYMARCH_GLASS_CHROMATIC_ABERRATION);

            if(dot(rdOut, rdOut) == 0.)
                rdOut = reflect(rdIn, nExit);

            res.r = envMap(rdOut, roughness).r;

            // Green
            rdOut = refract(rdIn, nExit, ior);

            if(dot(rdOut, rdOut) == 0.)
                rdOut = reflect(rdIn, nExit);

            res.g = envMap(rdOut, roughness).g;

            // Blue
            rdOut = refract(rdIn, nExit, ior + RAYMARCH_GLASS_CHROMATIC_ABERRATION);

            if(dot(rdOut, rdOut) == 0.)
                rdOut = reflect(rdIn, nExit);

            res.b = envMap(rdOut, roughness).b;

        let optDist = exp(-marchInside.RAYMARCH_MAP_DISTANCE * RAYMARCH_GLASS_DENSITY);

        res *= optDist * RAYMARCH_GLASS_COLOR;
        
            let fresnelVal = pow(1.+dot(ray, nEnter), RAYMARCH_GLASS_REFLECTION_EFFECT);
            return vec4f(mix(res, color, saturate(fresnelVal)), 1.);
            return vec4f(res, 1.0);
        rdOut = refract(rdIn, nExit, ior);

        if(dot(rdOut, rdOut) == 0.)
            rdOut = reflect(rdIn, nExit);

        let optDist = exp(-marchInside.RAYMARCH_MAP_DISTANCE * RAYMARCH_GLASS_DENSITY);

        res = envMap(rdOut, roughness).rgb;

        res *= optDist * RAYMARCH_GLASS_COLOR;

            let fresnelVal = pow(1.+dot(ray, nEnter), RAYMARCH_GLASS_REFLECTION_EFFECT);
            return vec4f(mix(res, color, saturate(fresnelVal)), 1.0);
            return vec4f(res, 1.0);
    } else {
        return vec4f(envMap(ray, 0.).rgb, 1.0);
    }
}

fn raymarchGlass(ray: vec3f, pos: vec3f, ior: f32, roughness: f32) -> vec4f {
    return RAYMARCH_GLASS_FNC(ray, pos, ior, roughness);
}
