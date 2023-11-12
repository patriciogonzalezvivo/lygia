#include "../math/const.glsl"
#include "../math/saturate.glsl"

#include "ray.glsl"
#include "common/henyeyGreenstein.glsl"
#include "common/rayleigh.glsl"

/*
contributor: Patricio Gonzalez Vivo
description: |
    Rayleigh and Mie scattering atmosphere system. 
    Based on:
    - ["Accurate Atmospheric Scattering" from GPU Gems2](https://developer.nvidia.com/gpugems/GPUGems2/gpugems2_chapter16.html)
    - [Alan Zucconi's Atmospheric Scattering articles](https://www.alanzucconi.com/2017/10/10/atmospheric-scattering-6/)
    - [Dimas Leenman atmosphere.glsl](https://github.com/Dimev/atmospheric-scattering-explained)
    - [Simulating the Colors of the Sky](https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/simulating-sky/simulating-colors-of-the-sky.html)
use: <vec3> atmosphere(<vec3> eye_dir, <vec3> sun_dir)
OPTIONS:
    ATMOSPHERE_FAST: use fast implementation
    ATMOSPHERE_ORIGIN: Defualt vec3(0.0)
    ATMOSPHERE_SUN_POWER: sun power. Default 20.0
    ATMOSPHERE_LIGHT_SAMPLES: Defualt 8 
    ATMOSPHERE_SAMPLES: Defualt 16
    ATMOSPHERE_GROUND: Defualt vec3( 0.37, 0.35, 0.34 )
    
examples:
    - /shaders/lighting_atmosphere.frag
*/

#ifndef ATMOSPHERE_ORIGIN
#define ATMOSPHERE_ORIGIN vec3(0.0)
#endif

#ifndef ATMOSPHERE_SUN_POWER
#define ATMOSPHERE_SUN_POWER 20.0
#endif

#ifndef ATMOSPHERE_RAY
#define ATMOSPHERE_RAY vec3(55e-7, 13e-6, 22e-6)
#endif

#ifndef ATMOSPHERE_MIE
#define ATMOSPHERE_MIE vec3(21e-6)
#endif

#ifndef ATMOSPHERE_LIGHT_SAMPLES
#define ATMOSPHERE_LIGHT_SAMPLES 8
#endif

#ifndef ATMOSPHERE_SAMPLES
#define ATMOSPHERE_SAMPLES 16
#endif

#ifndef ATMOSPHERE_GROUND
#define ATMOSPHERE_GROUND vec3( 0.37, 0.35, 0.34 )
#endif

#ifndef FNC_ATMOSPHERE
#define FNC_ATMOSPHERE

bool atmosphere_intersect(Ray ray, inout float t0, inout float t1) {
    vec3 L = ATMOSPHERE_ORIGIN - ray.origin; 
    float DT = dot(L, ray.direction);
    float D2 = dot(L, L) - DT * DT;

    const float R2 = 412164e8;
    if (D2 > R2) 
        return false;

    float AT = sqrt(R2 - D2);
    t0 = DT - AT;
    t1 = DT + AT;
    return true;
}

vec3 atmosphere_pos(Ray ray, float dist, float ds) {
    return ray.origin + ray.direction * (dist + ds * 0.5);
}

float atmosphere_height(Ray ray, float dist,  float ds, inout vec2 density) {
    vec3 p = atmosphere_pos(ray, dist, ds);
    float h = length(p) - 6371e3;
    if (h <= 0.0)
        return 0.0;
    density += exp(-h * vec2(125e-6, 833e-6)) * ds; // Rayleigh
    return h;
}

bool atmosphere_light(Ray ray, inout vec2 depth) {
    float t0 = 0.0;     // Atmosphere entry point 
    float t1 = 99999.0; // Atmosphere exit point

    #ifndef ATMOSPHERE_FAST
    if (!atmosphere_intersect(ray, t0, t1))
        return false;
    #endif

    float dist = 0.;
    float dstep = t1 / float(ATMOSPHERE_LIGHT_SAMPLES);
    for (int i = 0; i < ATMOSPHERE_LIGHT_SAMPLES; i++) {
        if (atmosphere_height(ray, dist,  dstep, depth) <= 0.0)
            return false;
        dist += dstep;
    }

    return true;
}

vec3 atmosphere(Ray ray, vec3 sun_dir) {
    float t0 = 0.0;
    float t1 = 99999.0;

    #ifndef ATMOSPHERE_FAST
    if (!atmosphere_intersect(ray, t0, t1))
        return vec3(0.0);
    #endif

    float dstep = t1 / float(ATMOSPHERE_SAMPLES);
    vec2 depth = vec2(0.0);

    vec3 sumR = vec3(0.0, 0.0, 0.0);
    vec3 sumM = vec3(0.0, 0.0, 0.0);
    float dist = 0.0;
    for (int i = 0; i < ATMOSPHERE_SAMPLES; i++) {
        vec2 density = vec2(0.);

        if (atmosphere_height(ray, dist, dstep, density) <= 0.0)
            return ATMOSPHERE_GROUND * sun_dir.y;

        depth += density;

        vec2 light = vec2(0.);
        if ( atmosphere_light(Ray(atmosphere_pos(ray, dist, dstep), sun_dir), light) )  {
            vec3 attn = exp(-ATMOSPHERE_RAY * (depth.x + light.x)
                            -ATMOSPHERE_MIE * (depth.y + light.y));
            sumR += density.x * attn;
            sumM += density.y * attn;
        }

        dist += dstep;
    }

    float mu = dot(ray.direction, sun_dir);
    sumR *= rayleigh(mu) * ATMOSPHERE_RAY;
    sumM *= henyeyGreenstein(mu) * ATMOSPHERE_MIE;
    return ATMOSPHERE_SUN_POWER * (sumR + sumM);
}

vec3 atmosphere(vec3 eye_dir, vec3 sun_dir) {
    Ray ray = Ray(vec3(0., 6371e3 + 1., 0.), eye_dir);
    return atmosphere(ray, sun_dir);
}

#endif