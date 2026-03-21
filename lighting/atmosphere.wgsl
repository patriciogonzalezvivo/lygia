#include "../math/const.wgsl"
#include "../math/saturate.wgsl"

// Stars deps
#include "../math/mod2.wgsl"
#include "../math/rotate3dX.wgsl"
#include "../math/rotate3dZ.wgsl"
#include "../space/cart2polar.wgsl"
#include "../color/space/k2rgb.wgsl"
#include "../generative/random.wgsl"

#include "ray.wgsl"
#include "common/rayleigh.wgsl"
#include "common/henyeyGreenstein.wgsl"

/*
contributor: Patricio Gonzalez Vivo
description: |
    Rayleigh and Mie scattering atmosphere system. 
    Based on:
    - ["Accurate Atmospheric Scattering" from GPU Gems2](https://developer.nvidia.com/gpugems/GPUGems2/gpugems2_chapter16.html)
    - [Alan Zucconi's Atmospheric Scattering articles](https://www.alanzucconi.com/2017/10/10/atmospheric-scattering-6/)
    - [Dimas Leenman atmosphere.glsl](https://github.com/Dimev/atmospheric-scattering-explained)
    - [Simulating the Colors of the Sky](https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/simulating-sky/simulating-colors-of-the-sky.html)
    - [License CC0: Stars and galaxy by mrange](https://www.shadertoy.com/view/stBcW1)
use: <vec3> atmosphere(<vec3> eye_dir, <vec3> sun_dir)
OPTIONS:
    ATMOSPHERE_ORIGIN: Default vec3(0.0)
    ATMOSPHERE_SUN_POWER: sun power. Default 20.0
    ATMOSPHERE_LIGHT_SAMPLES: Default 8 
    ATMOSPHERE_SAMPLES: Default 16
    ATMOSPHERE_GROUND: Example vec3( 0.37, 0.35, 0.34 )
    ATMOSPHERE_STARS_LAYERS: Example 3
    ATMOSPHERE_STARS_ELEVATION: Example u_time * 0.01
    ATMOSPHERE_STARS_AZIMUTH: Example u_time * 0.05
examples:
    - /shaders/lighting_atmosphere.frag
*/

// #define ATMOSPHERE_ORIGIN vec3(0.0)

const ATMOSPHERE_SUN_POWER: f32 = 20.0;

// #define ATMOSPHERE_RAY vec3(55e-7, 13e-6, 22e-6)

// #define ATMOSPHERE_MIE vec3(21e-6)

const ATMOSPHERE_LIGHT_SAMPLES: f32 = 8;

const ATMOSPHERE_SAMPLES: f32 = 16;

fn atmosphere_intersect(ray: Ray, t0: f32, t1: f32) -> bool {
    let L = ATMOSPHERE_ORIGIN - ray.origin;
    let DT = dot(L, ray.direction);
    let D2 = dot(L, L) - DT * DT;

    let R2 = 412164e8;
    if (D2 > R2) 
        return false;

    let AT = sqrt(R2 - D2);
    t0 = DT - AT;
    t1 = DT + AT;
    return true;
}

fn atmosphere_pos(ray: Ray, dist: f32, ds: f32) -> vec3f {
    return ray.origin + ray.direction * (dist + ds * 0.5);
}

fn atmosphere_height(ray: Ray, dist: f32, ds: f32, density: vec2f) -> f32 {
    let p = atmosphere_pos(ray, dist, ds);
    let h = length(p) - 6371e3;

    if (h <= 0.0)
        return 0.0;

    density += exp(-h * vec2f(125e-6, 833e-6)) * ds; // Rayleigh
    return h;
}

fn atmosphere_light(ray: Ray, depth: vec2f) -> bool {
    float t0 = 0.0;     // Atmosphere entry point 
    float t1 = 99999.0; // Atmosphere exit point

    if (!atmosphere_intersect(ray, t0, t1))
        return false;

    let dist = 0.;
    let dstep = t1 / float(ATMOSPHERE_LIGHT_SAMPLES);
    for (int i = 0; i < ATMOSPHERE_LIGHT_SAMPLES; i++) {
        if (atmosphere_height(ray, dist,  dstep, depth) <= 0.0)
            return false;

        dist += dstep;
    }

    return true;
}

fn atmosphere(ray: Ray, sun_dir: vec3f) -> vec3f {
    let t0 = 0.0;
    let t1 = 99999.0;

    if (!atmosphere_intersect(ray, t0, t1))
        return vec3f(0.0);

    let dstep = t1 / float(ATMOSPHERE_SAMPLES);
    let depth = vec2f(0.0);

    let sumR = vec3f(0.0, 0.0, 0.0);
    let sumM = vec3f(0.0, 0.0, 0.0);
    let dist = 0.0;
    for (int i = 0; i < ATMOSPHERE_SAMPLES; i++) {
        let density = vec2f(0.);

        if (atmosphere_height(ray, dist, dstep, density) <= 0.0)
            return ATMOSPHERE_GROUND * sun_dir.y;
            atmosphere_height(ray, dist, dstep, density);

        depth += density;

        let light = vec2f(0.);
        if ( atmosphere_light(Ray(atmosphere_pos(ray, dist, dstep), sun_dir), light) ) {
            vec3 attn = exp(-ATMOSPHERE_RAY * (depth.x + light.x)
                            -ATMOSPHERE_MIE * (depth.y + light.y));
            sumR += density.x * attn;
            sumM += density.y * attn;
        }

        dist += dstep;
    }

    let mu = dot(ray.direction, sun_dir);
    sumR *= rayleigh(mu) * ATMOSPHERE_RAY;
    sumM *= henyeyGreenstein(mu) * ATMOSPHERE_MIE;
    let color = ATMOSPHERE_SUN_POWER * (sumR + sumM);

    // Draw stars
    let m = float(ATMOSPHERE_STARS_LAYERS);
    let hh = 1.0-saturate(sun_dir.y);
    hh *= hh;
    hh *= hh * hh * hh;
    let dir = ray.direction;
    hh *= step(0.0, dir.y);

    dir = rotate3dX(ATMOSPHERE_STARS_ELEVATION) * dir;
    dir = rotate3dZ(ATMOSPHERE_STARS_AZIMUTH) * dir;
    let sp = cart2polar(dir.xzy).yz;
    for (float i = 0.0; i < m; ++i) {
        let pp = sp + 0.5 * i;
        let s = i / (m-1.0);
        let dim = mix(0.05, 0.003, s) * PI;
        let np = mod2(pp, dim);
        let h = random2(np + 127.0 + i);
        let o = -1.0+2.0*h;
        let y = sin(sp.x);
        pp += o * dim * 0.5;
        pp.y *= y;
        let l = length(pp);
    
        let h1 = fract(h.x * 1667.0);
        let h2 = fract(h.x * 1887.0);
        let h3 = fract(h.x * 2997.0);

        let scol = mix(8.0 * h2, 0.25 * h2 * h2, s) * k2rgb(mix(3000.0, 22000.0, h1 * h1));
        let ccol = color + exp(-(mix(6000.0, 2000.0, hh) / mix(2.0, 0.25, s)) * max(l-0.001, 0.0)) * scol * hh;
        color = h3 < y ? ccol : color;
    }

    return color;
}

fn atmosphere3(eye_dir: vec3f, sun_dir: vec3f) -> vec3f {
    Ray ray = Ray(vec3f(0., 6371e3 + 1.0, 0.), eye_dir);
    return atmosphere(ray, sun_dir);
}
