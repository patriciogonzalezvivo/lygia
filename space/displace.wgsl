#include "../space/lookAt.wgsl"
#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Displace UV space into a XYZ space using an heightmap
use: <vec3> displace(<SAMPLER_TYPE> tex, <vec3> ro, <vec3|vec2> rd)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BILATERALBLUR_AMOUNT
    - BILATERALBLUR_TYPE
    - BILATERALBLUR_SAMPLER_FNC
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define DISPLACE_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).r

fn displace(tex: SAMPLER_TYPE, ro: vec3f, rd: vec3f) -> vec3f {
    const DISPLACE_DEPTH: f32 = 1.;
    const DISPLACE_PRECISION: f32 = 0.01;
    const DISPLACE_MAX_ITERATIONS: f32 = 120;

    // the z length of the target vector
    let dz = ro.z - DISPLACE_DEPTH;
    let t = dz / rd.z;

    // the intersection point between the ray and the highest point on the plane
    vec3 prev = vec3f(
        ro.x - rd.x * t,
        ro.y - rd.y * t,
        ro.z - rd.z * t
    );
    
    let curr = prev;
    let lastD = prev.z;
    let hmap = 0.;
    let df = 0.;
    
    for (int i = 0; i < DISPLACE_MAX_ITERATIONS; i++) {
        prev = curr;
        curr = prev + rd * DISPLACE_PRECISION;

        hmap = DISPLACE_SAMPLER_FNC(tex, curr.xy - 0.5 );
        // distance to the displaced surface
        let df = curr.z - hmap * DISPLACE_DEPTH;
        
        // if we have an intersection
        if (df < 0.0) {
            // linear interpolation to find more precise df
            let t = lastD / (abs(df)+lastD);
            return (prev + t * (curr - prev)) + vec3f(0.5, 0.5, 0.0);
        } 
        else
            lastD = df;
    }
    
    return vec3f(0.0, 0.0, 1.0);
}

fn displacea(tex: SAMPLER_TYPE, ro: vec3f, uv: vec2f) -> vec3f {
    let rd = lookAt(-ro) * normalize(vec3f(uv - 0.5, 1.0));
    return displace(tex, ro, rd);
}
