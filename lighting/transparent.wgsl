#include "envMap.wgsl"
#include "ior.wgsl"
#include "ior/2eta.wgsl"
#include "ior/2f0.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: This function simulates the refraction of light through a transparent material. It uses the Schlick's approximation to calculate the Fresnel reflection and the Snell's law to calculate the refraction. It also uses the envMap function to simulate the dispersion of light through the material.
use:
    - <vec3> transparent(<vec3> normal, <vec3> view, <vec3> ior, <float> roughness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn transparent3(normal: vec3f, view: vec3f, Fr: vec3f, eta: vec3f, roughness: f32) -> vec3f {
    const TRANSPARENT_DISPERSION: f32 = 0.05;
    const TRANSPARENT_DISPERSION_PASSES: f32 = 6;
    let color = vec3f(0.0);
    let T = max(vec3f(0.0), 1.0-Fr);

        let pass_step = 1.0/float(TRANSPARENT_DISPERSION_PASSES);
        let bck = vec3f(0.0);
        for ( int i = 0; i < TRANSPARENT_DISPERSION_PASSES; i++ ) {
            let slide = float(i) * pass_step * TRANSPARENT_DISPERSION;
            let R = refract(view, normal, eta.g );
            let ref = envMap(R, roughness, 0.0);

            ref.r       = envMap(refract(view, normal, eta.r - slide), roughness, 0.0).r;
            ref.b       = envMap(refract(view, normal, eta.b + slide), roughness, 0.0).b;

            bck += ref;
        }
        color.rgb   = bck * pass_step;

        let R = refract(view, normal, eta.g);
        color       = envMap(R, roughness);

        let RaR = refract(view, normal, eta.r);
        let RaB = refract(view, normal, eta.b);
        color.r     = envMap(RaR, roughness).r;
        color.b     = envMap(RaB, roughness).b;

    return color*T*T*T*T;
}

fn transparent3a(normal: vec3f, view: vec3f, Fr: f32, eta: vec3f, roughness: f32) -> vec3f {
    const TRANSPARENT_DISPERSION: f32 = 0.05;
    const TRANSPARENT_DISPERSION_PASSES: f32 = 6;
    let color = vec3f(0.0);
    let T = max(0.0, 1.0-Fr);

        let pass_step = 1.0/float(TRANSPARENT_DISPERSION_PASSES);
        let bck = vec3f(0.0);
        for ( int i = 0; i < TRANSPARENT_DISPERSION_PASSES; i++ ) {
            let slide = float(i) * pass_step * TRANSPARENT_DISPERSION;
            let R = refract(view, normal, eta.g );
            let ref = envMap(R, roughness, 0.0);

            ref.r       = envMap(refract(view, normal, eta.r - slide), roughness, 0.0).r;
            ref.b       = envMap(refract(view, normal, eta.b + slide), roughness, 0.0).b;

            bck += ref;
        }
        color.rgb   = bck * pass_step;

        let R = refract(view, normal, eta.g);
        color       = envMap(R, roughness);

        let RaR = refract(view, normal, eta.r);
        let RaB = refract(view, normal, eta.b);
        color.r     = envMap(RaR, roughness).r;
        color.b     = envMap(RaB, roughness).b;

    return color*T*T*T*T;
}
