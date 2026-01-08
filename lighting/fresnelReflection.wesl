import lygia::lighting::fresnel::fresnel;

/*
contributors: Patricio Gonzalez Vivo
description: "Resolve fresnel coefficient and apply it to a reflection. It can apply\
    \ iridescence to \nusing a formula based on https://www.alanzucconi.com/2017/07/25/the-mathematics-of-thin-film-interference/\n"
use:
    - <vec3f> fresnelReflection(<vec3f> R, <vec3f> f0, <f32> NoV)
    - <vec3f> fresnelReflection(<vec3f> R, <vec3f> Fr)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// Default wavelengths for Fresnel reflection (RGB: 612nm, 549nm, 464nm)
const FRESNEL_REFLECTION_RGB: vec3f = vec3f(612.0, 549.0, 464.0);

// Note: fresnelReflection requires envMap function which is not yet available in WESL
// This is a stub implementation that will need envMap to be functional
// Deferred until LATER - envMap, ior, and Material/ShadingData structs not yet available

fn fresnelReflection(R: vec3f, f0: vec3f, NoV: f32) -> vec3f {
    let frsnl = fresnel(f0, NoV);  // Uses fresnel(vec3f, f32)
    // TODO: Replace with actual envMap call when available
    // let reflectColor = envMap(R, 1.0, 0.001);
    let reflectColor = vec3f(0.0);
    return reflectColor * frsnl;
}

fn fresnelReflection(R: vec3f, Fr: vec3f) -> vec3f {
    // TODO: Replace with actual envMap call when available
    // let reflectColor = envMap(R, 1.0, 0.001);
    let reflectColor = vec3f(0.0);
    return reflectColor * Fr;
}

// Note: fresnelIridescentReflection overloads (6 variants) deferred until LATER
// Note: fresnelReflection(Material, ShadingData) overload deferred until LATER
