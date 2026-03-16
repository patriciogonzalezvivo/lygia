#include "../math/saturate.wgsl"
/*
contributors: Patricio Gonzalez Vivo
description: Convert diffuse/specular/glossiness workflow to PBR metallic factor
use: <float> toMetallic(<vec3> diffuse, <vec3> specular, <float> maxSpecular)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const TOMETALLIC_MIN_REFLECTANCE: f32 = 0.04;

fn toMetallic3(diffuse: vec3f, specular: vec3f, maxSpecular: f32) -> f32 {
    let perceivedDiffuse = sqrt(0.299 * diffuse.r * diffuse.r + 0.587 * diffuse.g * diffuse.g + 0.114 * diffuse.b * diffuse.b);
    let perceivedSpecular = sqrt(0.299 * specular.r * specular.r + 0.587 * specular.g * specular.g + 0.114 * specular.b * specular.b);
    if (perceivedSpecular < TOMETALLIC_MIN_REFLECTANCE) {
        return 0.0;
    }
    let a = TOMETALLIC_MIN_REFLECTANCE;
    let b = perceivedDiffuse * (1.0 - maxSpecular) / (1.0 - TOMETALLIC_MIN_REFLECTANCE) + perceivedSpecular - 2.0 * TOMETALLIC_MIN_REFLECTANCE;
    let c = TOMETALLIC_MIN_REFLECTANCE - perceivedSpecular;
    let D = max(b * b - 4.0 * a * c, 0.0);
    return saturate((-b + sqrt(D)) / (2.0 * a));
}

fn toMetallic3a(diffuse: vec3f, specular: vec3f) -> f32 {
    let maxSpecula = max(max(specular.r, specular.g), specular.b);
    return toMetallic(diffuse, specular, maxSpecula);
}
