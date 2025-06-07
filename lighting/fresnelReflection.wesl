import lygia::lighting::{fresnel, envMap};

/*
contributors: Patricio Gonzalez Vivo
description: "Resolve fresnel coefficient and apply it to a reflection. It can apply\
    \ iridescence to \nusing a formula based on https://www.alanzucconi.com/2017/07/25/the-mathematics-of-thin-film-interference/\n"
use:
    - <vec3> fresnelReflection(<vec3> R, <vec3> f0, <float> NoV)
    - <vec3> fresnelIridescentReflection(<vec3> normal, <vec3> view, <vec3> f0, <vec3> ior1, <vec3> ior2, <float> thickness, <float> roughness)
    - <vec3> fresnelReflection(<Material> _M)
options:
    - FRESNEL_REFLECTION_RGB: <vec3> RGB values of the reflection
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn fresnelReflection(R: vec3f, f0: vec3f, NoV: f32) -> vec3f {
    let frsnl = fresnel(f0, NoV);

    let reflectColor = vec3f(0.0);
    reflectColor = envMap(R, 1.0, 0.001);

    return reflectColor * frsnl;
}