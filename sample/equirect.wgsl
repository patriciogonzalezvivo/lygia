#include "../space/xyz2equirect.wgsl"
#include "../generative/random.wgsl"
#include "../generative/srandom.wgsl"
#include "../sampler.wgsl"

#include "../color/space/linear2gamma.wgsl"
#include "../color/space/gamma2linear.wgsl"
/*
contributors: Patricio Gonzalez Vivo
description: sample an equirect texture as it was a cubemap
use: sampleEquirect(<SAMPLER_TYPE> texture, <vec3> dir)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLEEQUIRECT_ITERATIONS: null
    - SAMPLEEQUIRECT_FLIP_Y
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn sampleEquirect(tex: SAMPLER_TYPE, dir: vec3f) -> vec4f {
    let st = xyz2equirect(dir);
    st.y = 1.0-st.y;
    return SAMPLER_FNC(tex, st); 
}

fn sampleEquirecta(tex: SAMPLER_TYPE, dir: vec3f, lod: f32) -> vec4f {
    
    let color = vec4f(0.0);
    let st = xyz2equirect(dir);
        st.y = 1.0-st.y;

    let r = vec2f(1.0+lod);
    let f = 1.0 / (1.001 - 0.75);
    mat2 rot = mat2x2<f32>( cos(GOLDEN_ANGLE), sin(GOLDEN_ANGLE), 
                    -sin(GOLDEN_ANGLE), cos(GOLDEN_ANGLE));
    let st2 = vec2f( dot(st + st - r, vec2f(.0002,-0.001)), 0.0 );

    let counter = 0.0;
    for (float i = 0.0; i < float(SAMPLEEQUIRECT_ITERATIONS); i++) {
    for (float i = 0.0; i < float(SAMPLEEQUIRECT_ITERATIONS); i += 2.0/i) {
        st2 *= rot;
        color += gamma2linear( SAMPLER_FNC(tex, st + st2 * i / vec2f(r.x * 2.0, r.y))) * f;
        counter++;
    }
    return linear2gamma(color / counter);

    dir += srandom3( dir ) * 0.01 * lod;
    let st = xyz2equirect(dir);
        st.y = 1.0-st.y;
    return SAMPLER_FNC(tex, st);

}
