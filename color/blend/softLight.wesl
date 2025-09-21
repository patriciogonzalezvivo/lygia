/*
contributors: Jamie Owen
description: Photoshop Soft Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendSoftLight(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

fn blendSoftLight(base: f32, blend: f32) -> f32 {
    if (blend < 0.5) {
        return (2. * base * blend + base * base * (1. - 2.*blend));
    } else {
        return (sqrt(base) * (2. * blend - 1.) + 2. * base * (1. - blend));
    }
}

fn blendSoftLight3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3(blendSoftLight(base.r, blend.r),
                blendSoftLight(base.g, blend.g),
                blendSoftLight(base.b, blend.b));
}

fn blendSoftLight4(base: vec4f, blend: vec4f) -> vec4f {
    return vec4(blendSoftLight( base.r, blend.r ),
                blendSoftLight( base.g, blend.g ),
                blendSoftLight( base.b, blend.b ),
                blendSoftLight( base.a, blend.a )
    );
}

fn blendSoftLight3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return (blendSoftLight3(base, blend) * opacity + base * (1. - opacity));
}
