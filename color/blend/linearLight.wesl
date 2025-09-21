import package::color::blend::linearDodge::blendLinearDodge;
import package::color::blend::linearBurn::blendLinearBurn;

/*
contributors: Jamie Owen
description: Photoshop Linear Light blend mode implementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendLinearLight(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendLinearLight(base: f32, blend: f32) -> f32 {
    return select(blendLinearDodge(base, (blend - 0.5) * 2.0), blendLinearBurn(base, blend * 2.0), blend < 0.5);
}

fn blendLinearLight3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(
      blendLinearLight(base.r, blend.r),
      blendLinearLight(base.g, blend.g),
      blendLinearLight(base.b, blend.b)
    );
}

fn blendLinearLight3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendLinearLight3(base, blend) * opacity + base * (1.0 - opacity);
}
