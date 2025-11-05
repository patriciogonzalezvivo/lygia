import lygia::color::space::rgb2luma::rgb2luma;

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Get the luminosity of a color. From https://github.com/hughsk/glsl-luma/blob/master/index.glsl
use: luma(<f32|vec3f|vec4f> color)
*/

fn luma(value: f32) -> f32 {
    return value;
}

fn luma3(color: vec3f) -> f32 {
    return rgb2luma(color);
}

fn luma4(color: vec4f) -> f32 {
    return rgb2luma(color.rgb);
}
