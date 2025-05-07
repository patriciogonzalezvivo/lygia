#include "space/rgb2luma.wgsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Get the luminosity of a color. From https://github.com/hughsk/glsl-luma/blob/master/index.glsl
*/


fn luma(color: vec3f) -> f32 {
    return rgb2luma(color);
}