#include "rgb2hcv.wgsl"
#include "hue2rgb.wgsl"

/*
contributors:
    - David Schaeffer
    - tobspr
    - Patricio Gonzalez Vivo
description: 'Convert from linear RGB to HCY (Hue, Chroma, Luminance)

  HCY is a cylindrica. From: https://github.com/tobspr/GLSL-Color-Spaces/blob/master/ColorSpaces.inc.glsl'
license:
    - MIT License (MIT) Copyright (c) 2015 tobspr
*/


fn rgb2hcy(rgb: vec3f) -> vec3f {
    let HCYwts = vec3f(0.299, 0.587, 0.114);
    // Corrected by David Schaeffer
    var HCV = rgb2hcv(rgb);
    let Y = dot(rgb, HCYwts);
    let Z = dot(hue2rgb(HCV.x), HCYwts);
    if (Y < Z) {
        HCV.y *= Z / (HCY_EPSILON + Y);
    } else {
        HCV.y *= (1.0 - Z) / (HCY_EPSILON + 1.0 - Y);
    }
    return vec3f(HCV.x, HCV.y, Y);
}
