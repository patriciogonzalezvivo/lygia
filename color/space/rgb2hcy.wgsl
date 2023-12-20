#include "rgb2hcv.wgsl"
#include "hue2rgb.wgsl"

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
