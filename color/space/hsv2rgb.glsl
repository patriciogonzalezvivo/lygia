/*
original_author: Inigo Quiles
description: pass a color in HSB and get RGB color. Also use as reference http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
use: hsv2rgb(<vec3|vec4> color)
*/

#ifndef FNC_HSV2RGB
#define FNC_HSV2RGB
vec3 hsv2rgb(in vec3 hsb) {
    vec3 rgb = clamp(abs(mod(hsb.x * 6. + vec3(0., 4., 2.), 
                            6.) - 3.) - 1.,
                      0.,
                      1.);
    #ifdef HSV2RGB_SMOOTH
    rgb = rgb*rgb*(3. - 2. * rgb);
    #endif
    return hsb.z * mix(vec3(1.), rgb, hsb.y);
}

vec4 hsv2rgb(in vec4 hsb) {
    return vec4(hsv2rgb(hsb.rgb), hsb.a);
}
#endif