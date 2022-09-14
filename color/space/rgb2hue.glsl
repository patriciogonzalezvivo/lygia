/*
original_author: Sam Hocevar
description: pass a color in RGB and get HSB color. From http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
use: <float> rgb2hue(<vec3|vec4> color)
*/

#ifndef FNC_RGB2HUE
#define FNC_RGB2HUE
float rgb2hue(in vec3 c) {
    vec4 K = vec4(0., -.33333333333333333333, .6666666666666666666, -1.);

#ifdef RGB2HSV_MIX
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
#else
    vec4 p = c.g < c.b ? vec4(c.bg, K.wz) : vec4(c.gb, K.xy);
    vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx);
#endif

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return abs(q.z + (q.w - q.y) / (6. * d + e));
}

float rgb2hue(in vec4 c) {
    return rgb2hue(c.rgb);
}
#endif