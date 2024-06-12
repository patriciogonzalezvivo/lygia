/*
contributors: Sam Hocevar
description: Pass a color in RGB and get HSB color. From http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
use: rgb2hsv(<vec3|vec4> color)
*/

#ifndef HCV_EPSILON
#define HCV_EPSILON 1e-10
#endif

#ifndef FNC_RGB2HSV
#define FNC_RGB2HSV
vec3 rgb2hsv(const in vec3 c) {
    vec4 K = vec4(0., -0.33333333333333333333, 0.6666666666666666666, -1.0);
    vec4 p = c.g < c.b ? vec4(c.bg, K.wz) : vec4(c.gb, K.xy);
    vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx);
    float d = q.x - min(q.w, q.y);
    return vec3(abs(q.z + (q.w - q.y) / (6. * d + HCV_EPSILON)), 
                d / (q.x + HCV_EPSILON), 
                q.x);
}
vec4 rgb2hsv(const in vec4 c) { return vec4(rgb2hsv(c.rgb), c.a); }
#endif