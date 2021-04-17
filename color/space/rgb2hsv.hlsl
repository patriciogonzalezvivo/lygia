/*
author: Sam Hocevar
description: pass a color in RGB and get HSB color. From http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
use: rgb2hsv(<float3|float4> color)
licence: TODO
*/

#ifndef FNC_RGB2HSV
#define FNC_RGB2HSV
float3 rgb2hsv(in float3 c) {
    float4 K = float4(0., -.33333333333333333333, .6666666666666666666, -1.);

#ifdef RGB2HSV_MIX
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
#else
    vec4 p = c.g < c.b ? vec4(c.bg, K.wz) : vec4(c.gb, K.xy);
    vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx);
#endif

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(  abs(q.z + (q.w - q.y) / (6. * d + e)), 
                    d / (q.x + e), 
                    q.x );
}

float4 rgb2hsv(in float4 c) {
    return float4(rgb2hsv(c.rgb), c.a);
}
#endif