/*
contributors: 
    - Sam Hocevar
    - Patricio Gonzalez Vivo
description: Pass a color in RGB and get HSB color. From http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
use: <float> rgb2hue(<float3|float4> color)
*/

#ifndef HUE_EPSILON
#define HUE_EPSILON 1e-10
#endif

#ifndef FNC_RGB2HUE
#define FNC_RGB2HUE
float rgb2hue(in float3 c) {
    float4 K = float4(0.0, -0.33333333333333333333, 0.6666666666666666666, -1.0);
    float4 p = c.g < c.b ? float4(c.bg, K.wz) : float4(c.gb, K.xy);
    float4 q = c.r < p.x ? float4(p.xyw, c.r) : float4(c.r, p.yzx);
    float d = q.x - min(q.w, q.y);
    return abs(q.z + (q.w - q.y) / (6.0 * d + HUE_EPSILON));
}

float rgb2hue(in float4 c) { return rgb2hue(c.rgb); }
#endif