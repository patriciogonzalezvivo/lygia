/*
original_author: Inigo Quiles
description: pass a color in HSB and get RGB color. Also use as reference http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
use: hsv2rgb(<float3|float4> color)
*/

#ifndef FNC_HSV2RGB
#define FNC_HSV2RGB
float3 hsv2rgb(in float3 hsb) {
    float3 rgb = clamp(abs(fmod(hsb.x * 6. + float3(0., 4., 2.), 
                            6.) - 3.) - 1.,
                      0.,
                      1.);
    #ifdef HSV2RGB_SMOOTH
    rgb = rgb*rgb*(3. - 2. * rgb);
    #endif
    return hsb.z * lerp(float3(1., 1., 1.), rgb, hsb.y);
}

float4 hsv2rgb(in float4 hsb) {
    return float4(hsv2rgb(hsb.rgb), hsb.a);
}
#endif
