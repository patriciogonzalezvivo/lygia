/*
original_author: Patricio Gonzalez Vivo  
description: 
use: <float|vec3\vec4> srgb2rgb(<float|vec3|vec4> srgb)
*/


#ifndef SRGB_INVERSE_GAMMA
#define SRGB_INVERSE_GAMMA 2.2
#endif

#ifndef SRGB_ALPHA
#define SRGB_ALPHA 0.055
#endif

#ifndef FNC_SRGB2RGB
#define FNC_SRGB2RGB

float srgb2rgb(float channel) {
    if (channel <= 0.04045)
        return channel * 0.08333333333; // 1.0 / 12.92;
    else
        return pow((channel + SRGB_ALPHA) / (1.0 + SRGB_ALPHA), 2.4);
}

vec3 srgb2rgb(vec3 srgb) {
    #if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) | defined(PLATFORM_WEBGL)
        return pow(srgb, vec3(SRGB_INVERSE_GAMMA));
    #else 
        // return vec3(
        //     srgb2rgb(srgb.r),
        //     srgb2rgb(srgb.g),
        //     srgb2rgb(srgb.b)
        // );

        vec3 srgb_lo = srgb / 12.92;
        vec3 srgb_hi = pow((srgb + SRGB_ALPHA)/(1.0 + SRGB_ALPHA), vec3(2.4));
        return mix(srgb_lo, srgb_hi, step(vec3(0.04045), srgb));
    #endif
}

vec4 srgb2rgb(vec4 srgb) {
    return vec4(srgb2rgb(srgb.rgb), srgb.a);
}

#endif