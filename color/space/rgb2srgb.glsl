/*
original_author: Patricio Gonzalez Vivo  
description: 
use: <float|vec3\vec4> rgb2srgb(<float|vec3|vec4> srgb)
*/


#ifndef SRGB_GAMMA
#define SRGB_GAMMA 0.4545454545
#endif

#ifndef SRGB_ALPHA
#define SRGB_ALPHA 0.055
#endif

#ifndef FNC_RGB2SRGB
#define FNC_RGB2SRGB

float rgb2srgb(float channel) {
    if (channel <= 0.0031308)
        return 12.92 * channel;
    else
        return (1.0 + SRGB_ALPHA) * pow(channel, 0.4166666666666667) - SRGB_ALPHA;
}

vec3 rgb2srgb(vec3 rgb) {
    #if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) | defined(PLATFORM_WEBGL)
        return pow(rgb, vec3(SRGB_GAMMA));
    #else 

        // return vec3(
        //     rgb2srgb(srgb.r),
        //     rgb2srgb(srgb.g),
        //     rgb2srgb(srgb.b)
        // );

        vec3 rgb_lo = 12.92 * rgb;
        vec3 rgb_hi = (1.0 + SRGB_ALPHA) * pow(rgb, vec3(0.4166666666666667)) - SRGB_ALPHA;

        vec3 color = vec3(0.0);
        color.r = color.r < 0.0031308 ? rgb_lo.x : rgb_hi.x;
        color.g = color.g < 0.0031308 ? rgb_lo.y : rgb_hi.y;
        color.b = color.b < 0.0031308 ? rgb_lo.z : rgb_hi.z;
        return color;

        // return mix(rgb_lo, rgb_hi, step(vec3(0.0031308), rgb));

    #endif
}

vec4 rgb2srgb(vec4 rgb) {
    return vec4(rgb2srgb(rgb.rgb), rgb.a);
}

#endif