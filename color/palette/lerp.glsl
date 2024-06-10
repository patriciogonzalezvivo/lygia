#include "../space/srgb2rgb.glsl"
#include "../space/rgb2srgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: linear interpolation between colors in a palette
options:
    - PALETTE_LERP_SIZE: number of colors in the palette
    - PALETTE_LERP_MIX_FNC: mix function to use (default is mix)
    - PALETTE_LERP_SRGB: if defined, the palette is in sRGB space
examples:
    - /shaders/color_palette_lerp.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


#ifndef PALETTE_LERP_MIX_FNC
#define PALETTE_LERP_MIX_FNC(A, B, T) mix(A, B, T)
#endif

#ifndef FNC_PALETTE_LERP
#define FNC_PALETTE_LERP

vec3 paletteLerp_get(vec3 a[PALETTE_LERP_SIZE], int index){
    index = int(mod(float(index), float(PALETTE_LERP_SIZE)));

    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < PALETTE_LERP_SIZE; i++)
        #ifdef PALETTE_LERP_SRGB
        if (i == index) return srgb2rgb(a[i]);
        #else
        if (i == index) return a[i];
        #endif
    #else
    #ifdef PALETTE_LERP_SRGB
    return srgb2rgb(a[index]);
    #else
    return a[index];
    #endif
    #endif
}

vec3 paletteLerp(vec3 a[PALETTE_LERP_SIZE], float t){
    float size = float(PALETTE_LERP_SIZE) - 1.0;
    float index = t * size;
    float index1 = floor(min(index + 1.0, size));
    vec3 result = PALETTE_LERP_MIX_FNC( paletteLerp_get(a, int(index1)), 
                                        paletteLerp_get(a, int(index)), 
                                        index1 - index);

    #ifdef PALETTE_LERP_SRGB
    return rgb2srgb(result);
    #else
    return result;
    #endif
}

#endif
