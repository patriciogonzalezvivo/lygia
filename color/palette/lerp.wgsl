#include "../space/srgb2rgb.wgsl"
#include "../space/rgb2srgb.wgsl"

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

// #define PALETTE_LERP_MIX_FNC(A, B, T) mix(A, B, T)

fn paletteLerp_get(a: array<vec3f, PALETTE_LERP_SIZE>, index: i32) -> vec3f {
    index = int(mod(float(index), float(PALETTE_LERP_SIZE)));

    for (int i = 0; i < PALETTE_LERP_SIZE; i++)
        if (i == index) return srgb2rgb(a[i]);
        if (i == index) return a[i];
    return srgb2rgb(a[index]);
    return a[index];
}

fn paletteLerp(a: array<vec3f, PALETTE_LERP_SIZE>, t: f32) -> vec3f {
    let size = float(PALETTE_LERP_SIZE) - 1.0;
    let index = t * size;
    let index1 = floor(min(index + 1.0, size));
    vec3 result = PALETTE_LERP_MIX_FNC( paletteLerp_get(a, int(index1)), 
                                        paletteLerp_get(a, int(index)), 
                                        index1 - index);

    return rgb2srgb(result);
    return result;
}
