/*
contributors: Patricio Gonzalez Vivo
description: 'Vlachos 2016, "Advanced VR Rendering" http://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf'
use: <float4|float3|float> ditherShift(<float4|float3|float> value, <float> time)
options:
    - DITHER_SHIFT_ANIMATED
    - DITHER_SHIFT_CHROMATIC
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifdef DITHER_CHROMATIC
#define DITHER_SHIFT_CHROMATIC
#endif


#ifdef DITHER_ANIMATED
#define DITHER_SHIFT_ANIMATED
#endif

#ifndef FNC_DITHER_SHIFT
#define FNC_DITHER_SHIFT

float ditherShift(float b, float2 fragcoord, float time) {
    //Bit-depth of display. Normally 8 but some LCD monitors are 7 or even 6-bit.	
    float dither_bit = 8.0; 

    #ifdef DITHER_SHIFT_ANIMATED 
    fragcoord += 1337.0*frac(time);
    #endif
    //Calculate grid position
    float grid_position = frac( dot( fragcoord - float2(0.5,0.5) , float2(1.0/16.0,10.0/36.0) + 0.25 ) );

    //Calculate how big the shift should be
    float dither_shift = (0.25) * (1.0 / (pow(2.0,dither_bit) - 1.0));

    //modify shift acording to grid position.
    dither_shift = lerp(2.0 * dither_shift, -2.0 * dither_shift, grid_position); //shift acording to grid position.

    //shift the color by dither_shift
    return b + 0.5/255.0 + dither_shift; 
}

float3 ditherShift(float3 rgb, float2 fragcoord, float time) {
    //Bit-depth of display. Normally 8 but some LCD monitors are 7 or even 6-bit.	
    float dither_bit = 8.0; 

    //Calculate grid position
    #ifdef DITHER_SHIFT_ANIMATED 
    fragcoord += 1337.0*frac(time);
    #endif
    float grid_position = frac( dot( fragcoord - float2(0.5, 0.5) , float2(1.0/16.0,10.0/36.0) + 0.25 ) );

    //Calculate how big the shift should be
    float dither_shift = (0.25) * (1.0 / (pow(2.0,dither_bit) - 1.0));

    //Shift the individual colors differently, thus making it even harder to see the dithering pattern
    #ifdef DITHER_SHIFT_CHROMATIC
    float3 dither_shift_RGB = float3(dither_shift, -dither_shift, dither_shift);
    #else
    float3 dither_shift_RGB = float3(dither_shift, dither_shift, dither_shift);
    #endif

    //modify shift acording to grid position.
    dither_shift_RGB = lerp(2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position); //shift acording to grid position.

    //shift the color by dither_shift
    return rgb + 0.5/255.0 + dither_shift_RGB; 
}

float4 ditherShift(float4 rgba, float2 fragcoord, float time) {
    return float4(ditherShift(rgba.rgb, fragcoord, time), rgba.a);
}

#endif