/*
original_author: Patricio Gonzalez Vivo
description: |
    Vlachos 2016, "Advanced VR Rendering" http://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
use: <vec4|vec3|float> ditherShift(<vec4|vec3|float> value, <float> time)
options:
    - DITHER_SHIFT_ANIMATED
    - DITHER_SHIFT_CHROMATIC
examples:
    - /shaders/color_dither.frag
*/

#ifdef DITHER_CHROMATIC
#define DITHER_SHIFT_CHROMATIC
#endif


#ifdef DITHER_ANIMATED
#define DITHER_SHIFT_ANIMATED
#endif

#ifndef DITHER_SHIFT
#define DITHER_SHIFT

float ditherShift(float b, float time) {
    //Bit-depth of display. Normally 8 but some LCD monitors are 7 or even 6-bit.	
    float dither_bit = 8.0; 

    vec2 st = gl_FragCoord.xy;
    #ifdef DITHER_SHIFT_ANIMATED 
    st += 1337.0*fract(time);
    #endif
    //Calculate grid position
    float grid_position = fract( dot( st - vec2(0.5,0.5) , vec2(1.0/16.0,10.0/36.0) + 0.25 ) );

    //Calculate how big the shift should be
    float dither_shift = (0.25) * (1.0 / (pow(2.0,dither_bit) - 1.0));

    //modify shift acording to grid position.
    dither_shift = mix(2.0 * dither_shift, -2.0 * dither_shift, grid_position); //shift acording to grid position.

    //shift the color by dither_shift
    return b + 0.5/255.0 + dither_shift; 
}

vec3 ditherShift(vec3 rgb, float time) {
    //Bit-depth of display. Normally 8 but some LCD monitors are 7 or even 6-bit.	
    float dither_bit = 8.0; 

    //Calculate grid position
    vec2 st = gl_FragCoord.xy;
    #ifdef DITHER_SHIFT_ANIMATED 
    st += 1337.0*fract(time);
    #endif
    float grid_position = fract( dot( st - vec2(0.5,0.5) , vec2(1.0/16.0,10.0/36.0) + 0.25 ) );

    //Calculate how big the shift should be
    float dither_shift = (0.25) * (1.0 / (pow(2.0,dither_bit) - 1.0));

    //Shift the individual colors differently, thus making it even harder to see the dithering pattern
    #ifdef DITHER_SHIFT_CHROMATIC
    vec3 dither_shift_RGB = vec3(dither_shift, -dither_shift, dither_shift);
    #else
    vec3 dither_shift_RGB = vec3(dither_shift);
    #endif

    //modify shift acording to grid position.
    dither_shift_RGB = mix(2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position); //shift acording to grid position.

    //shift the color by dither_shift
    return rgb + 0.5/255.0 + dither_shift_RGB; 
}

vec4 ditherShift(vec4 rgba, float time) {
    return vec4(ditherShift(rgba.rgb, time), rgba.a);
}

#endif