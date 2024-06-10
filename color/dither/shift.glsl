#include "../../math/decimate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Vlachos 2016, "Advanced VR Rendering" http://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdfs'
use: <vec4|vec3|float> ditherShift(<vec4|vec3|float> value, <float> time)
options:
    - DITHER_SHIFT_TIME
    - DITHER_SHIFT_CHROMATIC
examples:
    - /shaders/color_dither.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef DITHER_SHIFT_COORD
#define DITHER_SHIFT_COORD gl_FragCoord.xy
#endif

#ifdef DITHER_TIME
#define DITHER_SHIFT_TIME DITHER_TIME
#endif

#ifdef DITHER_CHROMATIC
#define DITHER_SHIFT_CHROMATIC
#endif

#ifndef DITHER_SHIFT_PRECISION
#ifdef DITHER_PRECISION
#define DITHER_SHIFT_PRECISION DITHER_PRECISION
#else
#define DITHER_SHIFT_PRECISION 256
#endif
#endif

#ifndef FNC_DITHER_SHIFT
#define FNC_DITHER_SHIFT

float ditherShift(const in float b, const in vec2 st, const int pres) {
    //Bit-depth of display. Normally 8 but some LCD monitors are 7 or even 6-bit.   
    float dither_bit = 8.0; 

    #ifdef DITHER_SHIFT_TIME 
    st += 1337.0*fract(DITHER_SHIFT_TIME);
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

vec3 ditherShift(const in vec3 color, const in vec2 st, const int pres) {
    //Bit-depth of display. Normally 8 but some LCD monitors are 7 or even 6-bit.	
    float dither_bit = 8.0; 

    // Calculate grid position
    #ifdef DITHER_SHIFT_TIME 
    st += 1337.0*fract(DITHER_SHIFT_TIME);
    #endif
    float grid_position = fract( dot( st - vec2(0.5,0.5) , vec2(1.0/16.0,10.0/36.0) + 0.25 ) );

    //Calculate how big the shift should be
    float dither_shift = (0.25) * (1.0 / (pow(2.0,dither_bit) - 1.0));

    //Shift the individual colors differently, thus making it even harder to see the dithering pattern
    #ifdef DITHER_SHIFT_CHROMATIC
    vec3 ditherPattern = vec3(dither_shift, -dither_shift, dither_shift);
    #else
    vec3 ditherPattern = vec3(dither_shift);
    #endif

    //modify shift acording to grid position.
    ditherPattern = mix(2.0 * ditherPattern, -2.0 * ditherPattern, grid_position); //shift acording to grid position.

    //shift the color by dither_shift

    float d = float(pres);
    float h = 0.5/d;
    return decimate(color + h + ditherPattern, d);
}

float ditherShift(const in float value, const in vec2 xy) {  return ditherShift(value, xy, DITHER_SHIFT_PRECISION); }
vec3 ditherShift(const in vec3 color, const in vec2 xy) {  return ditherShift(color, xy, DITHER_SHIFT_PRECISION); }
vec4 ditherShift(const in vec4 color, const in vec2 xy) {  return vec4(ditherShift(color.rgb, xy, DITHER_SHIFT_PRECISION), color.a); }

float ditherShift(const in float val, int pres) { return ditherShift(vec3(val),DITHER_SHIFT_COORD, pres).r; }
vec3 ditherShift(const in vec3 color, int pres) { return ditherShift(color, DITHER_SHIFT_COORD, pres); }
vec4 ditherShift(const in vec4 color, int pres) { return vec4(ditherShift(color.rgb, DITHER_SHIFT_COORD, pres), color.a); }

float ditherShift(const in float val) { return ditherShift(vec3(val), DITHER_SHIFT_COORD, DITHER_SHIFT_PRECISION).r; }
vec3 ditherShift(const in vec3 color) { return ditherShift(color, DITHER_SHIFT_COORD, DITHER_SHIFT_PRECISION); }
vec4 ditherShift(const in vec4 color) { return vec4(ditherShift(color.rgb), color.a); }

#endif