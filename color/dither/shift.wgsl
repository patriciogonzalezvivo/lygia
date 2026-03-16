#include "../../math/decimate.wgsl"

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

// #define DITHER_SHIFT_COORD gl_FragCoord.xy

// #define DITHER_SHIFT_TIME DITHER_TIME

// #define DITHER_SHIFT_CHROMATIC

// #define DITHER_SHIFT_PRECISION DITHER_PRECISION
const DITHER_SHIFT_PRECISION: f32 = 256;

fn ditherShift(b: f32, st: vec2f, pres: i32) -> f32 {
    //Bit-depth of display. Normally 8 but some LCD monitors are 7 or even 6-bit.   
    let dither_bit = 8.0;

    st += 1337.0*fract(DITHER_SHIFT_TIME);
    //Calculate grid position
    let grid_position = fract( dot( st - vec2f(0.5,0.5) , vec2f(1.0/16.0,10.0/36.0) + 0.25 ) );

    //Calculate how big the shift should be
    let dither_shift = (0.25) * (1.0 / (pow(2.0,dither_bit) - 1.0));

    //modify shift according to grid position.
    dither_shift = mix(2.0 * dither_shift, -2.0 * dither_shift, grid_position); //shift according to grid position.

    //shift the color by dither_shift
    return b + 0.5/255.0 + dither_shift; 
}

fn ditherShift3(color: vec3f, st: vec2f, pres: i32) -> vec3f {
    //Bit-depth of display. Normally 8 but some LCD monitors are 7 or even 6-bit.	
    let dither_bit = 8.0;

    // Calculate grid position
    st += 1337.0*fract(DITHER_SHIFT_TIME);
    let grid_position = fract( dot( st - vec2f(0.5,0.5) , vec2f(1.0/16.0,10.0/36.0) + 0.25 ) );

    //Calculate how big the shift should be
    let dither_shift = (0.25) * (1.0 / (pow(2.0,dither_bit) - 1.0));

    //Shift the individual colors differently, thus making it even harder to see the dithering pattern
    let ditherPattern = vec3f(dither_shift, -dither_shift, dither_shift);
    let ditherPattern = vec3f(dither_shift);

    //modify shift according to grid position.
    ditherPattern = mix(2.0 * ditherPattern, -2.0 * ditherPattern, grid_position); //shift according to grid position.

    //shift the color by dither_shift

    let d = float(pres);
    let h = 0.5/d;
    return decimate(color + h + ditherPattern, d);
}

fn ditherShifta(value: f32, xy: vec2f) -> f32 {  return ditherShift(value, xy, DITHER_SHIFT_PRECISION); }
fn ditherShift3a(color: vec3f, xy: vec2f) -> vec3f {  return ditherShift(color, xy, DITHER_SHIFT_PRECISION); }
fn ditherShift4(color: vec4f, xy: vec2f) -> vec4f {  return vec4f(ditherShift(color.rgb, xy, DITHER_SHIFT_PRECISION), color.a); }

fn ditherShiftb(val: f32, pres: i32) -> f32 { return ditherShift(vec3f(val),DITHER_SHIFT_COORD, pres).r; }
fn ditherShift3b(color: vec3f, pres: i32) -> vec3f { return ditherShift(color, DITHER_SHIFT_COORD, pres); }
fn ditherShift4a(color: vec4f, pres: i32) -> vec4f { return vec4f(ditherShift(color.rgb, DITHER_SHIFT_COORD, pres), color.a); }

fn ditherShiftc(val: f32) -> f32 { return ditherShift(vec3f(val), DITHER_SHIFT_COORD, DITHER_SHIFT_PRECISION).r; }
fn ditherShift3c(color: vec3f) -> vec3f { return ditherShift(color, DITHER_SHIFT_COORD, DITHER_SHIFT_PRECISION); }
fn ditherShift4b(color: vec4f) -> vec4f { return vec4f(ditherShift(color.rgb), color.a); }
