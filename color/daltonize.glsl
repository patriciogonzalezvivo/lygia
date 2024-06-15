#include "space/rgb2lms.glsl"
#include "space/lms2rgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Daltonize functions based on https://web.archive.org/web/20081014161121/http://www.colorjack.com/labs/colormatrix/ http://www.daltonize.org/search/label/Daltonize
use: <vec3|vec4> daltonize(<vec3|vec4> rgb)
options:
    - DALTONIZE_FNC: |
        daltonizeProtanope, daltonizeProtanopia, daltonizeProtanomaly, daltonizeDeuteranope, daltonizeDeuteranopia, daltonizeDeuteranomaly, daltonizeTritanope, daltonizeTritanopia,
        daltonizeTritanomaly, daltonizeAchromatopsia and daltonizeAchromatomaly
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_daltonize.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef DALTONIZE_FNC
#define DALTONIZE_FNC daltonizeProtanope
#endif 

#ifndef FNC_DALTONIZE
#define FNC_DALTONIZE

// Protanope - reds are greatly reduced (1% men)
vec3 daltonizeProtanope(vec3 rgb) {
    vec3 lms = rgb2lms(rgb);

    lms.x = 0.0 * lms.x + 2.02344 * lms.y + -2.52581 * lms.z;
    lms.y = 0.0 * lms.x + 1.0 * lms.y + 0.0 * lms.z;
    lms.z = 0.0 * lms.x + 0.0 * lms.y + 1.0 * lms.z;

    return lms2rgb(lms);
}

vec4 daltonizeProtanope(vec4 rgba) {
    return vec4(daltonizeProtanope(rgba.rgb), rgba.a);
}

vec3 daltonizeProtanopia(vec3 rgb) {
    return vec3(rgb.r * 0.56667 + rgb.g * 0.43333 + rgb.b * 0.00000,
                rgb.r * 0.55833 + rgb.g * 0.44267 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.24167 + rgb.b * 0.75833);
}

vec4 daltonizeProtanopia(vec4 rgba) {
    return vec4(daltonizeProtanopia(rgba.rgb), rgba.a);
}

vec3 daltonizeProtanomaly(vec3 rgb) {
    return vec3(rgb.r * 0.81667 + rgb.g * 0.18333 + rgb.b * 0.00000,
                rgb.r * 0.33333 + rgb.g * 0.66667 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.12500 + rgb.b * 0.87500);
}

vec4 daltonizeProtanomaly(vec4 rgba) {
    return vec4(daltonizeProtanomaly(rgba.rgb), rgba.a);
}

// Deuteranope - greens are greatly reduced (1% men)
vec3 daltonizeDeuteranope(vec3 rgb) {
    vec3 lms = rgb2lms(rgb);

    lms.x = 1.0 * lms.x + 0.0 * lms.y + 0.0 * lms.z;
    lms.y = 0.494207 * lms.x + 0.0 * lms.y + 1.24827 * lms.z;
    lms.z = 0.0 * lms.x + 0.0 * lms.y + 1.0 * lms.z;

    return lms2rgb(lms);
}

vec4 daltonizeDeuteranope(vec4 rgba) {
    return vec4(daltonizeDeuteranope(rgba.rgb), rgba.a);
}

vec3 daltonizeDeuteranopia(vec3 rgb) {
    return vec3(rgb.r * 0.62500 + rgb.g * 0.37500 + rgb.b * 0.00000,
                rgb.r * 0.70000 + rgb.g * 0.30000 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.30000 + rgb.b * 0.70000);
}

vec4 daltonizeDeuteranopia(vec4 rgba) {
    return vec4(daltonizeDeuteranopia(rgba.rgb), rgba.a);
}

vec3 daltonizeDeuteranomaly(vec3 rgb) {
    return vec3(rgb.r * 0.80000 + rgb.g * 0.20000 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.25833 + rgb.b * 0.74167,
                rgb.r * 0.00000 + rgb.g * 0.14167 + rgb.b * 0.85833);
}

vec4 daltonizeDeuteranomaly(vec4 rgba) {
    return vec4(daltonizeDeuteranomaly(rgba.rgb), rgba.a);
}


// Tritanope - blues are greatly reduced (0.003% population)
vec3 daltonizeTritanope(vec3 rgb) {
    vec3 lms = rgb2lms(rgb);
    
    // Simulate rgb blindness
    lms.x = 1.0 * lms.x + 0.0 * lms.y + 0.0 * lms.z;
    lms.y = 0.0 * lms.x + 1.0 * lms.y + 0.0 * lms.z;
    lms.z = -0.395913 * lms.x + 0.801109 * lms.y + 0.0 * lms.z;
    
    return lms2rgb(lms);
}

vec4 daltonizeTritanope(vec4 rgba) {
    return vec4(daltonizeTritanope(rgba.rgb), rgba.a);
}

vec3 daltonizeTritanopia(vec3 rgb) {
    return vec3(rgb.r * 0.95 + rgb.g * 0.05 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.43333 + rgb.b * 0.56667,
                rgb.r * 0.00000 + rgb.g * 0.47500 + rgb.b * 0.52500);
}

vec4 daltonizeTritanopia(vec4 rgba) {
    return vec4(daltonizeTritanopia(rgba.rgb), rgba.a);
}

vec3 daltonizeTritanomaly(vec3 rgb) {
    return vec3(rgb.r * 0.96667 + rgb.g * 0.33333 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.73333 + rgb.b * 0.26667,
                rgb.r * 0.00000 + rgb.g * 0.18333 + rgb.b * 0.81667);
}

vec4 daltonizeTritanomaly(vec4 rgba) {
    return vec4(daltonizeTritanomaly(rgba.rgb), rgba.a);
}

vec3 daltonizeAchromatopsia(vec3 rgb) {
    return vec3(rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114,
                rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114,
                rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114);
}

vec4 daltonizeAchromatopsia(vec4 rgba) {
    return vec4(daltonizeAchromatopsia(rgba.rgb), rgba.a);
}

vec3 daltonizeAchromatomaly(vec3 rgb) {
    return vec3(rgb.r * 0.618 + rgb.g * 0.320 + rgb.b * 0.062,
                rgb.r * 0.163 + rgb.g * 0.775 + rgb.b * 0.062,
                rgb.r * 0.163 + rgb.g * 0.320 + rgb.b * 0.516);
}

vec4 daltonizeAchromatomaly(vec4 rgba) {
    return vec4(daltonizeAchromatomaly(rgba.rgb), rgba.a);
}


// GENERAL FUNCTION

vec3 daltonize(vec3 rgb) {
    return DALTONIZE_FNC(rgb);
}

vec4 daltonize( vec4 rgba ) {
    return DALTONIZE_FNC(rgba);
}

// From https://gist.github.com/jcdickinson/580b7fb5cc145cee8740
//
vec3 daltonizeCorrection(vec3 rgb) {
    // Isolate invisible rgbs to rgb vision deficiency (calculate error matrix)
    vec3 error = (rgb - daltonize(rgb));

    // Shift rgbs towards visible spectrum (apply error modifications)
    vec3 correction;
    correction.r = 0.0; // (error.r * 0.0) + (error.g * 0.0) + (error.b * 0.0);
    correction.g = (error.r * 0.7) + (error.g * 1.0); // + (error.b * 0.0);
    correction.b = (error.r * 0.7) + (error.b * 1.0); // + (error.g * 0.0);

    // Add compensation to original values
    correction = rgb + correction;

    return correction.rgb;
}

vec4 daltonizeCorrection(vec4 rgb) {
    return vec4(daltonizeCorrection( rgb.rgb ), rgb.a);
}

#endif