#include "space/rgb2lms.hlsl"
#include "space/lms2rgb.hlsl"

/*
author: Patricio Gonzalez Vivo  
description: daltonize functions based on https://web.archive.org/web/20081014161121/http://www.colorjack.com/labs/colormatrix/ http://www.daltonize.org/search/label/Daltonize
use: <float3|float4> daltonize(<float3|float4> rgb)
options:
    DALTONIZE_FNC: daltonizeProtanope, daltonizeProtanopia, daltonizeProtanomaly, daltonizeDeuteranope, daltonizeDeuteranopia, daltonizeDeuteranomaly, daltonizeTritanope, daltonizeTritanopia, daltonizeTritanomaly, daltonizeAchromatopsia and daltonizeAchromatomaly
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef DALTONIZE_FNC
#define DALTONIZE_FNC daltonizeProtanope
#endif 

#ifndef FNC_DALTONIZE
#define FNC_DALTONIZE

// Protanope - reds are greatly reduced (1% men)
float3 daltonizeProtanope(float3 rgb) {
    float3 lms = rgb2lms(rgb);

    lms.x = 0.0 * lms.x + 2.02344 * lms.y + -2.52581 * lms.z;
    lms.y = 0.0 * lms.x + 1.0 * lms.y + 0.0 * lms.z;
    lms.z = 0.0 * lms.x + 0.0 * lms.y + 1.0 * lms.z;

    return lms2rgb(lms);
}

float4 daltonizeProtanope(float4 rgba) {
    return float4(daltonizeProtanope(rgba.rgb), rgba.a);
}

float3 daltonizeProtanopia(float3 rgb) {
    return float3(rgb.r * 0.56667 + rgb.g * 0.43333 + rgb.b * 0.00000,
                rgb.r * 0.55833 + rgb.g * 0.44267 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.24167 + rgb.b * 0.75833);
}

float4 daltonizeProtanopia(float4 rgba) {
    return float4(daltonizeProtanopia(rgba.rgb), rgba.a);
}

float3 daltonizeProtanomaly(float3 rgb) {
    return float3(rgb.r * 0.81667 + rgb.g * 0.18333 + rgb.b * 0.00000,
                rgb.r * 0.33333 + rgb.g * 0.66667 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.12500 + rgb.b * 0.87500);
}

float4 daltonizeProtanomaly(float4 rgba) {
    return float4(daltonizeProtanomaly(rgba.rgb), rgba.a);
}

// Deuteranope - greens are greatly reduced (1% men)
float3 daltonizeDeuteranope(float3 rgb) {
    float3 lms = rgb2lms(rgb);

    lms.x = 1.0 * lms.x + 0.0 * lms.y + 0.0 * lms.z;
    lms.y = 0.494207 * lms.x + 0.0 * lms.y + 1.24827 * lms.z;
    lms.z = 0.0 * lms.x + 0.0 * lms.y + 1.0 * lms.z;

    return lms2rgb(lms);
}

float4 daltonizeDeuteranope(float4 rgba) {
    return float4(daltonizeDeuteranope(rgba.rgb), rgba.a);
}

float3 daltonizeDeuteranopia(float3 rgb) {
    return float3(rgb.r * 0.62500 + rgb.g * 0.37500 + rgb.b * 0.00000,
                rgb.r * 0.70000 + rgb.g * 0.30000 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.30000 + rgb.b * 0.70000);
}

float4 daltonizeDeuteranopia(float4 rgba) {
    return float4(daltonizeDeuteranopia(rgba.rgb), rgba.a);
}

float3 daltonizeDeuteranomaly(float3 rgb) {
    return float3(rgb.r * 0.80000 + rgb.g * 0.20000 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.25833 + rgb.b * 0.74167,
                rgb.r * 0.00000 + rgb.g * 0.14167 + rgb.b * 0.85833);
}

float4 daltonizeDeuteranomaly(float4 rgba) {
    return float4(daltonizeDeuteranomaly(rgba.rgb), rgba.a);
}


// Tritanope - blues are greatly reduced (0.003% population)
float3 daltonizeTritanope(float3 rgb) {
    float3 lms = rgb2lms(rgb);
    
    // Simulate rgb blindness
    lms.x = 1.0 * lms.x + 0.0 * lms.y + 0.0 * lms.z;
    lms.y = 0.0 * lms.x + 1.0 * lms.y + 0.0 * lms.z;
    lms.z = -0.395913 * lms.x + 0.801109 * lms.y + 0.0 * lms.z;
    
    return lms2rgb(lms);
}

float4 daltonizeTritanope(float4 rgba) {
    return float4(daltonizeTritanope(rgba.rgb), rgba.a);
}

float3 daltonizeTritanopia(float3 rgb) {
    return float3(rgb.r * 0.95 + rgb.g * 0.05 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.43333 + rgb.b * 0.56667,
                rgb.r * 0.00000 + rgb.g * 0.47500 + rgb.b * 0.52500);
}

float4 daltonizeTritanopia(float4 rgba) {
    return float4(daltonizeTritanopia(rgba.rgb), rgba.a);
}

float3 daltonizeTritanomaly(float3 rgb) {
    return float3(rgb.r * 0.96667 + rgb.g * 0.33333 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.73333 + rgb.b * 0.26667,
                rgb.r * 0.00000 + rgb.g * 0.18333 + rgb.b * 0.81667);
}

float4 daltonizeTritanomaly(float4 rgba) {
    return float4(daltonizeTritanomaly(rgba.rgb), rgba.a);
}

float3 daltonizeAchromatopsia(float3 rgb) {
    return float3(rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114,
                rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114,
                rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114);
}

float4 daltonizeAchromatopsia(float4 rgba) {
    return float4(daltonizeAchromatopsia(rgba.rgb), rgba.a);
}

float3 daltonizeAchromatomaly(float3 rgb) {
    return float3(rgb.r * 0.618 + rgb.g * 0.320 + rgb.b * 0.062,
                rgb.r * 0.163 + rgb.g * 0.775 + rgb.b * 0.062,
                rgb.r * 0.163 + rgb.g * 0.320 + rgb.b * 0.516);
}

float4 daltonizeAchromatomaly(float4 rgba) {
    return float4(daltonizeAchromatomaly(rgba.rgb), rgba.a);
}


// GENERAL FUNCTION

float3 daltonize(float3 rgb) {
    return DALTONIZE_FNC(rgb);
}

float4 daltonize( float4 rgba ) {
    return DALTONIZE_FNC(rgba);
}

// From https://gist.github.com/jcdickinson/580b7fb5cc145cee8740
//
float3 daltonizeCorrection(float3 rgb) {
    // Isolate invisible rgbs to rgb vision deficiency (calculate error matrix)
    float3 error = (rgb - daltonize(rgb));

    // Shift rgbs towards visible spectrum (apply error modifications)
    float3 correction;
    correction.r = 0.0; // (error.r * 0.0) + (error.g * 0.0) + (error.b * 0.0);
    correction.g = (error.r * 0.7) + (error.g * 1.0); // + (error.b * 0.0);
    correction.b = (error.r * 0.7) + (error.b * 1.0); // + (error.g * 0.0);

    // Add compensation to original values
    correction = rgb + correction;

    return correction.rgb;
}

float4 daltonizeCorrection(float4 rgb) {
    return float4(daltonizeCorrection( rgb.rgb ), rgb.a);
}

#endif