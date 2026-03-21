#include "space/rgb2lms.wgsl"
#include "space/lms2rgb.wgsl"

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

// #define DALTONIZE_FNC daltonizeProtanope

// Protanope - reds are greatly reduced (1% men)
fn daltonizeProtanope3(rgb: vec3f) -> vec3f {
    let lms = rgb2lms(rgb);

    lms.x = 0.0 * lms.x + 2.02344 * lms.y + -2.52581 * lms.z;
    lms.y = 0.0 * lms.x + 1.0 * lms.y + 0.0 * lms.z;
    lms.z = 0.0 * lms.x + 0.0 * lms.y + 1.0 * lms.z;

    return lms2rgb(lms);
}

fn daltonizeProtanope4(rgba: vec4f) -> vec4f {
    return vec4f(daltonizeProtanope(rgba.rgb), rgba.a);
}

fn daltonizeProtanopia3(rgb: vec3f) -> vec3f {
    return vec3f(rgb.r * 0.56667 + rgb.g * 0.43333 + rgb.b * 0.00000,
                rgb.r * 0.55833 + rgb.g * 0.44267 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.24167 + rgb.b * 0.75833);
}

fn daltonizeProtanopia4(rgba: vec4f) -> vec4f {
    return vec4f(daltonizeProtanopia(rgba.rgb), rgba.a);
}

fn daltonizeProtanomaly3(rgb: vec3f) -> vec3f {
    return vec3f(rgb.r * 0.81667 + rgb.g * 0.18333 + rgb.b * 0.00000,
                rgb.r * 0.33333 + rgb.g * 0.66667 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.12500 + rgb.b * 0.87500);
}

fn daltonizeProtanomaly4(rgba: vec4f) -> vec4f {
    return vec4f(daltonizeProtanomaly(rgba.rgb), rgba.a);
}

// Deuteranope - greens are greatly reduced (1% men)
fn daltonizeDeuteranope3(rgb: vec3f) -> vec3f {
    let lms = rgb2lms(rgb);

    lms.x = 1.0 * lms.x + 0.0 * lms.y + 0.0 * lms.z;
    lms.y = 0.494207 * lms.x + 0.0 * lms.y + 1.24827 * lms.z;
    lms.z = 0.0 * lms.x + 0.0 * lms.y + 1.0 * lms.z;

    return lms2rgb(lms);
}

fn daltonizeDeuteranope4(rgba: vec4f) -> vec4f {
    return vec4f(daltonizeDeuteranope(rgba.rgb), rgba.a);
}

fn daltonizeDeuteranopia3(rgb: vec3f) -> vec3f {
    return vec3f(rgb.r * 0.62500 + rgb.g * 0.37500 + rgb.b * 0.00000,
                rgb.r * 0.70000 + rgb.g * 0.30000 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.30000 + rgb.b * 0.70000);
}

fn daltonizeDeuteranopia4(rgba: vec4f) -> vec4f {
    return vec4f(daltonizeDeuteranopia(rgba.rgb), rgba.a);
}

fn daltonizeDeuteranomaly3(rgb: vec3f) -> vec3f {
    return vec3f(rgb.r * 0.80000 + rgb.g * 0.20000 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.25833 + rgb.b * 0.74167,
                rgb.r * 0.00000 + rgb.g * 0.14167 + rgb.b * 0.85833);
}

fn daltonizeDeuteranomaly4(rgba: vec4f) -> vec4f {
    return vec4f(daltonizeDeuteranomaly(rgba.rgb), rgba.a);
}

// Tritanope - blues are greatly reduced (0.003% population)
fn daltonizeTritanope3(rgb: vec3f) -> vec3f {
    let lms = rgb2lms(rgb);
    
    // Simulate rgb blindness
    lms.x = 1.0 * lms.x + 0.0 * lms.y + 0.0 * lms.z;
    lms.y = 0.0 * lms.x + 1.0 * lms.y + 0.0 * lms.z;
    lms.z = -0.395913 * lms.x + 0.801109 * lms.y + 0.0 * lms.z;
    
    return lms2rgb(lms);
}

fn daltonizeTritanope4(rgba: vec4f) -> vec4f {
    return vec4f(daltonizeTritanope(rgba.rgb), rgba.a);
}

fn daltonizeTritanopia3(rgb: vec3f) -> vec3f {
    return vec3f(rgb.r * 0.95 + rgb.g * 0.05 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.43333 + rgb.b * 0.56667,
                rgb.r * 0.00000 + rgb.g * 0.47500 + rgb.b * 0.52500);
}

fn daltonizeTritanopia4(rgba: vec4f) -> vec4f {
    return vec4f(daltonizeTritanopia(rgba.rgb), rgba.a);
}

fn daltonizeTritanomaly3(rgb: vec3f) -> vec3f {
    return vec3f(rgb.r * 0.96667 + rgb.g * 0.33333 + rgb.b * 0.00000,
                rgb.r * 0.00000 + rgb.g * 0.73333 + rgb.b * 0.26667,
                rgb.r * 0.00000 + rgb.g * 0.18333 + rgb.b * 0.81667);
}

fn daltonizeTritanomaly4(rgba: vec4f) -> vec4f {
    return vec4f(daltonizeTritanomaly(rgba.rgb), rgba.a);
}

fn daltonizeAchromatopsia3(rgb: vec3f) -> vec3f {
    return vec3f(rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114,
                rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114,
                rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114);
}

fn daltonizeAchromatopsia4(rgba: vec4f) -> vec4f {
    return vec4f(daltonizeAchromatopsia(rgba.rgb), rgba.a);
}

fn daltonizeAchromatomaly3(rgb: vec3f) -> vec3f {
    return vec3f(rgb.r * 0.618 + rgb.g * 0.320 + rgb.b * 0.062,
                rgb.r * 0.163 + rgb.g * 0.775 + rgb.b * 0.062,
                rgb.r * 0.163 + rgb.g * 0.320 + rgb.b * 0.516);
}

fn daltonizeAchromatomaly4(rgba: vec4f) -> vec4f {
    return vec4f(daltonizeAchromatomaly(rgba.rgb), rgba.a);
}

// GENERAL FUNCTION

fn daltonize3(rgb: vec3f) -> vec3f {
    return DALTONIZE_FNC(rgb);
}

fn daltonize4(rgba: vec4f) -> vec4f {
    return DALTONIZE_FNC(rgba);
}

// From https://gist.github.com/jcdickinson/580b7fb5cc145cee8740
//
fn daltonizeCorrection3(rgb: vec3f) -> vec3f {
    // Isolate invisible rgbs to rgb vision deficiency (calculate error matrix)
    let error = (rgb - daltonize(rgb));

    // Shift rgbs towards visible spectrum (apply error modifications)
    var correction: vec3f;
    correction.r = 0.0; // (error.r * 0.0) + (error.g * 0.0) + (error.b * 0.0);
    correction.g = (error.r * 0.7) + (error.g * 1.0); // + (error.b * 0.0);
    correction.b = (error.r * 0.7) + (error.b * 1.0); // + (error.g * 0.0);

    // Add compensation to original values
    correction = rgb + correction;

    return correction.rgb;
}

fn daltonizeCorrection4(rgb: vec4f) -> vec4f {
    return vec4f(daltonizeCorrection( rgb.rgb ), rgb.a);
}
