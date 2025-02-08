#include "../math/const.glsl"
#include "../math/rotate2d.glsl"
#include "../sampler.glsl"

/*
contributors: Guido Schmidt
description: |
    The Marching Squares algorithm generates contour lines (isolines) of a given input for a
    given grid and a threshold. Is uses a lookup of 16 cases to decide if a cell is inside, outside or
    a specific corner.
    **References:**
    - [urbanspr1nter.github.io](https://urbanspr1nter.github.io/marchingsquares/)
    - []()
use: 
    - <vec3> sampleMarchingSquares(in <vec2> uv, in <sampler2D> tex, in <float> cellSize, in <float> threshold, in <vec2> resolution) 
options:
    - SAMPLEMARCHINGSQUARES_SAMPLE_FNC(TEX, UV): optional sampling function
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/morphological_marchinSquares.frag
*/

#ifndef FNC_SAMPLEMARCHINGSQUARES
#define FNC_SAMPLEMARCHINGSQUARES 

#ifndef SAMPLEMARCHINGSQUARES_SAMPLE_FNC
#define SAMPLEMARCHINGSQUARES_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, clamp(UV, 0.01, 0.99)).r
#endif

float sampleMarchingSquares_outline(in vec2 p, in vec2 cellUv, in vec2 a, in vec2 b, in bool straight) {
    float lineStrength = (straight ? 1.3333 : 1.0);
    vec2 pa = p - a;
    vec2 ba = b - a;
    vec2 line = pa - ba * dot(pa, ba) / dot(ba, ba);
    return 1.0 - step(lineStrength, length(line));
}

float sampleMarchingSquares_tile(in vec2 p, in vec2 cellUv, in vec2 a, in vec2 b, in int tile) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    vec2 line = pa - ba * dot(pa, ba) / dot(ba, ba);
    if (tile == 0)
        return step(0.5, cellUv.x);
    else if (tile == 1)
        return step(0.5, 1.0 - cellUv.x);
    else if (tile == 2)
        return step(0.5, 1.0 - cellUv.y);
    else if (tile == 3)
        return step(0.5, cellUv.y);
    else if (tile == 4)
        return 1.0 - step(0.35, (rotate2d(PI * -0.75) * cellUv).x);
    else if (tile == 5)
        return step(0.35, (rotate2d(PI * -0.75) * cellUv).x);
    else if (tile == 6)
        return 1.0 - step(1.05, (rotate2d(PI * -0.25) * cellUv).x);
    else if (tile == 7)
        return step(1.05, (rotate2d(PI * -0.25) * cellUv).x);
    else if (tile == 8)
        return step(0.35, (rotate2d(PI * -0.25) * cellUv).x);
    else if (tile == 9)
        return 1.0 - step(0.35, (rotate2d(PI * -0.25) * cellUv).x);
    else if (tile == 10)
        return 1.0 - step(0.35, (rotate2d(PI * 0.25) * cellUv).x);
    else if (tile == 11) 
        return step(0.35, (rotate2d(PI * 0.25) * cellUv).x);
    else if (tile == 12) {
        float shape12_0 = 1.0 - step(0.35, (rotate2d(PI * 0.25) * cellUv).x);
        float shape12_1 = step(0.35, (rotate2d(PI * -0.75) * cellUv).x);
        return shape12_0 - shape12_1;
    }
    else if (tile == 13) {
        float shape13_0 = step(0.35, (rotate2d(PI * -0.25) * cellUv).x);
        float shape13_1 = step(1.05, (rotate2d(PI * -0.25) * cellUv).x);
        return shape13_0 - shape13_1;
    }
    else
          return 0.0;
}

vec2 sampleMarchinSquares(in sampler2D tex, in vec2 uv, in vec2 resolution, in float cellSize, in float threshold) {
    float gridX = resolution.x / cellSize;
    float gridY = resolution.y / cellSize;
    float cellIdx = floor(uv.x * gridX);
    float cellIdy = floor(uv.y * gridY);
    vec2 gridUV = vec2((cellIdx + 0.5) * (1.0 / gridX),
                       (cellIdy + 0.5) * (1.0 / gridY));
    vec2 gridUVCells = vec2(fract(uv.x * gridX),
                            fract(uv.y * gridY));

    vec2 pixelSize = 1.0 / resolution;
    vec2 gridStep = pixelSize * (cellSize * 0.5);

    vec2 p_bl = gridUV + vec2(-gridStep.x, +gridStep.y);
    vec2 p_br = gridUV + vec2(+gridStep.x, +gridStep.y);
    vec2 p_tr = gridUV + vec2(+gridStep.x, -gridStep.y);
    vec2 p_tl = gridUV + vec2(-gridStep.x, -gridStep.y);

    float v_bl = SAMPLEMARCHINGSQUARES_SAMPLE_FNC(tex, p_bl);
    float v_br = SAMPLEMARCHINGSQUARES_SAMPLE_FNC(tex, p_br);
    float v_tl = SAMPLEMARCHINGSQUARES_SAMPLE_FNC(tex, p_tl);
    float v_tr = SAMPLEMARCHINGSQUARES_SAMPLE_FNC(tex, p_tr);

    float thr = threshold;
    vec2 ms = vec2(0.0);
    vec2 a, b = vec2(0.0);
    int tile = 0;
    // Everything is inside
    if ((v_bl > thr && v_br > thr && v_tl > thr && v_tr > thr) ||
        (v_bl > thr && v_br > thr && v_tl > thr && v_tr > thr)) {
        ms.r = 1.0;
    }
    // Distance field cuts through the cell vertically
    if (v_bl < thr && v_br >= thr && v_tl < thr && v_tr >= thr) {
        a = mix(p_tl, p_tr, 0.5);
        b = mix(p_bl, p_br, 0.5);
        tile = 0;
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 0);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    if (v_bl >= thr && v_br < thr && v_tl >= thr && v_tr < thr) {
        a = mix(p_tl, p_tr, 0.5);
        b = mix(p_bl, p_br, 0.5);
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 1);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through the cell horizontally
    if ((v_tl < thr && v_tr < thr && v_bl >= thr && v_br >= thr) ||
        (v_tl >= thr && v_tr >= thr && v_bl < thr && v_br < thr)) {
        a = mix(p_bl, p_tl, 0.5);
        b = mix(p_tr, p_br, 0.5);
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 2);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    if (v_tl < thr && v_tr < thr && v_bl >= thr && v_br >= thr) {
        a = mix(p_bl, p_tl, 0.5);
        b = mix(p_tr, p_br, 0.5);
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 3);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through bottom left corner
    if (v_bl < thr && v_tl >= thr && v_tr >= thr && v_br >= thr) {
        a = mix(p_br, p_bl, 0.5);
        b = mix(p_bl, p_tl, 0.5);
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 4);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    if (v_bl >= thr && v_tl < thr && v_tr < thr && v_br < thr) {
        a = mix(p_br, p_bl, 0.5);
        b = mix(p_bl, p_tl, 0.5);
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 5);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through bottom right corner
    if (v_br < thr && v_bl >= thr && v_tl >= thr && v_tr >= thr) {
        a = mix(p_tr, p_br, 0.5);
        b = mix(p_br, p_bl, 0.5);
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 6);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    if (v_br >= thr && v_bl < thr && v_tl < thr && v_tr < thr) {
        a = mix(p_tr, p_br, 0.5);
        b = mix(p_br, p_bl, 0.5);
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 7);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through top left corner
    if (v_tl < thr && v_tr >= thr && v_bl >= thr && v_br >= thr) {
        a = mix(p_bl, p_tl, 0.5);
        b = mix(p_tl, p_tr, 0.5);
        ms.r =  sampleMarchingSquares_tile(uv, gridUVCells, a, b, 8);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    if (v_tl >= thr && v_tr < thr && v_bl < thr && v_br < thr) {
        a = mix(p_bl, p_tl, 0.5);
        b = mix(p_tl, p_tr, 0.5);
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 9);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through top right corner
    if (v_tr < thr && v_tl >= thr && v_bl >= thr && v_br >= thr) {
        a = mix(p_tl, p_tr, 0.5);
        b = mix(p_tr, p_br, 0.5);
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 10);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    if (v_tr >= thr && v_tl < thr && v_bl < thr && v_br < thr) {
        a = mix(p_tl, p_tr, 0.5);
        b = mix(p_tr, p_br, 0.5);
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 11);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through top right and bottom left corner
    if ((v_tl >= thr && v_tr < thr && v_bl < thr && v_br >= thr)) {
        a = mix(p_tl, p_tr, 0.5);
        b = mix(p_tr, p_br, 0.5);
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 12);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through top left and bottom right corner
    if ((v_tl < thr && v_tr >= thr && v_bl >= thr && v_br < thr)) {
        a = mix(p_tl, p_bl, 0.5);
        b = mix(p_tl, p_tr, 0.5);
        ms.r = sampleMarchingSquares_tile(uv, gridUVCells, a, b, 13);
        ms.g = sampleMarchingSquares_outline(uv, gridUVCells, a, b, false);
    }
    return ms;
}
#endif