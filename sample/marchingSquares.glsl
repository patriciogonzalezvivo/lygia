#include "../generative/pnoise.glsl"
#include "../math/const.glsl"
#include "../math/rotate2d.glsl"
#include "../sample.glsl"

/*
Author: Guido Schmidt
description: The Marching Squares algorithm generates contour lines (isolines) of a given input for a
given grid and a threshhold. Is uses a lookup of 16 cases to decide if a cell is inside, outside or
a specific corner.
use: <vec3> marchingSquares(in <vec2> uv) 
options:
    - SAMPLE_MARCHING_SQUARES_FNC(UV): optional sampling function, defaults so sampling of pnoise
*/

#ifndef SAMPLE_MARCHING_SQUARES_FNC
float defaultSampleFunction(in vec2 uv) {
    float dv = pnoise(uv * 5.0, vec2(0.0));
    return dv;
}
#define SAMPLE_MARCHING_SQUARES_FNC(UV) defaultSampleFunction(UV)
#endif

#define centerOf(p1, p2, v1, v2) mix(p1, p2, 0.5)

float sampleOutline(in vec2 p, in vec2 cellUv, in vec2 a, in vec2 b, in bool straight) {
    float lineStrength = (straight ? 1.3333 : 1.0);
    vec2 pa = p - a;
    vec2 ba = b - a;
    vec2 line = pa - ba * dot(pa, ba) / dot(ba, ba);
    return 1.0 - step(lineStrength, length(line));
}

float sampleTile(in vec2 p, in vec2 cellUv, in vec2 a, in vec2 b, in int tile) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    vec2 line = pa - ba * dot(pa, ba) / dot(ba, ba);
    switch (tile) {
        case 0:
          return step(0.5, cellUv.x);
        case 1:
          return step(0.5, 1.0 - cellUv.x);
        case 2:
          return step(0.5, 1.0 - cellUv.y);
        case 3:
          return step(0.5, cellUv.y);
        case 4:
          return 1.0 - step(0.35, (rotate2d(PI * 0.75) * cellUv).x);
        case 5:
          return step(0.35, (rotate2d(PI * 0.75) * cellUv).x);
        case 6:
          return 1.0 - step(1.05, (rotate2d(PI * 0.25) * cellUv).x);
        case 7:
          return step(1.05, (rotate2d(PI * 0.25) * cellUv).x);
        case 8:
          return step(0.35, (rotate2d(PI * 0.25) * cellUv).x);
        case 9:
          return 1.0 - step(0.35, (rotate2d(PI * 0.25) * cellUv).x);
        case 10:
          return 1.0 - step(0.35, (rotate2d(PI * -0.25) * cellUv).x);
        case 11:
          return step(0.35, (rotate2d(PI * -0.25) * cellUv).x);
        case 12:
          float shape12_0 = 1.0 - step(0.35, (rotate2d(PI * -0.25) * cellUv).x);
          float shape12_1 = step(0.35, (rotate2d(PI * 0.75) * cellUv).x);
          return shape12_0 - shape12_1;
        case 13:
          float shape13_0 = step(0.35, (rotate2d(PI * 0.25) * cellUv).x);
          float shape13_1 = step(1.05, (rotate2d(PI * 0.25) * cellUv).x);
          return shape13_0 - shape13_1;
        default:
          return 0.0;
        // @DEBUG: enable the smoothstep in order to show a line for tiles
        //return smoothstep(0.4, 0.0, length(line));
    }
}

vec2 marchinSquares(in vec2 uv, in float cellSize, in float threshold, in vec2 resolution) {
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

    float v_bl = SAMPLE_MARCHING_SQUARES_FNC(p_bl);
    float v_br = SAMPLE_MARCHING_SQUARES_FNC(p_br);
    float v_tl = SAMPLE_MARCHING_SQUARES_FNC(p_tl);
    float v_tr = SAMPLE_MARCHING_SQUARES_FNC(p_tr);

    float thr = threshold;
    vec2 ms = vec2(0.0);

    vec2 a, b = vec2(0.0);
    // Everything is inside
    if ((v_bl > thr && v_br > thr && v_tl > thr && v_tr > thr) ||
        (v_bl > thr && v_br > thr && v_tl > thr && v_tr > thr)) {
        ms.r = 1.0;
    }
    // Distance field cuts through the cell vertically
    if (v_bl < thr && v_br >= thr && v_tl < thr && v_tr >= thr) {
        a = centerOf(p_tl, p_tr, v_tl, v_tr);
        b = centerOf(p_bl, p_br, v_bl, v_br);
        ms.r = sampleTile(uv, gridUVCells, a, b, 0);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    if (v_bl >= thr && v_br < thr && v_tl >= thr && v_tr < thr) {
        a = centerOf(p_tl, p_tr, v_tl, v_tr);
        b = centerOf(p_bl, p_br, v_bl, v_br);
        ms.r = sampleTile(uv, gridUVCells, a, b, 1);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through the cell horizontally
    if ((v_tl < thr && v_tr < thr && v_bl >= thr && v_br >= thr) ||
    (v_tl >= thr && v_tr >= thr && v_bl < thr && v_br < thr)) {
        a = centerOf(p_bl, p_tl, v_tl, v_bl);
        b = centerOf(p_tr, p_br, v_tr, v_br);
        ms.r = sampleTile(uv, gridUVCells, a, b, 2);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    if (v_tl < thr && v_tr < thr && v_bl >= thr && v_br >= thr) {
        a = centerOf(p_bl, p_tl, v_tl, v_bl);
        b = centerOf(p_tr, p_br, v_tr, v_br);
        ms.r = sampleTile(uv, gridUVCells, a, b, 3);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through bottom left corner
    if (v_bl < thr && v_tl >= thr && v_tr >= thr && v_br >= thr) {
        a = centerOf(p_br, p_bl, v_br, v_bl);
        b = centerOf(p_bl, p_tl, v_bl, v_tl);
        ms.r = sampleTile(uv, gridUVCells, a, b, 4);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    if (v_bl >= thr && v_tl < thr && v_tr < thr && v_br < thr) {
        a = centerOf(p_br, p_bl, v_br, v_bl);
        b = centerOf(p_bl, p_tl, v_bl, v_tl);
        ms.r = sampleTile(uv, gridUVCells, a, b, 5);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through bottom right corner
    if (v_br < thr && v_bl >= thr && v_tl >= thr && v_tr >= thr) {
        a = centerOf(p_tr, p_br, v_tr, v_br);
        b = centerOf(p_br, p_bl, v_br, v_bl);
        ms.r = sampleTile(uv, gridUVCells, a, b, 6);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    if (v_br >= thr && v_bl < thr && v_tl < thr && v_tr < thr) {
        a = centerOf(p_tr, p_br, v_tr, v_br);
        b = centerOf(p_br, p_bl, v_br, v_bl);
        ms.r = sampleTile(uv, gridUVCells, a, b, 7);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through top left corner
    if (v_tl < thr && v_tr >= thr && v_bl >= thr && v_br >= thr) {
        a = centerOf(p_bl, p_tl, v_bl, v_tl);
        b = centerOf(p_tl, p_tr, v_tl, v_tr);
        ms.r =  sampleTile(uv, gridUVCells, a, b, 8);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    if (v_tl >= thr && v_tr < thr && v_bl < thr && v_br < thr) {
        a = centerOf(p_bl, p_tl, v_bl, v_tl);
        b = centerOf(p_tl, p_tr, v_tl, v_tr);
        ms.r = sampleTile(uv, gridUVCells, a, b, 9);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through top right corner
    if (v_tr < thr && v_tl >= thr && v_bl >= thr && v_br >= thr) {
        a = centerOf(p_tl, p_tr, v_tl, v_tr);
        b = centerOf(p_tr, p_br, v_tr, v_br);
        ms.r = sampleTile(uv, gridUVCells, a, b, 10);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    if (v_tr >= thr && v_tl < thr && v_bl < thr && v_br < thr) {
        a = centerOf(p_tl, p_tr, v_tl, v_tr);
        b = centerOf(p_tr, p_br, v_tr, v_br);
        ms.r = sampleTile(uv, gridUVCells, a, b, 11);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through top right and bottom left corner
    if ((v_tl >= thr && v_tr < thr && v_bl < thr && v_br >= thr)) {
        a = centerOf(p_tl, p_tr, v_tl, v_tr);
        b = centerOf(p_tr, p_br, v_tr, v_br);
        ms.r = sampleTile(uv, gridUVCells, a, b, 12);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    // Distance field cuts through top left and bottom right corner
    if ((v_tl < thr && v_tr >= thr && v_bl >= thr && v_br < thr)) {
        a = centerOf(p_tl, p_bl, v_tl, v_bl);
        b = centerOf(p_tl, p_tr, v_tl, v_tr);
        ms.r = sampleTile(uv, gridUVCells, a, b, 13);
        ms.g = sampleOutline(uv, gridUVCells, a, b, false);
    }
    return ms;
}
