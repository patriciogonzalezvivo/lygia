/*
contributors: Patricio Gonzalez Vivo
description: |
    Moves the center from 0.0 to 0.5
use: <float|float2|float3> uncenter(<float|float2|float3> st)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef FNC_UNCENTER
#define FNC_UNCENTER

float  uncenter(float x) { return x * 0.5 + 0.5; }
float2 uncenter(float2 st) { return st * 0.5 + 0.5; }
float3 uncenter(float3 pos) { return pos * 0.5 + 0.5; }

#endif