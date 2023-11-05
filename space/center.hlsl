/*
contributors: Patricio Gonzalez Vivo
description: |
    It center the coordinates from 0 to 1 to -1 to 1
    So the center goes from 0.5 to 0.0. 
use: <float|float2|float3> center(<float|float2|float3> st)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef FNC_CENTER
#define FNC_CENTER

float  center(float x) { return x * 2.0 - 1.0; }
float2 center(float2 st) { return st * 2.0 - 1.0; }
float3 center(float3 pos) { return pos * 2.0 - 1.0; }

#endif