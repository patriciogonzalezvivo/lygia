#include "tonemap/aces.glsl"
#include "tonemap/debug.glsl"
#include "tonemap/linear.glsl"
#include "tonemap/reinhard.glsl"
#include "tonemap/uncharted.glsl"
#include "tonemap/unreal.glsl"

/*
author: Patricio Gonzalez Vivo  
description: Tone maps the specified RGB color (meaning convert from HDR to LDR) inside the range [0..~8] to [0..1]. The input must be in linear HDR pre-exposed.
use: tonemap(<vec3|vec4> rgb)
options:
    TONEMAP_FNC: tonemapLinear, tonemapReinhard, tonemapUnreal, tonemapACES, tonemapDebug, tonemapUncharter
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef TONEMAP_FNC
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) || defined(PLATFORM_WEBGL)
    #define TONEMAP_FNC     tonemapUnreal
#else
    #define TONEMAP_FNC     tonemapACES
#endif
#endif

#ifndef FNC_TONEMAP
#define FNC_TONEMAP

vec3 tonemap(const vec3 color) { return TONEMAP_FNC(color); }
vec4 tonemap(const vec4 color) { return TONEMAP_FNC(color); }

#endif