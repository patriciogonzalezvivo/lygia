#include "../../color/space/gamma2linear.glsl"

/*
author: Patricio Gonzalez Vivo
description: get material BaseColor from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines 
use: vec4 materialBaseColor()
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef FNC_MATERIAL_BASECOLOR
#define FNC_MATERIAL_BASECOLOR

#ifdef MATERIAL_BASECOLORMAP
uniform sampler2D MATERIAL_BASECOLORMAP;
#endif

vec4 materialBaseColor() {
    vec4 base = vec4(1.0);
    
#if defined(MATERIAL_BASECOLORMAP) && defined(MODEL_VERTEX_TEXCOORD)
    vec2 uv = v_texcoord.xy;
    #if defined(MATERIAL_BASECOLORMAP_OFFSET)
    uv += (MATERIAL_BASECOLORMAP_OFFSET).xy;
    #endif
    #if defined(MATERIAL_BASECOLORMAP_SCALE)
    uv *= (MATERIAL_BASECOLORMAP_SCALE).xy;
    #endif
    base = gamma2linear( SAMPLER_FNC(MATERIAL_BASECOLORMAP, uv) );
    
#elif defined(MATERIAL_BASECOLOR)
    base = MATERIAL_BASECOLOR;

#endif

#if defined(MODEL_VERTEX_COLOR)
    base *= v_color;
#endif

    return base;
}

#endif