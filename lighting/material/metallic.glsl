#include "../toMetallic.glsl"
#include "specular.glsl"

/*
author: Patricio Gonzalez Vivo
description: get material metalic property from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines 
use: vec4 materialMetallic()
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_MATERIAL_METALLIC
#define FNC_MATERIAL_METALLIC

#ifdef MATERIAL_METALLICMAP
uniform sampler2D MATERIAL_METALLICMAP;
#endif

#if defined(MATERIAL_ROUGHNESSMETALLICMAP) && !defined(MATERIAL_ROUGHNESSMETALLICMAP_UNIFORM)
#define MATERIAL_ROUGHNESSMETALLICMAP_UNIFORM
uniform sampler2D MATERIAL_ROUGHNESSMETALLICMAP;
#endif

#if defined(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP) && !defined(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP_UNIFORM)
#define MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP_UNIFORM
uniform sampler2D MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP;
#endif
    
float materialMetallic() {
    float metallic = 0.0;

#if defined(MATERIAL_METALLICMAP) && defined(MODEL_VERTEX_TEXCOORD)
    vec2 uv = v_texcoord.xy;
    #if defined(MATERIAL_METALLICMAP_OFFSET)
    uv += (MATERIAL_METALLICMAP_OFFSET).xy;
    #endif
    #if defined(MATERIAL_METALLICMAP_SCALE)
    uv *= (MATERIAL_METALLICMAP_SCALE).xy;
    #endif
    metallic = texture2D(MATERIAL_METALLICMAP, uv).b;

#elif defined(MATERIAL_ROUGHNESSMETALLICMAP) && defined(MODEL_VERTEX_TEXCOORD)
    vec2 uv = v_texcoord.xy;
    metallic = texture2D(MATERIAL_ROUGHNESSMETALLICMAP, uv).b;

#elif defined(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP) && defined(MODEL_VERTEX_TEXCOORD)
    vec2 uv = v_texcoord.xy;
    metallic = texture2D(MATERIAL_OCCLUSIONROUGHNESSMETALLICMAP, uv).b;

#elif defined(MATERIAL_METALLIC)
    metallic = MATERIAL_METALLIC;

#else
    vec3 diffuse = materialBaseColor().rgb;
    vec3 specular = materialSpecular();
    metallic = toMetallic(diffuse, specular);
#endif

    return metallic;
}

#endif