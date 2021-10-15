/*
author: Patricio Gonzalez Vivo
description: Generic Material Structure
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef STR_MATERIAL
#define STR_MATERIAL
struct material {
    vec4    baseColor;
    vec3    emissive;
    vec3    normal;
    
    vec3    f0;// = vec3(0.04);
    float   reflectance;// = 0.5;

    float   roughness;
    float   metallic;

    float   ambientOcclusion;

#if defined(MATERIAL_CLEARCOAT_THICKNESS)
    float   clearCoat;
    float   clearCoatRoughness;
    #if defined(MATERIAL_CLEARCOAT_THICKNESS_NORMAL)
    vec3    clearCoatNormal;// = vec3(0.0, 0.0, 1.0);
    #endif
#endif

#if defined(SHADING_MODEL_SUBSURFACE)
    float   thickness; // = 0.5;
    float   subsurfacePower; // = 12.234;
#endif

#if defined(SHADING_MODEL_CLOTH)
    vec3    sheenColor;
#endif

#if defined(MATERIAL_SUBSURFACE_COLOR)
    vec3    subsurfaceColor; // = vec3(1.0);
#endif

#if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
    vec3    specularColor;
    float   glossiness;
#endif

#if defined(SHADING_SHADOWS)
    float   shadows;// = 1.0;
#endif
};
#endif