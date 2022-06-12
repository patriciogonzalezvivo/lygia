#include "../math/powFast.glsl"
#include "../math/saturate.glsl"
#include "../color/tonemap.glsl"
#include "../sample/textureShadowPCF.glsl"
#include "material.glsl"
#include "fresnel.glsl"

#include "envMap.glsl"
#include "sphericalHarmonics.glsl"
#include "diffuse.glsl"
#include "specular.glsl"

/*
author: Patricio Gonzalez Vivo
description: simple PBR shading model
use: 
    - <vec4> pbrLittle(<Material> material) 
    - <vec4> pbrLittle(<vec4> baseColor, <vec3> normal, <float> roughness, <float> metallic [, <vec3> f0] ) 
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
    - SURFACE_POSITION: in glslViewer is v_position
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef SURFACE_POSITION
#define SURFACE_POSITION v_position
#endif

#ifndef CAMERA_POSITION
#if defined(GLSLVIEWER)
#define CAMERA_POSITION u_camera
#else
#define CAMERA_POSITION vec3(0.0, 0.0, -10.0);
#endif
#endif


#ifndef LIGHT_POSITION
#if defined(GLSLVIEWER)
#define LIGHT_POSITION u_light
#else
#define LIGHT_POSITION vec3(0.0, 10.0, -50.0)
#endif
#endif

#ifndef LIGHT_COLOR
#if defined(GLSLVIEWER)
#define LIGHT_COLOR     u_lightColor
#else
#define LIGHT_COLOR     vec3(0.5)
#endif
#endif

#ifndef FNC_PBR_LITTLE
#define FNC_PBR_LITTLE

vec4 pbrLittle(vec4 baseColor, vec3 normal, float roughness, float metallic, vec3 f0 ) {
    vec3 L = normalize(LIGHT_POSITION - (SURFACE_POSITION).xyz);
    vec3 N = normalize(normal);
    vec3 V = normalize(CAMERA_POSITION - (SURFACE_POSITION).xyz);

    float notMetal = 1. - metallic;
    float smooth = .95 - saturate(roughness);

    // DIFFUSE
    float diffuse = diffuse(L, N, V, roughness);
    float specular = specular(L, N, V, roughness);

#if defined(LIGHT_SHADOWMAP) && defined(LIGHT_SHADOWMAP_SIZE) && defined(LIGHT_COORD) && !defined(PLATFORM_RPI)
    float bias = 0.005;
    float shadow = textureShadowPCF(LIGHT_SHADOWMAP, vec2(LIGHT_SHADOWMAP_SIZE), (LIGHT_COORD).xy, (LIGHT_COORD).z - bias);
    specular *= shadow;
    diffuse *= shadow;
#endif
    
    baseColor.rgb = baseColor.rgb * diffuse;
#ifdef SCENE_SH_ARRAY
    baseColor.rgb *= tonemapReinhard( sphericalHarmonics(N) );
#endif

    float NoV = dot(N, V); 
    vec3 F = fresnel(f0, NoV);

    // SPECULAR
    vec3 specIntensity =    vec3(1.0) *
                            (0.04 * notMetal + 2.0 * metallic) * 
                            F *
                            // saturate(-1.1 + NoV + metallic) * // Fresnel
                            (metallic + smooth * 4.0); // make smaller highlights brighter

    vec3 R = reflect(-V, N);
    vec3 ambientSpecular = vec3(0.0);
    ambientSpecular += tonemapReinhard( envMap(R, roughness, metallic) ) * specIntensity;
    ambientSpecular += F * metallic;

    baseColor.rgb = baseColor.rgb * notMetal + ( ambientSpecular 
                    + LIGHT_COLOR * 2.0 * specular
                    ) * (notMetal * smooth + baseColor.rgb * metallic);

    return baseColor;
}

vec4 pbrLittle(vec4 baseColor, vec3 normal, float roughness, float metallic) {
    return pbrLittle(baseColor, normal, roughness, metallic, vec3(0.04));
}

vec4 pbrLittle(Material material) {
    return pbrLittle(material.baseColor, material.normal, material.roughness, material.metallic) + vec4(material.emissive, 0.0);
}

#endif