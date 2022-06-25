#include "../math/saturate.glsl"
#include "../sample/textureShadowPCF.glsl"

#include "material.glsl"
#include "light/point.glsl"

#include "common/reflection.glsl"
#include "common/specularAO.glsl"
#include "common/envBRDFApprox.glsl"

// #include "light/point.glsl"

/*
author: Patricio Gonzalez Vivo
description: simple PBR shading model
use: <vec4> pbr( <Material> _material ) 
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
    - SURFACE_POSITION: in glslViewer is v_position
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
#define LIGHT_POSITION  u_light
#else
#define LIGHT_POSITION  vec3(0.0, 10.0, -50.0)
#endif
#endif

#ifndef LIGHT_COLOR
#if defined(GLSLVIEWER)
#define LIGHT_COLOR     u_lightColor
#else
#define LIGHT_COLOR     vec3(0.5)
#endif
#endif

#ifndef LIGHT_COORD
#if defined(GLSLVIEWER)
#define LIGHT_COLOR     u_lightColor
#else
#define LIGHT_COLOR     vec3(0.5)
#endif
#endif

#ifndef IBL_LUMINANCE
#if defined(GLSLVIEWER)
#define IBL_LUMINANCE   u_iblLuminance
#else
#define IBL_LUMINANCE   1.0
#endif
#endif

#ifndef FNC_PBR
#define FNC_PBR

vec4 pbr(const Material _mat) {
    // Calculate Color

    vec3    diffuseColor = _mat.baseColor.rgb * (vec3(1.0) - _mat.f0) * (1.0 - _mat.metallic);
    vec3    specularColor = mix(_mat.f0, _mat.baseColor.rgb, _mat.metallic);

    vec3    N = _mat.normal;                             // Normal
    vec3    V = normalize(u_camera - v_position.xyz);   // View
    float NoV = dot(N, V);                            // Normal . View
    float f0  = max(_mat.f0.r, max(_mat.f0.g, _mat.f0.b));
    float roughness = _mat.roughness;
    
    // Reflect
    vec3    R = reflection(V, N, roughness);

    // Ambient Occlusion
    // ------------------------
    float ssao = 1.0;
#ifdef SCENE_SSAO
    ssao = texture2D(SCENE_SSAO, gl_FragCoord.xy/u_resolution).r;
#endif 
    float diffuseAO = min(_mat.ambientOcclusion, ssao);
    float specularAO = specularAO(NoV, diffuseAO, roughness);

    // Global Ilumination ( mage Based Lighting )
    // ------------------------
    vec3 E = envBRDFApprox(specularColor, NoV, roughness);

    vec3 Fr = vec3(0.0);
    Fr = tonemapReinhard( envMap(R, roughness, _mat.metallic) ) * E;
    Fr += fresnel(R, _mat.f0, NoV) * _mat.metallic * (1.0-roughness) * 0.2;
    Fr *= specularAO;

    vec3 Fd = vec3(0.0);
    Fd = diffuseColor;
    #if defined(SCENE_SH_ARRAY)
    Fd *= tonemapReinhard( sphericalHarmonics(N) );
    #endif
    Fd *= diffuseAO;
    Fd *= (1.0 - E);

    // Local Ilumination
    // ------------------------
    vec3 lightDiffuse = vec3(0.0);
    vec3 lightSpecular = vec3(0.0);
    // lightWithShadow(diffuseColor, specularColor, N, V, NoV, roughness, f0, lightDiffuse, lightSpecular);
    
    {
        // calcPointLight(_comp, lightDiffuse, lightSpecular);
        // calcDirectionalLight(_comp, lightDiffuse, lightSpecular);

        float shadow = 1.0;
    #if defined(LIGHT_SHADOWMAP) && defined(LIGHT_SHADOWMAP_SIZE) && defined(LIGHT_COORD) && !defined(PLATFORM_RPI)
        // shadow = shadow();
        // shadosw = textureShadowPCF();
        shadow = textureShadowPCF(LIGHT_COORD.xyz);
    #endif

        lightPoint(diffuseColor, specularColor, N, V, NoV, roughness, f0, shadow, lightDiffuse, lightSpecular);
    }
    
    // Final Sum
    // ------------------------
    vec4 color = vec4(0.0);
    color.rgb += Fd * IBL_LUMINANCE + lightDiffuse;     // Diffuse
    color.rgb += Fr * IBL_LUMINANCE + lightSpecular;    // Specular
    color.rgb *= _mat.ambientOcclusion;
    color.rgb += _mat.emissive;
    color.a = _mat.baseColor.a;

    color = linear2gamma( color );

    return color;
}
#endif