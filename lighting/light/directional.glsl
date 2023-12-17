#include "../specular.glsl"
#include "../diffuse.glsl"
#include "../shadow.glsl"
#include "../common/penner.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: calculate directional light
use: lightDirectional(<vec3> _diffuseColor, <vec3> _specularColor, <vec3> _N, <vec3> _V, <float> _NoV, <float> _f0, out <vec3> _diffuse, out <vec3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_DIRECTION
    - LIGHT_COLOR: in GlslViewer is u_lightColor
    - LIGHT_INTENSITY: in GlslViewer is u_lightIntensity
*/

#ifndef STR_LIGHT_DIRECTIONAL
#define STR_LIGHT_DIRECTIONAL
struct LightDirectional {
    vec3    direction;
    vec3    color;
    float   intensity;
};
#endif

#ifndef FNC_LIGHT_DIRECTIONAL
#define FNC_LIGHT_DIRECTIONAL
void lightDirectional(
    const in vec3 _diffuseColor, const in vec3 _specularColor, 
    const in vec3 _V,
    const in vec3 _Ld, const in vec3 _Lc, const in float _Li,
    const in vec3 _N, const in float _NoV, const in float _NoL, 
    const in float _roughness, const in float _f0, 
    inout vec3 _diffuse, inout vec3 _specular) {
    float dif = diffuse(_Ld, _N, _V, _NoV, _NoL, _roughness);
    float spec = specular(_Ld, _N, _V, _NoV, _NoL, _roughness, _f0);

    _diffuse  += max(vec3(0.0), _Li * (_diffuseColor * _Lc * dif));
    _specular += max(vec3(0.0), _Li * (_specularColor * _Lc * spec));
}

#ifdef STR_MATERIAL
void lightDirectional(
    const in vec3 _diffuseColor, const in vec3 _specularColor,
    LightDirectional _L, const in Material _mat, 
    inout vec3 _diffuse, inout vec3 _specular) {

    float f0    = max(_mat.f0.r, max(_mat.f0.g, _mat.f0.b));
    float NoL   = dot(_mat.normal, _L.direction);

    lightDirectional(
        _diffuseColor, _specularColor, 
        _mat.V, 
        _L.direction, _L.color, _L.intensity,
        _mat.normal, _mat.NoV, NoL, _mat.roughness, f0, 
        _diffuse, _specular);

    #ifdef SHADING_MODEL_SUBSURFACE
    vec3  h     = normalize(_mat.V + _L.direction);
    float NoH   = saturate(dot(_mat.normal, h));
    float LoH   = saturate(dot(_L.direction, h));

    float scatterVoH = saturate(dot(_mat.V, -_L.direction));
    float forwardScatter = exp2(scatterVoH * _mat.subsurfacePower - _mat.subsurfacePower);
    float backScatter = saturate(NoL * _mat.subsurfaceThickness + (1.0 - _mat.subsurfaceThickness)) * 0.5;
    float subsurface = mix(backScatter, 1.0, forwardScatter) * (1.0 - _mat.subsurfaceThickness);
    _diffuse += _mat.subsurfaceColor * (subsurface * diffuseLambert());
    #endif
}
#endif

#endif