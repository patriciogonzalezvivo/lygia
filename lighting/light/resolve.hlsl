#include "point.hlsl"
#include "directional.hlsl"
#include "../material.hlsl"

#ifndef FNC_LIGHT_RESOLVE
#define FNC_LIGHT_RESOLVE

void lightResolve(float3 _diffuseColor, float3 _specularColor, Material _M, LightPoint _L, inout float3 _lightDiffuse, inout float3 _lightSpecular)
{
    lightPoint(_diffuseColor, _specularColor, _L, _M, _lightDiffuse, _lightSpecular);
}


void lightResolve(float3 _diffuseColor, float3 _specularColor, Material _M, LightDirectional _L, inout float3 _lightDiffuse, inout float3 _lightSpecular)
{
    lightDirectional(_diffuseColor, _specularColor, _L, _M, _lightDiffuse, _lightSpecular);
}

#endif