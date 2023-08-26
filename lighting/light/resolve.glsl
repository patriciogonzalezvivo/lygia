#include "point.glsl"
#include "directional.glsl"
#include "../material.glsl"

#ifndef FNC_LIGHT_RESOLVE
#define FNC_LIGHT_RESOLVE

void lightResolve(vec3 _diffuseColor, vec3 _specularColor, Material _M, LightPoint _L, inout vec3 _lightDiffuse, inout vec3 _lightSpecular) {
    lightPoint(_diffuseColor, _specularColor, _L, _M, _lightDiffuse, _lightSpecular);
}


void lightResolve(vec3 _diffuseColor, vec3 _specularColor, Material _M, LightDirectional _L, inout vec3 _lightDiffuse, inout vec3 _lightSpecular) {
    lightDirectional(_diffuseColor, _specularColor, _L, _M, _lightDiffuse, _lightSpecular);
}

#endif