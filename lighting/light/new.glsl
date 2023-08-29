#include "directional.glsl"
#include "point.glsl"
#include "../shadow.glsl"

#ifndef FNC_LIGHT_NEW
#define FNC_LIGHT_NEW

void lightNew(out LightDirectional _L) {
    #ifdef LIGHT_DIRECTION
    _L.direction    = normalize(LIGHT_DIRECTION);
    #elif defined(LIGHT_POSITION)
    _L.direction    = normalize(LIGHT_POSITION);
    #else
    _L.direction    = normalize(vec3(0.0, 1.0, -1.0));
    #endif

    #ifdef LIGHT_COLOR
    _L.color        = LIGHT_COLOR;
    #else 
    _L.color        = vec3(1.0);
    #endif

    #ifdef LIGHT_INTENSITY
    _L.intensity    = LIGHT_INTENSITY;
    #else
    _L.intensity    = 1.0;
    #endif
    
    _L.shadow       = 1.0;
    #if defined(LIGHT_SHADOWMAP) && defined(LIGHT_SHADOWMAP_SIZE) && defined(LIGHT_COORD)
    _L.shadow *= shadow(LIGHT_SHADOWMAP, vec2(LIGHT_SHADOWMAP_SIZE), (LIGHT_COORD).xy, (LIGHT_COORD).z);
    #endif
}

void lightNew(out LightPoint _L) {
    #if defined(SURFACE_POSITION)
    _L.position     = LIGHT_POSITION - SURFACE_POSITION.xyz;
    #else
    _L.position     = LIGHT_POSITION;
    #endif
    _L.dist         = length(_L.position);
    _L.direction    = _L.position/_L.dist;

    _L.color        = LIGHT_COLOR;
    _L.intensity    = LIGHT_INTENSITY;

    #ifdef LIGHT_COLOR
    _L.color        = LIGHT_COLOR;
    #else 
    _L.color        = vec3(1.0);
    #endif

    #ifdef LIGHT_INTENSITY
    _L.intensity    = LIGHT_INTENSITY;
    #else
    _L.intensity    = 1.0;
    #endif

    #ifdef LIGHT_FALLOFF
    _L.falloff      = LIGHT_FALLOFF;
    #endif
    
    _L.shadow       = 1.0;

    #if defined(LIGHT_SHADOWMAP) && defined(LIGHT_SHADOWMAP_SIZE) && defined(LIGHT_COORD)
    _L.shadow *= shadow(LIGHT_SHADOWMAP, vec2(LIGHT_SHADOWMAP_SIZE), (LIGHT_COORD).xy, (LIGHT_COORD).z);
    #endif
}

LightDirectional LightDirectionalNew() {
    LightDirectional l;
    lightNew(l);
    return l;
}

LightPoint LightPointNew() {
    LightPoint l;
    lightNew(l);
    return l;
}

#endif