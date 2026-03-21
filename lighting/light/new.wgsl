#include "directional.wgsl"
#include "point.wgsl"
#include "../shadow.wgsl"

fn lightNew(_L: LightDirectional) {
    _L.direction    = normalize(LIGHT_DIRECTION);
    _L.direction    = normalize(LIGHT_POSITION);
    _L.direction    = normalize(vec3f(0.0, 1.0, -1.0));

    _L.color        = LIGHT_COLOR;
    _L.color        = vec3f(1.0);

    _L.intensity    = LIGHT_INTENSITY;
    _L.intensity    = 1.0;
    
    _L.intensity *= shadow(LIGHT_SHADOWMAP, vec2f(LIGHT_SHADOWMAP_SIZE), (LIGHT_COORD).xy, (LIGHT_COORD).z);
}

LightDirectional LightDirectionalNew() { LightDirectional l; lightNew(l); return l; }

fn lightNewa(_L: LightPoint) {
    _L.position     = LIGHT_POSITION - SURFACE_POSITION.xyz;
    _L.position     = LIGHT_POSITION;
    
    _L.color        = LIGHT_COLOR;
    _L.color        = vec3f(1.0);

    _L.intensity    = LIGHT_INTENSITY;
    _L.intensity    = 1.0;

    _L.falloff      = LIGHT_FALLOFF;
    _L.falloff      = 0.0;

    _L.intensity *= shadow(LIGHT_SHADOWMAP, vec2f(LIGHT_SHADOWMAP_SIZE), (LIGHT_COORD).xy, (LIGHT_COORD).z);
}

LightPoint LightPointNew() {
    LightPoint l;
    lightNew(l);
    return l;
}
