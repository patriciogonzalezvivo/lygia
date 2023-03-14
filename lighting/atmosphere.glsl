#include "../math/const.glsl"
#include "../math/saturate.glsl"

#include "ray.glsl"
#include "common/henyeyGreenstein.glsl"
#include "common/rayleigh.glsl"

/*
description: |
    Rayleigh and Mie scattering atmosphere system. Implementation of the techniques described here: https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/simulating-sky/simulating-colors-of-the-sky
use: <vec3> atmosphere(<vec3> eye_dir, <vec3> sun_dir)
OPTIONS:
    ATMOSPHERE_FAST: use fast implementation from https://www.shadertoy.com/view/3dBSDW
    ATMOSPHERE_RADIUS_MIN: planet radius
    ATMOSPHERE_RADIUS_MAX: atmosphere radious
    ATMOSPHERE_SUN_POWER: sun power. Default 20.0
    ATMOSPHERE_LIGHT_SAMPLES: Defualt 8 
    ATMOSPHERE_SAMPLES: Defualt 16
    HENYEYGREENSTEIN_SCATTERING: nan
examples:
    - /shaders/lighting_atmosphere.frag
*/

#ifndef ATMOSPHERE_RADIUS_MIN
#define ATMOSPHERE_RADIUS_MIN 6360e3
#endif

#ifndef ATMOSPHERE_RADIUS_MAX
#define ATMOSPHERE_RADIUS_MAX 6420e3
#endif

#ifndef ATMOSPHERE_SUN_POWER
#define ATMOSPHERE_SUN_POWER 20.0
#endif

#ifndef ATMOSPHERE_LIGHT_SAMPLES
#define ATMOSPHERE_LIGHT_SAMPLES 8
#endif

#ifndef ATMOSPHERE_SAMPLES
#define ATMOSPHERE_SAMPLES 16
#endif

// scale height (m)
// thickness of the atmosphere if its density were uniform
#ifndef ATMOSPHERE_RAYLEIGH_THICKNESS
#define ATMOSPHERE_RAYLEIGH_THICKNESS 7994.0
#endif 

#ifndef ATMOSPHERE_MIE_THICKNESS
#define ATMOSPHERE_MIE_THICKNESS 1200.0
#endif 

// scattering coefficients at sea level (m)
#ifndef ATMOSPHERE_RAYLEIGH_SCATTERING
#define ATMOSPHERE_RAYLEIGH_SCATTERING vec3(5.5e-6, 13.0e-6, 22.4e-6)
#endif 

#ifndef ATMOSPHERE_MIE_SCATTERING
#define ATMOSPHERE_MIE_SCATTERING vec3(21e-6)
#endif 

#ifndef FNC_ATMOSPHERE
#define FNC_ATMOSPHERE

bool atmosphere_intersect( const in Ray ray, inout float t0, inout float t1) {
    vec3 rc = vec3(0.0, 0.0, 0.0) - ray.origin;
    float radius2 = ATMOSPHERE_RADIUS_MAX * ATMOSPHERE_RADIUS_MAX;
    float tca = dot(rc, ray.direction);
    float d2 = dot(rc, rc) - tca * tca;
    if (d2 > radius2) 
        return false;

    float thc = sqrt(radius2 - d2);
    t0 = tca - thc;
    t1 = tca + thc;
    return true;
}

bool atmosphere_light(const in Ray ray, inout float optical_depthR, inout float optical_depthM) {
    float t0 = 0.0;
    float t1 = 0.0;
    atmosphere_intersect(ray, t0, t1);

    // this is the implementation using classical raymarching 
    float march_pos = 0.;
    float march_step = t1 / float(ATMOSPHERE_LIGHT_SAMPLES);
    
    for (int i = 0; i < ATMOSPHERE_LIGHT_SAMPLES; i++) {
        vec3 s =    ray.origin +
                    ray.direction * (march_pos + 0.5 * march_step);
        float height = length(s) - ATMOSPHERE_RADIUS_MIN;
        if (height < 0.)
            return false;
    
        optical_depthR += exp(-height / ATMOSPHERE_RAYLEIGH_THICKNESS) * march_step;
        optical_depthM += exp(-height / ATMOSPHERE_MIE_THICKNESS) * march_step;
    
        march_pos += march_step;
    }

    return true;
}

vec3 atmosphere(const in Ray ray, vec3 sun_dir) {
    // "pierce" the atmosphere with the viewing ray
    float t0 = 0.0;
    float t1 = 0.0;
    // atmosphere_intersect(ray, t0, t1);
    if (!atmosphere_intersect(ray, t0, t1))
        return vec3(0.0);

    float march_step = t1 / float(ATMOSPHERE_SAMPLES);

    // cosine of angle between view and light directions
    float mu = dot(ray.direction, sun_dir);

    // Rayleigh and Mie phase functions
    // A black box indicating how light is interacting with the material
    // Similar to BRDF except
    // * it usually considers a single angle
    //   (the phase angle between 2 directions)
    // * integrates to 1 over the entire sphere of directions
    float phaseR = rayleigh(mu);
    float phaseM = henyeyGreenstein(mu);

    // optical depth (or "average density")
    // represents the accumulated extinction coefficients
    // along the path, multiplied by the length of that path
    float optical_depthR = 0.;
    float optical_depthM = 0.;

    vec3 sumR = vec3(0.0, 0.0, 0.0);
    vec3 sumM = vec3(0.0, 0.0, 0.0);
    float march_pos = 0.0;

    for (int i = 0; i < ATMOSPHERE_SAMPLES; i++) {
        vec3 s =    ray.origin +
                    ray.direction * (march_pos + 0.5 * march_step);
        float height = length(s) - ATMOSPHERE_RADIUS_MIN;

        // integrate the height scale
        float hr = exp(-height / ATMOSPHERE_RAYLEIGH_THICKNESS) * march_step;
        float hm = exp(-height / ATMOSPHERE_MIE_THICKNESS) * march_step;
        optical_depthR += hr;
        optical_depthM += hm;

        // gather the sunlight
        Ray ray = Ray(s, sun_dir);

        float optical_depth_lightR = 0.;
        float optical_depth_lightM = 0.;

        if ( atmosphere_light( ray, optical_depth_lightR, optical_depth_lightM) ) {
            // If it's over the horizon
            vec3 tau =  ATMOSPHERE_RAYLEIGH_SCATTERING * (optical_depthR + optical_depth_lightR) +
                        ATMOSPHERE_MIE_SCATTERING * 1.1 * (optical_depthM + optical_depth_lightM);
            vec3 attenuation = exp(-tau);
            sumR += hr * attenuation;
            sumM += hm * attenuation;
        }

        march_pos += march_step;
    }

    return  ATMOSPHERE_SUN_POWER * (sumR * phaseR * ATMOSPHERE_RAYLEIGH_SCATTERING +
                                    sumM * phaseM * ATMOSPHERE_MIE_SCATTERING);
}

vec3 atmosphere(vec3 eye_dir, vec3 sun_dir) {
    Ray ray = Ray(vec3(0., ATMOSPHERE_RADIUS_MIN + 1., 0.), eye_dir);
    return atmosphere(ray, sun_dir);
}

#endif