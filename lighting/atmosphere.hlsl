#include "lygia/math/const.hlsl"

/*
description: Rayleigh and Mie scattering atmosphere system. Implementation of the techniques described here: https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/simulating-sky/simulating-colors-of-the-sky
use: <float3> atmosphere(<float3> eye_dir, <float3> sun_dir)
OPTIONS:
    ATMOSPHERE_FAST: use fast implementation from https://www.shadertoy.com/view/3dBSDW
    ATMOSPHERE_RADIUS_MIN: planet radius
    ATMOSPHERE_RADIUS_MAX: atmosphere radious
    ATMOSPHERE_SUN_POWER: sun power. Default 20.0
    ATMOSPHERE_LIGHT_SAMPLES: Defualt 8 
    ATMOSPHERE_SAMPLES: Defualt 16
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

#ifndef FNC_ATMOSPHERE
#define FNC_ATMOSHPERE

struct ray_t {
    float3 origin;
    float3 direction;
};

struct sphere_t {
    float3 origin;
    float radius;
    int material;
};

bool isect_sphere( const in ray_t ray, const in sphere_t sphere, inout float t0, inout float t1) {
    float3 rc = sphere.origin - ray.origin;
    float radius2 = sphere.radius * sphere.radius;
    float tca = dot(rc, ray.direction);
    float d2 = dot(rc, rc) - tca * tca;
    if (d2 > radius2) return false;
    float thc = sqrt(radius2 - d2);
    t0 = tca - thc;
    t1 = tca + thc;
    return true;
}

// scattering coefficients at sea level (m)
const float3 betaR = float3(5.5e-6, 13.0e-6, 22.4e-6); // Rayleigh 
const float3 betaM = float3(21e-6, 21e-6, 21e-6); // Mie

// scale height (m)
// thickness of the atmosphere if its density were uniform
const float hR = 7994.0; // Rayleigh
const float hM = 1200.0; // Mie

// Rayleigh phase
float rayleigh_phase(float mu) {
    return 3. * (1. + mu*mu) / (16. * PI);
}

// Henyey-Greenstein phase function factor [-1, 1]
// represents the average cosine of the scattered directions
// 0 is isotropic scattering
// > 1 is forward scattering, < 1 is backwards
const float henyey_greenstein_g = 0.76;
float henyey_greenstein_phase(float mu) {
    return (1.0 - henyey_greenstein_g*henyey_greenstein_g) / ((4. + PI) * pow(1.0 + henyey_greenstein_g*henyey_greenstein_g - 2.0 * henyey_greenstein_g * mu, 1.5));
}

sphere_t atmos;

// #ifdef ATMOSPHERE_FAST
// const float BIG = 1e20;
// const float SMALL = 1e-20;

// float approx_air_column_density_ratio_through_atmosphere(
//     in float a,
//     in float b,
//     in float z2,
//     in float r0
// ){
//     // GUIDE TO VARIABLE NAMES:
//     //  "x*" distance along the ray from closest approach
//     //  "z*" distance from the center of the world at closest approach
//     //  "r*" distance ("radius") from the center of the world
//     //  "*0" variable at reference point
//     //  "*2" the square of a variable
//     //  "ch" a nudge we give to prevent division by zero, analogous to the Chapman function
//     float SQRT_HALF_PI = sqrt(HALF_PI);
//     const float k = 0.6; // "k" is an empirically derived constant
//     float x0 = sqrt(max(r0*r0 - z2, SMALL));
//     // if obstructed by the world, approximate answer by using a ludicrously large number
//     if (a < x0 && -x0 < b && z2 < r0*r0) { return BIG; }
//     float abs_a  = abs(a);
//     float abs_b  = abs(b);
//     float z      = sqrt(z2);
//     float sqrt_z = sqrt(z);
//     float ra     = sqrt(a*a+z2);
//     float rb     = sqrt(b*b+z2);
//     float ch0    = (1. - 1./(2.*r0)) * SQRT_HALF_PI * sqrt_z + k*x0;
//     float cha    = (1. - 1./(2.*ra)) * SQRT_HALF_PI * sqrt_z + k*abs_a;
//     float chb    = (1. - 1./(2.*rb)) * SQRT_HALF_PI * sqrt_z + k*abs_b;
//     float s0     = min(exp(r0- z),1.) / (x0/r0 + 1./ch0);
//     float sa     = exp(r0-ra) / max(abs_a/ra + 1./cha, 0.01);
//     float sb     = exp(r0-rb) / max(abs_b/rb + 1./chb, 0.01);
//     return max( sign(b)*(s0-sb) - sign(a)*(s0-sa), 0.0 );
// }

// // "approx_air_column_density_ratio_along_3d_ray_for_curved_world" is just a convenience wrapper 
// //   for the above function that works with 3d vectors.
// float approx_air_column_density_ratio_along_3d_ray_for_curved_world (
//     float3  P, // position of viewer
//     float3  V, // direction of viewer (unit vector)
//     float x, // distance from the viewer at which we stop the "raymarch"
//     float r, // radius of the planet
//     float H  // scale height of the planet's atmosphere
// ){
//     float xz = dot(-P,V);           // distance ("radius") from the ray to the center of the world at closest approach, squared
//     float z2 = dot( P,P) - xz * xz; // distance from the origin at which closest approach occurs
//     return approx_air_column_density_ratio_through_atmosphere( 0.-xz, x-xz, z2, r/H );
// }
// #endif


bool get_sun_light(const in ray_t ray, inout float optical_depthR, inout float optical_depthM) {
    float t0 = 0.0;
    float t1 = 0.0;
    isect_sphere(ray, atmos, t0, t1);

    // #ifdef ATMOSPHERE_FAST
    //     // this is the implementation using a fast closed form approximation
    // optical_depthR = 
    //     approx_air_column_density_ratio_along_3d_ray_for_curved_world (
    //         ray.origin,    // position of viewer
    //         ray.direction, // direction of viewer (unit vector)
    //         t1, // distance from the viewer at which we stop the "raymarch"
    //         ATMOSPHERE_RADIUS_MIN, // radius of the planet
    //         hR  // scale height of the planet's atmosphere
    //     );
    // optical_depthM = 
    //     approx_air_column_density_ratio_along_3d_ray_for_curved_world (
    //         ray.origin,    // position of viewer
    //         ray.direction, // direction of viewer (unit vector)
    //         t1, // distance from the viewer at which we stop the "raymarch"
    //         ATMOSPHERE_RADIUS_MIN, // radius of the planet
    //         hM  // scale height of the planet's atmosphere
    //     );

    // #else

    // this is the implementation using classical raymarching 
    float march_pos = 0.;
    float march_step = t1 / float(ATMOSPHERE_LIGHT_SAMPLES);
    
    for (int i = 0; i < ATMOSPHERE_LIGHT_SAMPLES; i++) {
        float3 s =    ray.origin +
                    ray.direction * (march_pos + 0.5 * march_step);
        float height = length(s) - ATMOSPHERE_RADIUS_MIN;
        if (height < 0.)
            return false;
    
        optical_depthR += exp(-height / hR) * march_step;
        optical_depthM += exp(-height / hM) * march_step;
    
        march_pos += march_step;
    }
    // #endif

    return true;
}

float3 get_incident_light(const in ray_t ray, float3 sun_dir) {
    // "pierce" the atmosphere with the viewing ray
    float t0 = 0.0;
    float t1 = 0.0;
    if (!isect_sphere(ray, atmos, t0, t1))
        return float3(0.0, 0.0, 0.0);

    float march_step = t1 / float(ATMOSPHERE_SAMPLES);

    // cosine of angle between view and light directions
    float mu = dot(ray.direction, sun_dir);

    // Rayleigh and Mie phase functions
    // A black box indicating how light is interacting with the material
    // Similar to BRDF except
    // * it usually considers a single angle
    //   (the phase angle between 2 directions)
    // * integrates to 1 over the entire sphere of directions
    float phaseR = rayleigh_phase(mu);
    float phaseM = henyey_greenstein_phase(mu);

    // optical depth (or "average density")
    // represents the accumulated extinction coefficients
    // along the path, multiplied by the length of that path
    float optical_depthR = 0.;
    float optical_depthM = 0.;

    float3 sumR = float3(0.0, 0.0, 0.0);
    float3 sumM = float3(0.0, 0.0, 0.0);
    float march_pos = 0.;

    for (int i = 0; i < ATMOSPHERE_SAMPLES; i++) {
        float3 s =  ray.origin +
                    ray.direction * (march_pos + 0.5 * march_step);
        float height = length(s) - ATMOSPHERE_RADIUS_MIN;

        // integrate the height scale
        float hr = exp(-height / hR) * march_step;
        float hm = exp(-height / hM) * march_step;
        optical_depthR += hr;
        optical_depthM += hm;

        // gather the sunlight
        ray_t light_ray;// = ray_t(s, sun_dir);
        light_ray.origin = s;
        light_ray.direction = sun_dir;

        float optical_depth_lightR = 0.;
        float optical_depth_lightM = 0.;
        bool overground = get_sun_light(
            light_ray,
            optical_depth_lightR,
            optical_depth_lightM);

        if (overground) {
            float3 tau =  betaR * (optical_depthR + optical_depth_lightR) +
                        betaM * 1.1 * (optical_depthM + optical_depth_lightM);
            float3 attenuation = exp(-tau);
            sumR += hr * attenuation;
            sumM += hm * attenuation;
        }

        march_pos += march_step;
    }

    return  ATMOSPHERE_SUN_POWER * (sumR * phaseR * betaR +
                                    sumM * phaseM * betaM);
}

float3 atmosphere(float3 eye_dir, float3 sun_dir) {
    ray_t ray;
    ray.origin = float3(0., ATMOSPHERE_RADIUS_MIN + 1., 0.);
    ray.direction = eye_dir;

    atmos.origin = float3(0.0, 0.0, 0.0);
    atmos.radius = ATMOSPHERE_RADIUS_MAX;
    atmos.material = 0.0;
    return get_incident_light(ray, sun_dir);
}

#endif