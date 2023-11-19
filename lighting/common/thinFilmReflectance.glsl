#include "../../math/const.glsl"

#ifndef FNC_THINFILMREFLECTANCE
#define FNC_THINFILMREFLECTANCE

//    https://en.wikipedia.org/wiki/Polarization_(waves)#s_and_p_designations
//    https://en.wikipedia.org/wiki/Fresnel_equations

//    Light waves are polarized. This can be described in the coordinate system of the 
//    plane of incidence which holds the surface normal and the view ray. s-polarized is 
//    perpendicular and p-polarized is parallel to the plane. The following functions give
//    the Fresnel equations for reflection and transmission of p- and s-polarized light.

//    cosI: angle of normal and incident ray
//    cosR: angle of normal and reflected/refracted ray

// Reflection coefficient (s-polarized)
float rs(float n1, float n2, float cosI, float cosR) { return (n1 * cosI - n2 * cosR) / (n1 * cosI + n2 * cosR); }
 
// Reflection coefficient (p-polarized)
float rp(float n1, float n2, float cosI, float cosR) { return (n2 * cosI - n1 * cosR) / (n1 * cosR + n2 * cosI); }
 
// Transmission coefficient (s-polarized)
float ts(float n1, float n2, float cosI, float cosR) { return 2.0 * n1 * cosI / (n1 * cosI + n2 * cosR); }
 
// Transmission coefficient (p-polarized)
float tp(float n1, float n2, float cosI, float cosR) { return 2.0 * n1 * cosI / (n1 * cosR + n2 * cosI); }


//   We model a thin layer of material between the outer medium (air) and an internal material 
//   (can be air for bubbles).
//   External medium index is ior0, thin layer index is 1 and the internal index is 2.

//   We are interested in how much light at a given weavelength is reflected back into the outer
//   medium (R). We assume nothing is absorbed.
//   It's easier to find how much is transmitted into the internal medium (T) and use the 
//   equation R = 1 - T to find the reflected quantity.


// cos0 is the cosine of the incident angle, that is, NoV = dot(view angle, normal) 
// https://www.gamedev.net/tutorials/programming/graphics/thin-film-interference-for-computer-graphics-r2962/
vec3 thinFilmReflectance(float cos0, float thickness, float ior0, float ior1, float ior2) { 
    // lambda is the wavelength of the incident light (e.g. lambda = 510 for green) 
    const vec3 lambda = vec3(650.0, 510.0, 475.0);

    // Precompute the reflection phase changes (depends on IOR)
    float delta10 = (ior1 < ior2) ? PI : 0.0; 
    float delta12 = (ior1 < ior0) ? PI : 0.0; 
    float delta = delta10 + delta12; 
    
    // Calculate the thin film layer (and transmitted) angle cosines.
    float sin1 = pow(ior2 / ior1, 2.0) * (1.0 - pow(cos0, 2.0)); 
    if ((sin1 > 1.0)) 
        return vec3(1.0); 
    float sin2 = pow(ior2 / ior0, 2.0) * (1.0 - pow(cos0, 2.0)); 
    if ((sin2 > 1.0)) 
        return vec3(1.0); 

    // Account for TIR.
    float cos1 = sqrt(1.0 - sin1);
    float cos2 = sqrt(1.0 - sin2); 

    // Calculate the interference phase change.
    vec3 phi = vec3(2.0 * ior1 * thickness * cos1); 
    phi *= TAU / lambda; 
    phi += delta; 

    // Obtain the various Fresnel amplitude coefficients. 
    float alpha_s = rs(ior1, ior2, cos1, cos0) * rs(ior1, ior0, cos1, cos2); 
    float alpha_p = rp(ior1, ior2, cos1, cos0) * rp(ior1, ior0, cos1, cos2); 
    float beta_s = ts(ior2, ior1, cos0, cos1) * ts(ior1, ior0, cos1, cos2); 
    float beta_p = tp(ior2, ior1, cos0, cos1) * tp(ior1, ior0, cos1, cos2); 

    // Calculate the s- and p-polarized intensity transmission coefficient. 
    vec3 ts = pow(beta_s, 2.0) / (pow(alpha_s, 2.0) - 2.0 * alpha_s * cos(phi) + 1.0); 
    vec3 tp = pow(beta_p, 2.0) / (pow(alpha_p, 2.0) - 2.0 * alpha_p * cos(phi) + 1.0);

    // Calculate the transmitted power ratio for medium change. 
    float beamRatio = (ior0 * cos2) / (ior2 * cos0); 

    // Calculate the average reflectance. 
    return 1.0 - beamRatio * (ts + tp) * 0.5; 
} 
#endif