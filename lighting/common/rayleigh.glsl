#include "../../math/const.glsl"

#ifndef FNC_RAYLEIGH
#define FNC_RAYLEIGH

// Rayleigh phase
float rayleigh(float mu) {
    return 3. * (1. + mu*mu) / (16. * PI);
}

#endif