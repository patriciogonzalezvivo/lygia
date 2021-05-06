#ifndef FNC_FIRE
#define FNC_FIRE

vec3 fire(float x) {
    return vec3(1.0, 0.25, 0.0625) * exp(4.0 * x - 1.0);
}

#endif