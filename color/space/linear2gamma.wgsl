fn linear2gamma(v : vec3f) -> vec3f {
    return pow(v, vec3f(1. / 2.2));
}
