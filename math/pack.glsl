#ifndef UNPACK_FNC
#define UNPACK_FNC unpack256
#endif 

#ifndef FNC_UNPACK
#define FNC_UNPACK

float unpack8(vec3 value) {
    vec3 factor = vec3( 8.0, 8.0 * 8.0, 8.0 * 8.0 * 8.0 );
    return dot(value, factor) / 512.0;
}

float unpack16(vec3 value) {
    vec3 factor = vec3( 16.0, 16.0 * 16.0, 16.0 * 16.0 * 16.0 );
    return dot(value, factor) / 4096.0;
}

float unpack32(vec3 value) {
    vec3 factor = vec3( 32.0, 32.0 * 32.0, 32.0 * 32.0 * 32.0 );
    return dot(value, factor) / 32768.0;
}

float unpack64(vec3 value) {
    vec3 factor = vec3( 64.0, 64.0 * 64.0, 64.0 * 64.0 * 64.0 );
    return dot(value, factor) / 262144.0;
}

float unpack128(vec3 value) {
    vec3 factor = vec3( 128.0, 128.0 * 128.0, 128.0 * 128.0 * 128.0 );
    return dot(value, factor) / 2097152.0;
}

float unpack256(vec3 value) {
    vec3 factor = vec3( 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
    return dot(value, factor) / 16581375.0;
}

float unpack(vec3 value, float base) {
    float base3 = base * base * base;
    vec3 factor = vec3( base, base * base, base3);
    return dot(value, factor) / base3;
}

float unpack(vec3 value) {
    return UNPACK_FNC(value);
}
#endif