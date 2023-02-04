#ifndef UNPACK_FNC
#define UNPACK_FNC unpack256
#endif 

#ifndef FNC_UNPACK
#define FNC_UNPACK

float unpack8(float3 value) {
    float3 factor = float3( 8.0, 8.0 * 8.0, 8.0 * 8.0 * 8.0 );
    return dot(value, factor) / 512.0;
}

float unpack16(float3 value) {
    float3 factor = float3( 16.0, 16.0 * 16.0, 16.0 * 16.0 * 16.0 );
    return dot(value, factor) / 4096.0;
}

float unpack32(float3 value) {
    float3 factor = float3( 32.0, 32.0 * 32.0, 32.0 * 32.0 * 32.0 );
    return dot(value, factor) / 32768.0;
}

float unpack64(float3 value) {
    float3 factor = float3( 64.0, 64.0 * 64.0, 64.0 * 64.0 * 64.0 );
    return dot(value, factor) / 262144.0;
}

float unpack128(float3 value) {
    float3 factor = float3( 128.0, 128.0 * 128.0, 128.0 * 128.0 * 128.0 );
    return dot(value, factor) / 2097152.0;
}

float unpack256(float3 value) {
    float3 factor = float3( 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
    return dot(value, factor) / 16581375.0;
}

float unpack(float3 value, float base) {
    float base3 = base * base * base;
    float3 factor = float3( base, base * base, base3);
    return dot(value, factor) / base3;
}

float unpack(float3 value) {
    return UNPACK_FNC(value);
}
#endif