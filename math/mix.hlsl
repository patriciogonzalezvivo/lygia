/*
original_author: Patricio Gonzalez Vivo
description: expands mix to linearly mix more than two values
use: mix(<float|float2|float3|float4> a, <float|float2|float3|float4> b, <float|float2|float3|float4> c [, <float|float2|float3|float4> d], <float> pct)
*/

#ifndef FNC_MIX
#define FNC_MIX

float   mix(in float a, in float b, in float c) { return lerp(a, b, c); }
float2  mix(in float2 a, in float2 b, in float c) { return lerp(a, b, c); }
float2  mix(in float2 a, in float2 b, in float2 c) { return lerp(a, b, c); }
float3  mix(in float3 a, in float3 b, in float c) { return lerp(a, b, c); }
float3  mix(in float3 a, in float3 b, in float3 c) { return lerp(a, b, c); }
float4  mix(in float4 a, in float4 b, in float c) { return lerp(a, b, c); }
float4  mix(in float4 a, in float4 b, in float4 c) { return lerp(a, b, c); }

float mix(float a , float b, float c, float pct) {
    return lerp(
        lerp(a, b, 2. * pct),
        lerp(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

float2 mix(float2 a , float2 b, float2 c, float pct) {
    return lerp(
        lerp(a, b, 2. * pct),
        lerp(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

float2 mix(float2 a , float2 b, float2 c, float2 pct) {
    return lerp(
        lerp(a, b, 2. * pct),
        lerp(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

float3 mix(float3 a , float3 b, float3 c, float pct) {
    return lerp(
        lerp(a, b, 2. * pct),
        lerp(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

float3 mix(float3 a , float3 b, float3 c, float3 pct) {
    return lerp(
        lerp(a, b, 2. * pct),
        lerp(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

float4 mix(float4 a , float4 b, float4 c, float pct) {
    return lerp(
        lerp(a, b, 2. * pct),
        lerp(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

float4 mix(float4 a , float4 b, float4 c, float4 pct) {
    return lerp(
        lerp(a, b, 2. * pct),
        lerp(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

float mix(in float a , in float b, in float c, in float d, in float pct) {
    return lerp(
        lerp(a, b, 3. * pct),
        lerp(b,
            lerp(c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

float2 mix(in float2 a , in float2 b, in float2 c, in float2 d, in float pct) {
    return lerp(
        lerp(a, b, 3. * pct),
        lerp(b,
            lerp(c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

float2 mix(in float2 a , in float2 b, in float2 c, in float2 d, in float2 pct) {
    return lerp(
        lerp(a, b, 3. * pct),
        lerp(b,
            lerp(c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

float3 mix(in float3 a , in float3 b, in float3 c, in float3 d, in float pct) {
    return lerp(
        lerp(a, b, 3. * pct),
        lerp(b,
            lerp(c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

float3 mix(in float3 a , in float3 b, in float3 c, in float3 d, in float3 pct) {
    return lerp(
        lerp(a, b, 3. * pct),
        lerp(b,
            lerp(c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

float4 mix(in float4 a , in float4 b, in float4 c, in float4 d, in float pct) {
    return lerp(
        lerp(a, b, 3. * pct),
        lerp(b,
            lerp(c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

float4 mix(in float4 a , in float4 b, in float4 c, in float4 d, in float4 pct) {
    return lerp(
        lerp(a, b, 3. * pct),
        lerp(b,
            lerp(c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}
#endif
