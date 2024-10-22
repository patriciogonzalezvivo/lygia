/*
contributors: Patricio Gonzalez Vivo
description: |
    Draws all the digits of a floating point number, useful for debugging.
    Requires high precision to work properly.
    Note, think on improving the performance of this function using Freya Holmer approach https://gist.github.com/FreyaHolmer/71717be9f3030c1b0990d3ed1ae833e3
use: digits(<float2> st, <float> value [, <float> nDecDigit])
options:
    DIGITS_DECIMALS: number of decimals after the point, defaults to 2
    DIGITS_SIZE: size of the font, defaults to float2(.02, 0.02)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef DIGITS_SIZE
#define DIGITS_SIZE float2(.02, .02)
#endif

#ifndef DIGITS_VALUE_OFFSET
#define DIGITS_VALUE_OFFSET float2(-6.0, 3.0)
#endif

#ifndef DIGITS_DECIMALS
#define DIGITS_DECIMALS 2.0
#endif

#ifndef FNC_DIGITS
#define FNC_DIGITS
float digits(in float2 st, in float value, in float nDecDigit) {
    st /= DIGITS_SIZE;

    float absValue = abs(value);
    float biggestDigitIndex = max(floor(log2(absValue) / log2(10.)), 0.);

    float counter = floor(absValue);
    float nIntDigits = 0.;
    for (int i = 0; i < 9; i++) {
        counter = floor(counter*.1);
        nIntDigits++;
        if (counter == 0.)
            break;
    }
    float digit = 12.;
    float digitIndex = (nIntDigits-1.) - floor(st.x);
    if (digitIndex > (-nDecDigit - 1.5)) {
        if (digitIndex > biggestDigitIndex) {
            if (value < 0.) {
                if (digitIndex < (biggestDigitIndex+1.5)) {
                    digit = 11.0;
                }
            }
        } 
        else {
            if (digitIndex == -1.) {
                if (nDecDigit > 0.) {
                    digit = 10.;
                }
            } 
            else {
                if (digitIndex < 0.) {
                    digitIndex += 1.;
                }
                float digitValue = (absValue / (pow(10., digitIndex)));
                digit = floor(0.0001+digitValue) % 10.;
            }
        }
    }
    float2 pos = float2(frac(st.x), st.y);

    if (pos.x < 0.) return 0.;
    if (pos.y < 0.) return 0.;
    if (pos.x >= 1.) return 0.;
    if (pos.y >= 1.) return 0.;

    // make a 4x5 array of bits
    float bin = 0.;
    if (digit < 0.5) // 0
        bin = 7. + 5. * 16. + 5. * 256. + 5. * 4096. + 7. * 65536.; 
    else if (digit < 1.5) // 1
        bin = 2. + 2. * 16. + 2. * 256. + 2. * 4096. + 2. * 65536.;
    else if (digit < 2.5) // 2
        bin = 7. + 1. * 16. + 7. * 256. + 4. * 4096. + 7. * 65536.;
    else if (digit < 3.5) // 3
        bin = 7. + 4. * 16. + 7. * 256. + 4. * 4096. + 7. * 65536.;
    else if (digit < 4.5) // 4
        bin = 4. + 7. * 16. + 5. * 256. + 1. * 4096. + 1. * 65536.;
    else if (digit < 5.5) // 5
        bin = 7. + 4. * 16. + 7. * 256. + 1. * 4096. + 7. * 65536.;
    else if (digit < 6.5) // 6
        bin = 7. + 5. * 16. + 7. * 256. + 1. * 4096. + 7. * 65536.;
    else if (digit < 7.5) // 7
        bin = 4. + 4. * 16. + 4. * 256. + 4. * 4096. + 7. * 65536.;
    else if (digit < 8.5) // 8
        bin = 7. + 5. * 16. + 7. * 256. + 5. * 4096. + 7. * 65536.;
    else if (digit < 9.5) // 9
        bin = 7. + 4. * 16. + 7. * 256. + 5. * 4096. + 7. * 65536.;
    else if (digit < 10.5) // '.'
        bin = 2. + 0. * 16. + 0. * 256. + 0. * 4096. + 0. * 65536.;
    else if (digit < 11.5) // '-'
        bin = 0. + 0. * 16. + 7. * 256. + 0. * 4096. + 0. * 65536.;

    float2 pixel = floor(pos * float2(4., 5.));
    return floor(bin / pow(2., (pixel.x + (pixel.y * 4.)))) % 2.0;
}

float digits(in float2 st, in float value, in float nDecDigit, in float nIntDigits) {
    float2 st2 = st;
    float result = 0.0;
    float dig = nDecDigit;

    #ifndef DIGITS_LEADING_INT
    #define DIGITS_LEADING_INT nIntDigits
    #endif

    for (float i = DIGITS_LEADING_INT - 1.0; i > 0.0 ; i--) {
        if (i * 10.0 > value) {
            result += digits(st2, 0.0, 0.0);
            st2.x -= DIGITS_SIZE.x;
        }
    }
    result += digits(st2, value, nDecDigit);
    return result; 
}

float digits(in float2 st, in int value) {
    return digits(st, float(value), 0.0);
}

float digits(in float2 st, in float value) {
    return digits(st, value, (DIGITS_DECIMALS));
}

float digits(in float2 st, in float2 v) {
    float rta = 0.0;
    for (int i = 0; i < 2; i++) {
        float2 pos = st + float2(float(i), 0.0) * DIGITS_SIZE * DIGITS_VALUE_OFFSET;
        float value = i == 0 ? v.x : v.y;
        rta += digits( pos, value );
    }
    return rta;
}

float digits(in float2 st, in float3 v) {
    float rta = 0.0;
    for (int i = 0; i < 3; i++) {
        float2 pos = st + float2(float(i), 0.0) * DIGITS_SIZE * DIGITS_VALUE_OFFSET;
        float value = i == 0 ? v.x : i == 1 ? v.y : v.z;
        rta += digits( pos, value );
    }
    return rta;
}

float digits(in float2 st, in float4 v) {
    float rta = 0.0;
    for (int i = 0; i < 4; i++) {
        float2 pos = st + float2(float(i), 0.0) * DIGITS_SIZE * DIGITS_VALUE_OFFSET;
        float value = i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
        rta += digits( pos, value );
    }
    return rta;
}

float digits(in float2 st, in float2x2 _matrix) {
    float rta = 0.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float2 pos = st + float2(float(i), float(j)) * DIGITS_SIZE * DIGITS_VALUE_OFFSET - DIGITS_SIZE * float2(0.0, 3.0);
            float value = _matrix[j][i];
            rta += digits( pos, value );
        }
    }
    return rta;
}

float digits(in float2 st, in float3x3 _matrix) {
    float rta = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float2 pos = st + float2(float(i), float(j)) * DIGITS_SIZE * DIGITS_VALUE_OFFSET - DIGITS_SIZE * float2(0.0, 6.0);
            float value = _matrix[j][i];
            rta += digits( pos, value );
        }
    }
    return rta;
}

float digits(in float2 st, in float4x4 _matrix) {
    float rta = 0.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float2 pos = st + float2(float(i), float(j)) * DIGITS_SIZE * DIGITS_VALUE_OFFSET - DIGITS_SIZE * float2(0.0, 9.0);
            float value = _matrix[j][i];
            rta += digits( pos, value );
        }
    }
    return rta;
}
#endif
