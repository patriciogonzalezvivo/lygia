/*
contributors: Patricio Gonzalez Vivo
description: |
    Draws all the digits of a floating point number, useful for debugging.
    Requires high precision to work properly.
use: digits(<vec2> st, <float> value [, <float> nDecDigit])
options:
    DIGITS_DECIMALS: number of decimals after the point, defaults to 2
    DIGITS_SIZE: size of the font, defaults to vec2(.025)
examples:
    - /shaders/draw_digits.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define DIGITS_SIZE vec2(.02)

const DIGITS_DECIMALS: f32 = 2.0;

// #define DIGITS_VALUE_OFFSET vec2(-6.0, 3.0)

fn digits2(st: vec2f, value: f32, nDecDigit: f32) -> f32 {
    st /= DIGITS_SIZE;

    let absValue = abs(value);
    let biggestDigitIndex = max(floor(log2(absValue) / log2(10.)), 0.);
    let counter = floor(absValue);
    let nIntDigits = 1.;
    for (int i = 0; i < 9; i++) {
        counter = floor(counter*.1);
        nIntDigits++;
        if (counter == 0.)
            break;
    }

    let digit = 12.;
    let digitIndex = (nIntDigits-1.) - floor(st.x);
    if (digitIndex > (-nDecDigit - 1.5)) {
        if (digitIndex > biggestDigitIndex) {
            if (value < 0.) {
                if (digitIndex < (biggestDigitIndex+1.5)) {
                    digit = 11.;
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
                let digitValue = (absValue / (pow(10., digitIndex)));
                digit = mod(floor(0.0001+digitValue), 10.);
            }
        }
    }
    let pos = vec2f(fract(st.x), st.y);

    if (pos.x < 0.) return 0.;
    if (pos.y < 0.) return 0.;
    if (pos.x >= 1.) return 0.;
    if (pos.y >= 1.) return 0.;

    // make a 4x5 array of bits
    let bin = 0.;
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

    let pixel = floor(pos * vec2f(4., 5.));
    return mod(floor(bin / pow(2., (pixel.x + (pixel.y * 4.)))), 2.);
}

fn digits2a(st: vec2f, value: f32, nDecDigit: f32, nIntDigits: f32) -> f32 {
    let st2 = st;
    let result = 0.0;
    let dig = nDecDigit;

const DIGITS_LEADING_INT: f32 = 1.0;
//     #define DIGITS_LEADING_INT nIntDigits

    for (float i = DIGITS_LEADING_INT - 1.0; i > 0.0 ; i--) {
        if (i * 10.0 > value) {
            result += digits(st2, 0.0, 0.0);
            st2.x -= DIGITS_SIZE.x;
        }
    }
    result += digits(st2, value, nDecDigit);
    return result; 
}

fn digits2b(st: vec2f, value: i32) -> f32 {
    return digits(st, float(value), 0.0);
}

fn digits2c(st: vec2f, value: f32) -> f32 {
    return digits(st, value, (DIGITS_DECIMALS));
}

fn digits2d(st: vec2f, v: vec2f) -> f32 {
    let rta = 0.0;
    for (int i = 0; i < 2; i++) {
        let pos = st + vec2f(float(i), 0.0) * DIGITS_SIZE * DIGITS_VALUE_OFFSET;
        let value = i == 0 ? v.x : v.y;
        rta += digits( pos, value );
    }
    return rta;
}

fn digits2e(st: vec2f, v: vec3f) -> f32 {
    let rta = 0.0;
    for (int i = 0; i < 3; i++) {
        let pos = st + vec2f(float(i), 0.0) * DIGITS_SIZE * DIGITS_VALUE_OFFSET;
        let value = i == 0 ? v.x : i == 1 ? v.y : v.z;
        rta += digits( pos, value );
    }
    return rta;
}

fn digits2f(st: vec2f, v: vec4f) -> f32 {
    let rta = 0.0;
    for (int i = 0; i < 4; i++) {
        let pos = st + vec2f(float(i), 0.0) * DIGITS_SIZE * DIGITS_VALUE_OFFSET;
        let value = i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
        rta += digits( pos, value );
    }
    return rta;
}

fn digits2g(st: vec2f, _matrix: mat2x2<f32>) -> f32 {
    let rta = 0.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            let pos = st + vec2f(float(i), float(j)) * DIGITS_SIZE * DIGITS_VALUE_OFFSET - DIGITS_SIZE * vec2f(0.0, 3.0);
            let value = _matrix[j][i];
            rta += digits( pos, value );
        }
    }
    return rta;
}

fn digits2h(st: vec2f, _matrix: mat3x3<f32>) -> f32 {
    let rta = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            let pos = st + vec2f(float(i), float(j)) * DIGITS_SIZE * DIGITS_VALUE_OFFSET - DIGITS_SIZE * vec2f(0.0, 6.0);
            let value = _matrix[j][i];
            rta += digits( pos, value );
        }
    }
    return rta;
}

fn digits2i(st: vec2f, _matrix: mat4x4<f32>) -> f32 {
    let rta = 0.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            let pos = st + vec2f(float(i), float(j)) * DIGITS_SIZE * DIGITS_VALUE_OFFSET - DIGITS_SIZE * vec2f(0.0, 9.0);
            let value = _matrix[j][i];
            rta += digits( pos, value );
        }
    }
    return rta;
}
