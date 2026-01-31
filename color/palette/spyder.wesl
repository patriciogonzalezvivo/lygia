/*
contributors: Patricio Gonzalez Vivo
description: |
    SpyderChecker 48 color palette values from:
    - https://www.northlight-images.co.uk/datacolor-spydercheckr-colour-test-card-review/
    - https://www.bartneck.de/2017/10/24/patch-color-definitions-for-datacolor-spydercheckr-48/
use:
    - <vec3f> spyder(<i32> index)
    - <vec3f> spyderA(<i32> index) - first 24 colors (side A)
    - <vec3f> spyderB(<i32> index) - last 24 colors (side B)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_colorChecker.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const SPYDER_TOTAL: u32 = 48u;

// Side A - Row 1
const LOW_SAT_RED: vec3f = vec3f(0.824, 0.475, 0.459);
const LOW_SAT_YELLOW: vec3f = vec3f(0.847, 0.702, 0.353);
const LOW_SAT_GREEN: vec3f = vec3f(0.498, 0.686, 0.471);
const LOW_SAT_CYAN: vec3f = vec3f(0.259, 0.616, 0.702);
const LOW_SAT_BLUE: vec3f = vec3f(0.455, 0.576, 0.761);
const LOW_SAT_MAGENTA: vec3f = vec3f(0.745, 0.475, 0.604);

// Side A - Row 2
const RED_TINT_10: vec3f = vec3f(0.855, 0.796, 0.788);
const GREEN_TINT_10: vec3f = vec3f(0.796, 0.804, 0.769);
const BLUE_TINT_10: vec3f = vec3f(0.808, 0.796, 0.816);
const RED_TONE_90: vec3f = vec3f(0.259, 0.224, 0.227);
const GREEN_TONE_90: vec3f = vec3f(0.212, 0.239, 0.220);
const BLUE_TONE_90: vec3f = vec3f(0.247, 0.235, 0.271);

// Side A - Row 3
const LIGHTEST_SKIN: vec3f = vec3f(0.929, 0.808, 0.729);
const LIGHTER_SKIN: vec3f = vec3f(0.827, 0.686, 0.522);
const MODERATE_SKIN: vec3f = vec3f(0.757, 0.584, 0.357);
const MEDIUM_SKIN: vec3f = vec3f(0.545, 0.365, 0.239);
const DEEP_SKIN: vec3f = vec3f(0.290, 0.216, 0.180);
const GRAY_95: vec3f = vec3f(0.224, 0.212, 0.220);

// Side A - Row 4
const GRAY_05: vec3f = vec3f(0.945, 0.914, 0.898);
const GRAY_10: vec3f = vec3f(0.898, 0.871, 0.863);
const GRAY_30: vec3f = vec3f(0.714, 0.698, 0.690);
const GRAY_50: vec3f = vec3f(0.545, 0.533, 0.529);
const GRAY_70: vec3f = vec3f(0.392, 0.388, 0.380);
const GRAY_90: vec3f = vec3f(0.247, 0.239, 0.243);

// Side B - Row 1
const AQUA: vec3f = vec3f(0.384, 0.733, 0.651);
const LAVENDER: vec3f = vec3f(0.494, 0.490, 0.682);
const EVERGREEN: vec3f = vec3f(0.322, 0.423, 0.247);
const STEEL_BLUE: vec3f = vec3f(0.341, 0.467, 0.603);
const CLASSIC_LIGHT_SKIN: vec3f = vec3f(0.769, 0.557, 0.494);
const CLASSIC_DARK_SKIN: vec3f = vec3f(0.439, 0.302, 0.247);

// Side B - Row 2
const PRIMARY_ORANGE: vec3f = vec3f(0.871, 0.463, 0.125);
const BLUEPRINT: vec3f = vec3f(0.227, 0.349, 0.627);
const PINK: vec3f = vec3f(0.765, 0.310, 0.373);
const VIOLET: vec3f = vec3f(0.325, 0.227, 0.416);
const APPLE_GREEN: vec3f = vec3f(0.616, 0.737, 0.212);
const SUNFLOWER: vec3f = vec3f(0.933, 0.620, 0.098);

// Side B - Row 3
const PRIMARY_CYAN: vec3f = vec3f(0.000, 0.498, 0.623);
const PRIMARY_MAGENTA: vec3f = vec3f(0.753, 0.294, 0.569);
const PRIMARY_YELLOW: vec3f = vec3f(0.961, 0.804, 0.000);
const PRIMARY_RED: vec3f = vec3f(0.729, 0.102, 0.200);
const PRIMARY_GREEN: vec3f = vec3f(0.224, 0.573, 0.251);
const PRIMARY_BLUE: vec3f = vec3f(0.098, 0.216, 0.529);

// Side B - Row 4
const CARD_WHITE: vec3f = vec3f(0.976, 0.949, 0.933);
const GRAY_20: vec3f = vec3f(0.792, 0.777, 0.765);
const GRAY_40: vec3f = vec3f(0.631, 0.616, 0.604);
const GRAY_60: vec3f = vec3f(0.478, 0.463, 0.455);
const GRAY_80: vec3f = vec3f(0.314, 0.314, 0.306);
const CARD_BLACK: vec3f = vec3f(0.169, 0.161, 0.169);

fn spyder(index: i32) -> vec3f {
    var colors = array<vec3f, SPYDER_TOTAL>(
        LOW_SAT_RED, LOW_SAT_YELLOW, LOW_SAT_GREEN, LOW_SAT_CYAN, LOW_SAT_BLUE, LOW_SAT_MAGENTA,
        RED_TINT_10, GREEN_TINT_10, BLUE_TINT_10, RED_TONE_90, GREEN_TONE_90, BLUE_TONE_90,
        LIGHTEST_SKIN, LIGHTER_SKIN, MODERATE_SKIN, MEDIUM_SKIN, DEEP_SKIN, GRAY_95,
        GRAY_05, GRAY_10, GRAY_30, GRAY_50, GRAY_70, GRAY_90,
        AQUA, LAVENDER, EVERGREEN, STEEL_BLUE, CLASSIC_LIGHT_SKIN, CLASSIC_DARK_SKIN,
        PRIMARY_ORANGE, BLUEPRINT, PINK, VIOLET, APPLE_GREEN, SUNFLOWER,
        PRIMARY_CYAN, PRIMARY_MAGENTA, PRIMARY_YELLOW, PRIMARY_RED, PRIMARY_GREEN, PRIMARY_BLUE,
        CARD_WHITE, GRAY_20, GRAY_40, GRAY_60, GRAY_80, CARD_BLACK
    );
    let idx = index % i32(SPYDER_TOTAL);
    return colors[idx];
}

fn spyderA(index: i32) -> vec3f {
    return spyder(index);
}

fn spyderB(index: i32) -> vec3f {
    return spyder(index + 24);
}
