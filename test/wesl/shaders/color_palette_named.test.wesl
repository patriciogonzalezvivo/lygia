// Native WESL tests for named color palettes (zorn, macbeth, spyder)

import wgsl_test::expectNearVec3;
import lygia::color::palette::zorn::zorn;
import lygia::color::palette::macbeth::{macbeth, DARK_SKIN, BLUE_SKY};
import lygia::color::palette::spyder::{spyder, spyderA, spyderB, LOW_SAT_RED, PRIMARY_YELLOW};
import lygia::color::palette::pigments::{TITANIUM_WHITE, CADMIUM_RED};

// ============ zorn palette ============
// 4-color limited palette: Titanium White, Yellow Ochre, Cadmium Red, Ivory Black

@test
fn zornWhite() {
    // Index 0: Titanium White
    expectNearVec3(zorn(0), vec3f(0.973, 0.961, 0.929));
}

@test
fn zornYellowOchre() {
    // Index 1: Yellow Ochre
    expectNearVec3(zorn(1), vec3f(0.706, 0.486, 0.188));
}

@test
fn zornCadmiumRed() {
    // Index 2: Cadmium Red
    expectNearVec3(zorn(2), vec3f(1.000, 0.153, 0.008));
}

@test
fn zornIvoryBlack() {
    // Index 3: Ivory Black
    expectNearVec3(zorn(3), vec3f(0.180, 0.180, 0.180));
}

@test
fn zornWraps() {
    // Index 4 should wrap to 0
    expectNearVec3(zorn(4), zorn(0));
}

// ============ macbeth ColorChecker ============
// 24-color reference target for color calibration

@test
fn macbethDarkSkin() {
    // Index 0: Dark skin
    expectNearVec3(macbeth(0), vec3f(0.46017, 0.33059, 0.27477));
}

@test
fn macbethBlueSky() {
    // Index 2: Blue sky
    expectNearVec3(macbeth(2), vec3f(0.356, 0.472, 0.609));
}

@test
fn macbethOrange() {
    // Index 6: Orange
    expectNearVec3(macbeth(6), vec3f(0.867, 0.487, 0.184));
}

@test
fn macbethYellow() {
    // Index 15: Yellow
    expectNearVec3(macbeth(15), vec3f(0.925, 0.784, 0.094));
}

@test
fn macbethWhite() {
    // Index 18: White
    expectNearVec3(macbeth(18), vec3f(0.956, 0.956, 0.945));
}

@test
fn macbethBlack() {
    // Index 23: Black
    expectNearVec3(macbeth(23), vec3f(0.200, 0.200, 0.204));
}

// ============ spyder ColorChecker 48 ============
// 48-color reference target with side A and side B

@test
fn spyderLowSatRed() {
    // Index 0: Low Saturation Red
    expectNearVec3(spyder(0), vec3f(0.824, 0.475, 0.459));
}

@test
fn spyderPrimaryYellow() {
    // Index 38: Primary Yellow
    expectNearVec3(spyder(38), vec3f(0.961, 0.804, 0.000));
}

@test
fn spyderCardWhite() {
    // Index 42: Card White
    expectNearVec3(spyder(42), vec3f(0.976, 0.949, 0.933));
}

@test
fn spyderCardBlack() {
    // Index 47: Card Black
    expectNearVec3(spyder(47), vec3f(0.169, 0.161, 0.169));
}

@test
fn spyderAMatchesSpyder() {
    // spyderA(5) should equal spyder(5)
    expectNearVec3(spyderA(5), spyder(5));
}

@test
fn spyderBMatchesSpyderOffset() {
    // spyderB(0) should equal spyder(24)
    expectNearVec3(spyderB(0), spyder(24));
}

// ============ exported constants ============
// Verify that named constants can be imported and used directly

@test
fn pigmentsCadmiumRed() {
    // CADMIUM_RED from pigments matches zorn(2)
    expectNearVec3(CADMIUM_RED, vec3f(1.000, 0.153, 0.008));
}

@test
fn pigmentsTitaniumWhite() {
    // TITANIUM_WHITE from pigments matches zorn(0)
    expectNearVec3(TITANIUM_WHITE, vec3f(0.973, 0.961, 0.929));
}

@test
fn macbethDarkSkinConst() {
    // DARK_SKIN constant matches macbeth(0)
    expectNearVec3(DARK_SKIN, macbeth(0));
}

@test
fn macbethBlueSkyConst() {
    // BLUE_SKY constant matches macbeth(2)
    expectNearVec3(BLUE_SKY, macbeth(2));
}

@test
fn spyderLowSatRedConst() {
    // LOW_SAT_RED constant matches spyder(0)
    expectNearVec3(LOW_SAT_RED, spyder(0));
}

@test
fn spyderPrimaryYellowConst() {
    // PRIMARY_YELLOW constant matches spyder(38)
    expectNearVec3(PRIMARY_YELLOW, spyder(38));
}
