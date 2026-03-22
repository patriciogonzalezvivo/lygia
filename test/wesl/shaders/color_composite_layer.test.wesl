import wgsl_test::expectNearVec4;
import lygia::color::layer::multiplySourceOver::layerMultiplySourceOver4;
import lygia::color::layer::screenSourceOver::layerScreenSourceOver4;
import lygia::color::layer::addSourceOver::layerAddSourceOver4;
import lygia::color::layer::overlaySourceOver::layerOverlaySourceOver4;
import lygia::color::layer::darkenSourceOver::layerDarkenSourceOver4;
import lygia::color::layer::lightenSourceOver::layerLightenSourceOver4;
import lygia::color::layer::differenceSourceOver::layerDifferenceSourceOver4;
import lygia::color::layer::exclusionSourceOver::layerExclusionSourceOver4;
import lygia::color::layer::phoenixSourceOver::layerPhoenixSourceOver4;
import lygia::color::layer::subtractSourceOver::layerSubtractSourceOver4;

@test fn layerMultiply() {
  let s = vec4f(0.8, 0.6, 0.4, 0.75);
  let d = vec4f(0.5, 0.7, 0.9, 0.5);
  expectNearVec4(layerMultiplySourceOver4(s, d), vec4f(0.3625, 0.4025, 0.3825, 0.875));
}

@test fn layerScreen() {
  let s = vec4f(0.6, 0.4, 0.2, 0.5);
  let d = vec4f(0.3, 0.5, 0.7, 0.6);
  expectNearVec4(layerScreenSourceOver4(s, d), vec4f(0.45, 0.5, 0.59, 0.8));
}

@test fn layerAdd() {
  let s = vec4f(0.3, 0.4, 0.5, 0.6);
  let d = vec4f(0.2, 0.3, 0.4, 0.5);
  expectNearVec4(layerAddSourceOver4(s, d), vec4f(0.34, 0.48, 0.62, 0.8));
}

@test fn layerOverlay() {
  let s = vec4f(0.7, 0.3, 0.5, 0.8);
  let d = vec4f(0.4, 0.6, 0.5, 0.4);
  expectNearVec4(layerOverlaySourceOver4(s, d), vec4f(0.544, 0.336, 0.44, 0.88));
}

@test fn layerDarken() {
  let s = vec4f(0.3, 0.7, 0.5, 0.5);
  let d = vec4f(0.6, 0.4, 0.5, 0.5);
  expectNearVec4(layerDarkenSourceOver4(s, d), vec4f(0.3, 0.3, 0.375, 0.75));
}

@test fn layerLighten() {
  let s = vec4f(0.3, 0.7, 0.5, 0.5);
  let d = vec4f(0.6, 0.4, 0.5, 0.5);
  expectNearVec4(layerLightenSourceOver4(s, d), vec4f(0.45, 0.45, 0.375, 0.75));
}

@test fn layerDifference() {
  let s = vec4f(0.8, 0.3, 0.6, 0.7);
  let d = vec4f(0.5, 0.7, 0.4, 0.6);
  expectNearVec4(layerDifferenceSourceOver4(s, d), vec4f(0.3, 0.406, 0.212, 0.88));
}

@test fn layerExclusion() {
  let s = vec4f(0.6, 0.4, 0.8, 0.5);
  let d = vec4f(0.3, 0.7, 0.2, 0.5);
  expectNearVec4(layerExclusionSourceOver4(s, d), vec4f(0.345, 0.445, 0.39, 0.75));
}

@test fn layerPhoenix() {
  let s = vec4f(0.7, 0.5, 0.3, 0.6);
  let d = vec4f(0.4, 0.6, 0.8, 0.4);
  expectNearVec4(layerPhoenixSourceOver4(s, d), vec4f(0.484, 0.636, 0.428, 0.76));
}

@test fn layerSubtract() {
  let s = vec4f(0.8, 0.5, 0.3, 0.5);
  let d = vec4f(0.4, 0.6, 0.7, 0.5);
  expectNearVec4(layerSubtractSourceOver4(s, d), vec4f(0.2, 0.2, 0.175, 0.75));
}
