import wgsl_test::{expectNearVec3, expectNearVec4};
import lygia::color::levels::inputRange::{levelsInputRange3, levelsInputRange3Float, levelsInputRange4, levelsInputRange4Float};
import lygia::color::levels::gamma::{levelsGamma3, levelsGamma3Float, levelsGamma4, levelsGamma4Float};
import lygia::color::levels::outputRange::{levelsOutputRange3Float, levelsOutputRange4, levelsOutputRange4Float};
import lygia::color::levels::{levels3Float, levels3, levels4, levels4Float};

// Input Range tests

@test fn inputRange3() {
  let color = vec3f(0.3, 0.5, 0.7);
  expectNearVec3(levelsInputRange3(color, vec3f(0.2), vec3f(0.8)), vec3f(0.1667, 0.5, 0.8333));
}

@test fn inputRange3Float() {
  let color = vec3f(0.2, 0.5, 0.8);
  expectNearVec3(levelsInputRange3Float(color, 0.1, 0.9), vec3f(0.125, 0.5, 0.875));
}

@test fn inputRange4() {
  let color = vec4f(0.3, 0.6, 0.9, 0.75);
  expectNearVec4(levelsInputRange4(color, vec3f(0.2, 0.4, 0.5), vec3f(0.8, 0.9, 1.0)), vec4f(0.1667, 0.4, 0.8, 0.75));
}

@test fn inputRange4Float() {
  let color = vec4f(0.15, 0.45, 0.75, 0.95);
  expectNearVec4(levelsInputRange4Float(color, 0.1, 0.8), vec4f(0.0714, 0.5, 0.9286, 0.95));
}

// Gamma tests

@test fn gamma3() {
  let color = vec3f(0.25, 0.5, 0.75);
  expectNearVec3(levelsGamma3(color, vec3f(2.0)), vec3f(0.5, 0.7071, 0.866));
}

@test fn gamma3Float() {
  let color = vec3f(0.16, 0.36, 0.64);
  expectNearVec3(levelsGamma3Float(color, 2.0), vec3f(0.4, 0.6, 0.8));
}

@test fn gamma4() {
  let color = vec4f(0.25, 0.5, 0.75, 0.9);
  expectNearVec4(levelsGamma4(color, vec3f(2.0, 1.5, 3.0)), vec4f(0.5, 0.63, 0.9086, 0.9));
}

@test fn gamma4Float() {
  let color = vec4f(0.09, 0.25, 0.49, 0.85);
  expectNearVec4(levelsGamma4Float(color, 2.0), vec4f(0.3, 0.5, 0.7, 0.85));
}

// Output Range tests

@test fn outputRange3Float() {
  let color = vec3f(0.0, 0.5, 1.0);
  expectNearVec3(levelsOutputRange3Float(color, 0.2, 0.9), vec3f(0.2, 0.55, 0.9));
}

@test fn outputRange4() {
  let color = vec4f(0.25, 0.5, 0.75, 0.8);
  expectNearVec4(levelsOutputRange4(color, vec3f(0.1, 0.2, 0.3), vec3f(0.8, 0.9, 1.0)), vec4f(0.275, 0.55, 0.825, 0.8));
}

@test fn outputRange4Float() {
  let color = vec4f(0.2, 0.6, 0.8, 0.7);
  expectNearVec4(levelsOutputRange4Float(color, 0.3, 0.95), vec4f(0.43, 0.69, 0.82, 0.7));
}

// Full levels pipeline tests

@test fn levels3FloatPipeline() {
  let color = vec3f(0.3, 0.5, 0.7);
  expectNearVec3(levels3Float(color, 0.2, 2.0, 0.8, 0.1, 0.9), vec3f(0.4266, 0.6657, 0.8303));
}

@test fn levels3Pipeline() {
  let color = vec3f(0.4, 0.6, 0.8);
  expectNearVec3(levels3(color, vec3f(0.2), vec3f(2.0), vec3f(0.9), vec3f(0.1), vec3f(0.8)), vec3f(0.4742, 0.6292, 0.7481));
}

@test fn levels4Pipeline() {
  let color = vec4f(0.4, 0.6, 0.8, 0.75);
  expectNearVec4(levels4(color, vec3f(0.2), vec3f(2.0), vec3f(0.9), vec3f(0.1), vec3f(0.8)), vec4f(0.4742, 0.6292, 0.7481, 0.75));
}

@test fn levels4FloatPipeline() {
  let color = vec4f(0.5, 0.7, 0.3, 0.85);
  expectNearVec4(levels4Float(color, 0.3, 1.5, 0.8, 0.2, 0.9), vec4f(0.58, 0.8032, 0.2, 0.85));
}
