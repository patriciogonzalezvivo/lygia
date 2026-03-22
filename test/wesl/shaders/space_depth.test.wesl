import wgsl_test::expectNear;
import lygia::space::linearizeDepth::linearizeDepth;
import lygia::space::depth2viewZ::depth2viewZ;
import lygia::space::viewZ2depth::viewZ2depth;

@test fn testLinearizeDepth() {
  // near=0.1, far=100.0, depth=0.5
  // d = 2*0.5 - 1 = 0
  // result = (2 * 0.1 * 100) / (100 + 0.1 - 0 * (100 - 0.1)) = 20 / 100.1
  expectNear(linearizeDepth(0.5, 0.1, 100.0), 0.1998);
}

@test fn testDepth2viewZ() {
  // (near * far) / ((far - near) * depth - far)
  // = (1 * 100) / ((99) * 0.5 - 100) = 100 / -50.5
  expectNear(depth2viewZ(0.5, 1.0, 100.0), -1.9802);
}

@test fn testViewZ2depth() {
  // Should reverse the depth2viewZ operation
  expectNear(viewZ2depth(-1.9802, 1.0, 100.0), 0.5);
}
