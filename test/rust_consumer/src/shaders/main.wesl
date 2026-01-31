// Test shader that imports from lygia
import lygia::math::saturate::saturate;
import lygia::color::luma::luma3;

@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main() {
    // Test saturate function
    let clamped = saturate(1.5);

    // Test luma function (should bring in rgb2luma as transitive dep)
    let gray = luma3(vec3f(0.5, 0.3, 0.2));

    output[0] = clamped;
    output[1] = gray;
}
