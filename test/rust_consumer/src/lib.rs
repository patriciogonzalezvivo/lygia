//! Test crate that consumes the lygia package

/// The compiled shader source
pub const TEST_SHADER: &str = wesl::include_wesl!("test_shader");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_compiles() {
        println!("Compiled shader:\n{}", TEST_SHADER);

        // Verify the shader contains the entry point
        assert!(TEST_SHADER.contains("fn main()"), "missing main entry point");

        // Verify lygia functions were inlined (including transitive deps)
        assert!(TEST_SHADER.contains("saturate"), "missing saturate function");
        assert!(TEST_SHADER.contains("luma"), "missing luma function");
        assert!(TEST_SHADER.contains("rgb2luma"), "luma should pull in rgb2luma");
    }
}
