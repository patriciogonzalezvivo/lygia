fn main() {
    println!("cargo:rerun-if-changed=src/shaders");

    // Compile a shader that uses lygia
    wesl::Wesl::new("src/shaders")
        .add_package(&lygia::PACKAGE)
        .build_artifact(&"package::main".parse().unwrap(), "test_shader");
}
