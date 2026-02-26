fn main() {
    // Tell cargo to rerun if wesl.toml or any wesl files change
    println!("cargo:rerun-if-changed=wesl.toml");
    println!("cargo:rerun-if-changed=animation");
    println!("cargo:rerun-if-changed=color");
    println!("cargo:rerun-if-changed=distort");
    println!("cargo:rerun-if-changed=draw");
    println!("cargo:rerun-if-changed=filter");
    println!("cargo:rerun-if-changed=generative");
    println!("cargo:rerun-if-changed=geometry");
    println!("cargo:rerun-if-changed=lighting");
    println!("cargo:rerun-if-changed=math");
    println!("cargo:rerun-if-changed=morphological");
    println!("cargo:rerun-if-changed=sample");
    println!("cargo:rerun-if-changed=sdf");
    println!("cargo:rerun-if-changed=simulate");
    println!("cargo:rerun-if-changed=space");
    println!("cargo:rerun-if-changed=version.wesl");

    // Scan using wesl.toml configuration (respects include = ["**/*.wesl"])
    let pkg = wesl::PkgBuilder::new("lygia")
        .scan_toml(".")
        .expect("failed to scan WESL files from wesl.toml")
        .validate()
        .expect("failed to validate package");

    pkg.build_artifact().expect("failed to build artifact");
}
