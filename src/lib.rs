//! LYGIA - The biggest shader library for WESL/WGSL
//!
//! This crate provides the LYGIA shader library as a WESL package that can be
//! used in Rust projects with the `wesl` crate.
//!
//! # Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [build-dependencies]
//! lygia = "0.1"
//! wesl = { version = "0.3", features = ["package"] }
//! ```
//!
//! In your `build.rs`:
//!
//! ```ignore
//! fn main() {
//!     wesl::Wesl::new("src/shaders")
//!         .add_package(&lygia::PACKAGE)
//!         .build_artifact(&"package::main".parse().unwrap(), "my_shader");
//! }
//! ```
//!
//! Then in your shader files, import from lygia:
//!
//! ```wgsl
//! import lygia::math::saturate::saturate;
//! import lygia::color::luma::luma;
//! ```

wesl::wesl_pkg!(pub lygia);
pub use self::lygia::*;
