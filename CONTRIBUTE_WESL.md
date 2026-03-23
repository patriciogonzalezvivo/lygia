# Contributing to LYGIA WESL

This document covers testing WESL packages for Rust and JavaScript/TypeScript.
For general shader contributions, see [CONTRIBUTE.md](CONTRIBUTE.md).

## npm Package

JavaScript/TypeScript projects use Lygia as an npm package.

Testing (requires GPU):

```sh
pnpm ci:check
```

## Rust Crate

Rust projects use Lygia as a cargo crate.

Test the crate (compiles a shader that imports from lygia):

```sh
cd test/rust_consumer && cargo test -- --nocapture
```