## Contributing a WebGPU WESL Shader
WESL is a strict superset of WGSL, adding import statements, conditional compilation, and other features.

### To add a shader to Lygia WESL:

1. Add a `.wesl` file alongside the other shader files.
   - The Lygia convention is to have one user-facing function per file.
     This keeps things organized while keeping user application bundle sizes small.
   - It's fine to put multiple type variants of the same function in the same file.
     For example, variants for a `fn` with `f32` and the same `fn` with `vec3f` arguments go in the same file.
1. Add appropriate tests in `test/wesl`.
    - Use `testCompute()` for pure math functions
    - Use `testFragment()` for derivative functions (`fwidth`, `dpdx`, `dpdy`) or texture sampling
    - Use `toMatchImage()` for visual regression tests (filters, generative patterns, complex rendering)

### Notes when porting to WESL:
- WESL file names can't start with a number, and can't be current WGSL or WESL keywords.
- The file/directory hierarchy translates to double colons in WESL:
  ```rs
    import lygia::color::space::hsl2rgb::hsl2rgb;
  ```
  See [WESL imports](https://wesl-lang.dev/spec/Imports) for details. 
- You can also reference other shader modules inline without import statements.
  ```rs
    fn foo() -> f32 {
      let p = lygia::math::consts::PI;
      return p;
    }
  ```
- Conditional transpilation is supported via `@if`, `@else`, and `@elseif` statements.
Conditions can be set at runtime or build time.[^1]
See [WESL Conditions](https://wesl-lang.dev/spec/ConditionalTranslation) for details.

### WESL Resources
WESL is documented at [wesl-lang.dev](https://wesl-lang.dev).
Community support for WESL is available on [GitHub](https://github.com/wgsl-tooling-wg) and on [Discord](https://discord.gg/5UhkaSu4dt).

### Future Plans
- Function overloads or generics to reduce duplication across type variants
- Setting conditions from imported shaders.
