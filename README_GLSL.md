LYGIA relies on `#include "path/to/file.glsl"` to resolve dependencies, which is defined by Khronos GLSL standard, but it's up to the project developer to implement. Good news is that it's not that hard, it just requires a typical C-like MACRO pre-compiler, which is easy to implement with just basic string operations.

Here you can find some examples in different languages:

- C#:

  - [GLSLIncludes](https://github.com/seb776/GLSLIncludes) a small utility to add the include feature to GLSL by [z0rg](https://github.com/seb776).

- C++:

  - [VERA's routines](https://github.com/patriciogonzalezvivo/vera/blob/main/src/ops/fs.cpp#L110-L171) for resolving GLSL dependencies.

- Python:

  - [Small and simple routing to resolve includes](https://gist.github.com/patriciogonzalezvivo/9a50569c2ef9b08058706443a39d838e)

- JavaScript:

  - [vanilla JS (online resolver)](https://lygia.xyz/resolve.js) This small file brings `resolveLygia()` which takes a `string` or `string[]` and parses it, solving all the `#include` dependencies into a single `string` you can load on your shaders. It also has a `resolveLygiaAsync()` version that resolves all the dependencies in parallel. Both support dependencies to previous versions of LYGIA by using this pattern `lygia/vX.X.X/...` on you dependency paths.

  - [npm module (online resolver)](https://www.npmjs.com/package/resolve-lygia) by Eduardo Fossas. This brings the same `resolveLygia()` and `resolveLygiaAsync()` functions but as npm module.

  - [vite glsl plugin (local bundle)](https://github.com/UstymUkhman/vite-plugin-glsl) by Ustym Ukhman. Imports `.glsl` local dependencies, or loads inline shaders through vite.

  - [esbuild glsl plugin (local bundle)](https://github.com/ricardomatias/esbuild-plugin-glsl-include) by Ricardo Matias. Imports local `.glsl` dependencies through esbuild.

  - [webpack glsl plugin (local bundle)](https://github.com/grieve/webpack-glsl-loader) by Ryan Grieve that imports local `.glsl` dependencies through webpack.
