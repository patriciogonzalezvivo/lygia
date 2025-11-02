This page describes how to link Lygia shader functions
into your WebGPU application using WESL tools.

[WESL](https://wesl-lang.dev) is a superset of WGSL that adds
features with community-supported tools.
WESL tools are available in Rust and JavaScript/TypeScript.

_Most (but not all) Lygia GLSL shaders are now available
for WebGPU using WESL.
If a Lygia function you need is missing,
please file an [issue](https://github.com/patriciogonzalezvivo/lygia/issues)
or help [contribute](./README_WESL.md)._[^1][^2]

## Using JavaScript or TypeScript

Install with `npm install lygia` or `pnpm install lygia`
([lygia npm package](https://www.npmjs.com/package/lygia)).
Once you install, 500+ Lygia functions
and constants will be available for you to use
via `import` statements in your application shader code.

```rs
import lygia::math::consts::PI;

fn main() {
  let p = PI;
}
```

The JS tools automatically tree-shake, including only the Lygia
functions your application uses.

WESL integrates with popular bundlers (vite, webpack, rollup) through plugins,
or for custom build pipelines, you can use the `wesl-link` command-line tool
or the link API. 

### Using a JavaScript/TypeScript Bundler

If you build your application with a JavaScript/TypeScript bundler
like `vite`, `webpack` or `rollup`,
install the
[wesl](https://www.npmjs.com/package/wesl?activeTab=readme) and
[wesl-plugin](https://www.npmjs.com/package/wesl-plugin) packages.

#### Runtime Linking with a Bundler
Import shader code into your JavaScript or TypeScript application using the 
import statements suffixed with `?link` and the shaders will be linked together at runtime:

```ts
import appWesl from "../shaders/app.wesl?link";
import { link } from "wesl";

const linked = await link(appWesl);
linked.createShaderModule(gpuDevice);
```

For more details, check the [WESL bundler documentation](https://wesl-lang.dev/docs/JavaScript-Builds#wesl-with-javascript-bundlers) or refer to this
[lygia example using vite](https://stackblitz.com/github/wgsl-tooling-wg/examples/tree/main/lygia-example?file=README.md).

#### Static Linking with a Bundler
Alternatively,
you can statically link your shaders in advance using the `?static` suffix:
```ts
import appWgsl from "../shaders/app.wesl?static";
```
See the [lygia static linking example](https://stackblitz.com/github/wgsl-tooling-wg/examples/tree/main/lygia-static-example)
for details. 


#### Runtime vs. Static Linking
Static linking bundles your shader modules together into a single transpiled WGSL string.
Static linking means you don't need WESL linker in your runtime bundle (~15KB savings).
However, static linking is less flexible 
because your application can't use WESL's conditional compilation
features (`@if` directives) to adapt shaders at runtime based on GPU capabilities
or user configuration.

Many Lygia shader functions include `@if` [conditions](https://wesl-lang.dev/spec/ConditionalTranslation).
For example see `@if(YUV_SDTV)` in [`yuv2rgb`](https://github.com/patriciogonzalezvivo/lygia/blob/main/color/space/yuv2rgb.wesl).
With runtime linking, you can set these conditions dynamically.
With static linking, conditions are resolved at build time.

### Command Line Linking

For custom build pipelines, use the `wesl-link` command-line tool to statically link shaders.

#### Linking Application Shaders
Link your application shader file that imports Lygia functions:

```sh
npx wesl-link ./shaders/main.wesl
```

See the [Lygia CLI linking example](https://stackblitz.com/github/wgsl-tooling-wg/examples/tree/main/lygia-cli-example?file=README.md) for details.

#### Linking Lygia Modules on the Command Line
To get standalone WGSL for a specific Lygia module, 
call wesl-link on that module.

**From the npm package:**
```sh
npx wesl-link lygia::color::layer::addSourceOver
```

**From a lygia repository clone:**
```sh
npx wesl-link package::color::layer::addSourceOver

# alternate syntax
npx wesl-link color/layer/addSourceOver.wesl
```

Either approach produces the WGSL for the requested Lygia module linked with its dependencies.

### Link Using the API
You can use the linking API directly to build custom solutions:

```ts
import { link } from "wesl";

const main = `import lygia::math::consts::PI; ...`;
const linked = await link({weslSrc: {main }});
const shaderModule = linked.createShaderModule(gpuDevice);
```

See the [API documentation](https://wesl-lang.dev/docs/JavaScript-Builds) for details.

### Additional WESL Examples

More WESL examples are available [here](https://github.com/wgsl-tooling-wg/examples).
Most examples run with one click in a browser sandbox.
The examples can also be used as starter templates with `degit`.

## Using Rust

```sh
cargo add lygia
```

### Linking at build time
```sh
cargo add --build wesl
```

```rs
/// build.rs
fn main() {
    wesl::Wesl::new("src/shaders").build_artifact("main.wesl", "my_shader");
}
```

### Linking at run-time

```sh
cargo add wesl
```

```rs
let shader_string = Wesl::new("src/shaders")
    .compile("main.wesl")
    .inspect_err(|e| eprintln!("WESL error: {e}")) // pretty errors with `display()`
    .unwrap()
    .to_string();
```

### Using the Rust CLI tool
```sh
cargo install wesl-cli
wesl compile <path/to/shader.wesl>
```


### WESL Rust Documentation
See [Getting Started Rust](https://wesl-lang.dev/docs/Getting-Started-Rust), 
the [wesl crate documentation](https://docs.rs/wesl/latest/wesl/),
and [WESL rust examples](https://github.com/wgsl-tooling-wg/wesl-rs/tree/main/examples).


## About WESL
WESL extends WGSL with:
- `import` statements to split shader code across files and load npm/cargo libraries
- `@if @else @elseif` statements to assemble specialized shaders at build time or runtime

Current WESL tools include:
- Linkers in Rust and JavaScript (to combine WGSL/WESL files into applications)
- Syntax highlighters for zed, helix and nvim

### Coming Soon
Additional WESL tools in development:
- HTML [documentation generator](https://github.com/jannik4/wesldoc)
- [Language server](https://github.com/wgsl-analyzer/wgsl-analyzer)
- VSCode plugin
- Code formatter

Read more about WESL at [wesl-lang.dev](https://wesl-lang.dev).

[^1]: Lygia functions are small and self-contained.
You can just copy, paste and edit them into your app if you prefer doing things manually!

[^2]: Lygia currently hosts two versions of WebGPU shaders. 
The newer versions use the WESL language and have a `.wesl` suffix.
For new Lygia users, we recommend `.wesl`. 
(The older versions use a `.wgsl` suffix, but are less complete and use a custom `#include` syntax.) 
