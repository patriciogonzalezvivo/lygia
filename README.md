<img src="https://lygia.xyz/imgs/lygia.svg" alt="LYGIA" width="200" style="display: block; margin-left: auto; margin-right: auto; filter: drop-shadow(2px 3px 4px gray);">

# LYGIA Shader Library 

LYGIA is a shader library of reusable functions that will let you prototype, port or ship a project in just few minutes. It's very granular, flexible and efficient. Support multiple shading languages and can easily be added to any project, enviroment or framework of your choice. 

[![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub)](https://github.com/sponsors/patriciogonzalezvivo)

Here are a couple of integrations examples:

<p style="text-align: center;" >
    <a href="https://github.com/patriciogonzalezvivo/lygia_unity_examples"><img src="https://lygia.xyz/imgs/unity.png" alt="uUnity" width="64" /></a>
    <a href="https://github.com/guidoschmidt/lygia_threejs_examples"><img src="https://lygia.xyz/imgs/threejs.png" alt="threejs" width="64" /></a>
    <a href="https://github.com/patriciogonzalezvivo/lygia_p5_examples"><img src="https://lygia.xyz/imgs/p5.png" alt="p5" width="64" /></a>
    <a href="https://editor.p5js.org/patriciogonzalezvivo/sketches"><img src="https://lygia.xyz/imgs/p5js.png" alt="p5js" width="64" /></a>
    <a href="https://github.com/patriciogonzalezvivo/lygia_of_examples"><img src="https://lygia.xyz/imgs/of.png" alt="openFrameworks" width="64" /></a>
    <a href="https://www.curseforge.com/minecraft/search?page=1&pageSize=20&sortType=1&search=LYGIA%20Shader%20Library"><img src="https://lygia.xyz/imgs/minecraft.png" alt="minecraft" width="64" /></a>
    <a href="https://www.figma.com/community/plugin/1138854718618193875"><img src="https://lygia.xyz/imgs/figma.png" alt="Figma" width="64" /></a>
    <a href="https://github.com/vectorsize/lygia-td"><img title="static-resolver by vectorsize" src="https://lygia.xyz/imgs/td.png" alt="touchDesigner" width="64" /></a>
    <a href="https://github.com/kujohn/lygia_ogl_examples"><img src="https://lygia.xyz/imgs/ogl.png" alt="ogl" width="64" /></a>
    <a href="https://www.npmjs.com/package/lygia"><img src="https://lygia.xyz/imgs/npm.png" alt="npm" width="64" /></a>
    <a href="https://github.com/patriciogonzalezvivo/lygia_examples"><img src="https://lygia.xyz/imgs/glslViewer.png" alt="glslViewer" width="64" /></a>
    <a href="https://observablehq.com/@radames/hello-lygia-shader-library"><img src="https://lygia.xyz/imgs/ob.png" alt="ob" width="64" /></a>
    <a href="https://codesandbox.io/s/lygia-react-starter-fftx6p"><img src="https://lygia.xyz/imgs/r3f.png" alt="r3rf" width="64" /></a>
    <a href="https://github.com/irmf/irmf-examples/tree/master/examples/028-lygia"><img src="https://lygia.xyz/imgs/irmf.png" alt="irmf" width="64" /></a>
    <a href="https://github.com/ossia/score-examples"><img src="https://lygia.xyz/imgs/ossia.png" alt="Ossia" width="64" /></a>
    <a href="https://www.productioncrate.com/laforge/"><img src="https://lygia.xyz/imgs/laforge.png" alt="laforge" width="64" /></a>
    <a href="https://synesthesia.live/"><img src="https://lygia.xyz/imgs/synesthesia.png" alt="synesthesia" width="64" /></a>
</p>


## How to use it?

In your shader `#include` the functions you need:

<div class="codeAndCanvas" data="example.frag">

	#ifdef GL_ES
    precision mediump float;
    #endif

    uniform vec2    u_resolution;
    uniform float   u_time;

    #include "lygia/space/ratio.glsl"
    #include "lygia/math/decimate.glsl"
    #include "lygia/draw/circle.glsl"

    void main(void) {
        vec3 color = vec3(0.0);
        vec2 st = gl_FragCoord.xy/u_resolution.xy;
        st = ratio(st, u_resolution);
        
        color = vec3(st.x,st.y,abs(sin(u_time)));
        color = decimate(color, 20.);
        color += circle(st, .5, .1);
        
        gl_FragColor = vec4(color, 1.0);
    }
    
</div>

Then you need to resovle this dependencies, the fastest way would be to drag&drop your shader file here:

<div class="container">
    <div class="file-drop-area">
    <span class="file-msg">Drop you shader file <a href="https://lygia.xyz/">here</a></span>
    </div>
</div>

The other options is using a **local** version that then you can bundle into your project, or use the **server** to resolve the dependencies online.

### LYGIA Locally

If you are working **locally** in an environment that can resolve `#include` dependencies, just clone LYGIA into your project relative to the shader you are loading:

```bash
    git clone https://github.com/patriciogonzalezvivo/lygia.git
```

or as a submodule:

```bash
    git submodule add https://github.com/patriciogonzalezvivo/lygia.git
```

or you may clone LYGIA without the git history and reduce the project size (9MB+) with the following command:

```bash
    npx degit https://github.com/patriciogonzalezvivo/lygia.git lygia
```

### LYGIA server

If you are working on a **cloud platform** you probably want to resolve the dependencies without needing to install anything. Just add a link to `https://lygia.xyz/resolve.js` (JS) or `https://lygia.xyz/resolve.esm.js` (ES6 module): 

```html
    <!-- as a JavaScript source -->
    <script src="https://lygia.xyz/resolve.js"></script>

    <!-- Or as a ES6 module -->
    <script type="module">
        import resolveLygia from "https://lygia.xyz/resolve.esm.js"
    </script>
```

To then resolve the dependencies by passing a `string` or `strings[]` to `resolveLygia()` or `resolveLygiaAsync()`:

```js
    // 1. FIRST

    // Sync resolver, one include at a time
    vertSource = resolveLygia(vertSource);
    fragSource = resolveLygia(fragSource);

    // OR.
    
    // ASync resolver, all includes in parallel calls
    vertSource = resolveLygiaAsync(vertSource);
    fragSource = resolveLygiaAsync(fragSource);
    
    // 2. SECOND

    // Use the resolved source code 
    shdr = createShader(vertSource, fragSource);
```

This this function can also resolve dependencies to previous versions of LYGIA by using this pattern `lygia/vX.X.X/...` on you dependency paths. For example:

```glsl
#include "lygia/v1.0.0/math/decimation.glsl"
#include "lygia/v1.1.0/math/decimation.glsl"
```

### Integrations examples

Learn more about LYGIA and how to use it from these **examples**:

* [2D examples for Processing (GLSL)](https://github.com/patriciogonzalezvivo/lygia_p5_examples)
* [2D/3D examples for P5.js (GLSL)](https://editor.p5js.org/patriciogonzalezvivo/sketches)
* [2D examples for Three.js + React (GLSL)](https://codesandbox.io/s/lygia-react-starter-fftx6p) by [Eduard Fossas](https://eduardfossas.vercel.app/)
* [2D examples for Three.js (GLSL)](https://github.com/patriciogonzalezvivo/lygia_threejs_examples)
* [3D examples for Three.js (GLSL)](https://github.com/guidoschmidt/lygia_threejs_examples) by [Guido Schmidt](https://guidoschmidt.cc/)
* [2D examples for OpenFrameworks (GLSL)](https://github.com/patriciogonzalezvivo/lygia_of_examples) 
* [2D/3D examples for Unity3D (HLSL)](https://github.com/patriciogonzalezvivo/lygia_unity_examples)
* [2D examples for Touch Designer (GLSL)](https://derivative.ca/community-post/asset/lygia-touchdesginer/66804) (dynamic resolver) by [Leith Ben Abdessalem](https://leithba.com)
* [2D examples for Touch Designer (GLSL)](https://github.com/vectorsize/lygia-td) (static resolver) by [Victor Saz](https://github.com/vectorsize)
* [2D examples on Observable Notebook (GLSL)](https://observablehq.com/@radames/hello-lygia-shader-library) by [Radames Ajna](https://twitter.com/radamar)
* [Figma's noise&texture plugin](https://www.figma.com/community/plugin/1138854718618193875) by [Rogie King](https://twitter.com/rogie). You will need to go to the "Custom" tab on the plugin to edit shaders and load LYGIA modules  
* [3D example on Irmf](https://github.com/irmf/irmf-examples/tree/master/examples/028-lygia) by [Glenn Lewis](https://github.com/gmlewis)
* [2D/3D examples on GlslViewer (GLSL)](https://github.com/patriciogonzalezvivo/lygia_examples)
* [2D examples on Ossia](https://github.com/ossia/score-examples) by [Jean-Michaël Celerier](https://jcelerier.name/)
* [Ogl integration](https://github.com/kujohn/lygia_ogl_examples) by [John Ku](https://github.com/kujohn)
* [2D templates for Ogl(TS) and Three.js(JS/TS)](https://github.com/cdaein/create-ssam) by [Daeinc](https://paperdove.com)

For more information, guidance, or feedback about using LYGIA, join [#Lygia channel on shader.zone discord](https://shader.zone/).


### How is it organized?

The functions are divided into different categories:

* [`math/`](https://lygia.xyz/math): general math functions and constants: `PI`, `SqrtLength()`, etc. 
* [`space/`](https://lygia.xyz/space): general spatial operations: `scale()`, `rotate()`, etc. 
* [`color/`](https://lygia.xyz/color): general color operations: `luma()`, `saturation()`, blend modes, palettes, color space conversion, and tonemaps.
* [`animation/`](https://lygia.xyz/animation): animation operations: easing
* [`generative/`](https://lygia.xyz/generative): generative functions: `random()`, `noise()`, etc. 
* [`sdf/`](https://lygia.xyz/sdf): signed distance field functions.
* [`draw/`](https://lygia.xyz/draw): drawing functions like `digits()`, `stroke()`, `fill`, etc/.
* [`sample/`](https://lygia.xyz/sample): sample operations
* [`filter/`](https://lygia.xyz/filter): typical filter operations: different kind of blurs, mean and median filters.
* [`distort/`](https://lygia.xyz/distort): distort sampling operations
* [`lighting/`](https://lygia.xyz/lighting): different lighting models and functions for foward/deferred/raymarching rendering
* [`geometry/`](https://lygia.xyz/geometry): operation related to geometries: intersections and AABB accelerating structures.
* [`morphological/`](https://lygia.xyz/morphological): morphological filters: dilation, erosion, alpha and poisson fill.

### Flexible how?

There are some functions whose behavior can be changed using the `#defines` keyword before including it. For example, [gaussian blurs](filter/gaussianBlur.glsl) are usually are done in two passes. By default, these are performed on their 1D version, but if you are interested in using a 2D kernel, all in the same pass, you will need to add the `GAUSSIANBLUR_2D` keyword this way:

```glsl

    #define GAUSSIANBLUR_2D
    #include "filter/gaussianBlur.glsl"

    void main(void) {
        ...
        
        vec2 pixel = 1./u_resolution;
        color = gaussianBlur(u_tex0, uv, pixel, 9);
        ...
    }

```


## Design Principles

1. It relies on `#include "path/to/file.*lsl"` which is defined by Khronos GLSL standard and requires a typical C-like pre-compiler MACRO which is easy to implement with just basic string operations to resolve dependencies. 

Here you can find some implementations on different languages:

  - C#:
  
    . [GLSLIncludes](https://github.com/seb776/GLSLIncludes) a small utility to add the include feature to glsl by [z0rg](https://github.com/seb776).

  - C++:

    . [VERA's routines](https://github.com/patriciogonzalezvivo/vera/blob/main/src/ops/fs.cpp#L110-L171) for resolving GLSL dependencies.

  - Python:

    . [Small and simple routing to resolve includes](https://gist.github.com/patriciogonzalezvivo/9a50569c2ef9b08058706443a39d838e)

  - JavaScript: 
  
    . [vanilla JS (online resolver)](https://lygia.xyz/resolve.js) This small file brings `resolveLygia()` which takes a `string` or `string[]` and parses it, solving all the `#include` dependencies into a single `string` you can load on your shaders. It also has a `resolveLygiaAsync()` version that resolves all the dependencies in parallel. Both support dependencies to previous versions of LYGIA by using this pattern `lygia/vX.X.X/...` on you dependency paths. 
  
    . [npm module (online resolver)](https://www.npmjs.com/package/resolve-lygia) by Eduardo Fossas. This is bring the same `resolveLygia()` and `resolveLygiaAsync()` function but as a npm module.

    . [vite glsl plugin (local bundle)](https://github.com/UstymUkhman/vite-plugin-glsl) by Ustym Ukhman. Imports `.glsl` local dependencies, or load inline shaders through vite.
  
    . [esbuild glsl plugin (local bundle)](https://github.com/ricardomatias/esbuild-plugin-glsl-include) by Ricardo Matias. Imports local `.glsl` dependencies through esbuild.

    . [webpack glsl plugin (local bundle)](https://github.com/grieve/webpack-glsl-loader) by Ryan Grieve that imports local `.glsl` dependencies through webpack.
        
* It's **very granular**. One function per file. The file and the function share the same name, namely: `myFunc.glsl` contains `myFunct()`. There are some files that just include a collection of files inside a folder with the same name. For example:

```
    color/blend.glsl
    // which includes
    color/blend/*.glsl

```

* It's **multi language**. Right now most of is GLSL (`*.glsl`) and HLSL (`*.hlsl`), but we are slowly extending to WGSL (`*.wgsl`), CUDA (`*.cuh`) and Metal (`*.msl`).

```
    math/mirror.glsl
    math/mirror.hlsl
    math/mirror.wgsl
    math/mirror.msl
    math/mirror.cuh
```

* **Self documented**. Each file contains a structured comment (in YAML) at the top of the file. This one contains the name of the original author, description, use, and `#define` options

```glsl

    /*
    contributors: <FULL NAME>
    description: [DESCRIPTION + URL]
    use: <vec2> myFunc(<vec2> st, <float> x [, <float> y])
    options:
        - MYFUNC_TYPE
        - MYFUNC_SAMPLER_FNC()
    */

```

* Prevents **name collisions** by using the following pattern where `FNC_` is followed with the function name:

```glsl

    #ifndef FNC_MYFUNC
    #define FNC_MYFUNC

    float myFunc(float in) {
        return in;
    }

    #endif

```

* **Templating capabilities through `#defines`**. Probably the most frequent use is templating the sampling function for reusability. The `#define` options start with the name of the function, in this example `MYFUNC_`. They are added as `options:` in the header.
 
```glsl

    #ifndef MYFUNC_TYPE
    #define MYFUNC_TYPE vec4
    #endif

    #ifndef MYFUNC_SAMPLER_FNC
    #define MYFUNC_SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
    #endif

    #ifndef FNC_MYFUNC
    #define FNC_MYFUNC
    MYFUNC_TYPE myFunc(SAMPLER_TYPE tex, vec2 st) {
        return MYFUNC_SAMPLER_FNC(tex, st);
    }
    #endif

```

* **Function Overloading**. Arguments are arranged in such a way that optional elements are at the end. When possible sort them according their memory size (except textures that remain at the top). Ex.: `SAMPLER_TYPE, mat4, mat3, mat2, vec4, vec3, vec2, float, ivec4, ivec3, ivec2, int, bool`

```glsl

    /*
    ...
    use: myFunc(<vec2> st, <vec2|float> x[, <float> y])
    */

    #ifndef FNC_MYFUNC
    #define FNC_MYFUNC
    vec2 myFunc(vec2 st, vec2 x) {
        return st * x;
    }

    vec2 myFunc(vec2 st, float x) {
        return st * x;
    }

    vec2 myFunc(vec2 st, float x, float y) {
        return st * vec2(x, y);
    }
    #endif

```

## Contributions

LYGIA has a long way to go. Your support will be appreciated and rewarded! All contributors are automatically added to the [commercial license](https://lygia.xyz/license). This support can take multiple forms:

* fixing bugs!
* expanding the cross-compatibility between languages GLSL/HLSL/MSL/WGSL/CUDA
* contributing new functions
* adding new examples and integrations for new environments like: [GoDot](https://godotengine.org/), [ISF](https://isf.video/), [MaxMSP](https://cycling74.com/products/max), etc.
* through [sponsorships](https://github.com/sponsors/patriciogonzalezvivo)


## License 

LYGIA belongs to those that support it. For that it uses a dual-licensed under the [Prosperity License](https://prosperitylicense.com/versions/3.0.0) and the [Patron License](https://lygia.xyz/license) for [sponsors](https://github.com/sponsors/patriciogonzalezvivo) and [contributors](https://github.com/patriciogonzalezvivo/lygia/graphs/contributors).

[Sponsors](https://github.com/sponsors/patriciogonzalezvivo) and [contributors](https://github.com/patriciogonzalezvivo/lygia/graphs/contributors) are automatically added to the [Patron License](https://lygia.xyz/license) and they can ignore any non-commercial rule of the [Prosperity License](https://prosperitylicense.com/versions/3.0.0) software.

It's also possible to get a permanent commercial license hooked to a single and specific version of LYGIA.

If you have doubts please reaching out to patriciogonzalezvivo at gmail dot com

## Credits

Created and mantained by [Patricio Gonzalez Vivo](https://patriciogonzalezvivo.com/)( <a rel="me" href="https://merveilles.town/@patricio">Mastodon</a> | [Twitter](https://twitter.com/patriciogv) | [Instagram](https://www.instagram.com/patriciogonzalezvivo/) | [GitHub](https://github.com/sponsors/patriciogonzalezvivo) ) and every direct or indirect [contributors](https://github.com/patriciogonzalezvivo/lygia/graphs/contributors) to the GitHub. This library has been built over years, and in many cases on top of the work of brilliant and generous people like: [Inigo Quiles](https://www.iquilezles.org/), [Morgan McGuire](https://casual-effects.com/), [Alan Wolfe](https://blog.demofox.org/), [Hugh Kennedy](https://github.com/hughsk), [Matt DesLauriers](https://www.mattdesl.com/), and many others.


## Get the latest news and releases

Sign up for the news letter below, join [the LYGIA's channel on Discord](https://shader.zone) or follow the [Github repository](https://github.com/patriciogonzalezvivo/lygia)
