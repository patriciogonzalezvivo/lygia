## Contributing a WebGPU TSL Shader
TSL is the new shader language used by Three.js for `WebGPURenderer`. This project uses TypeScript to add TSL nodes and ESM modules to import and export them.

### To add a shader to Lygia TSL:

1. Add a `.ts` file alongside the other shader files.
   - The Lygia convention is to have one user-facing function per file.
     This keeps things organized while keeping user application bundle sizes small.
   - It's fine to put multiple type variants of the same function in the same file.
     We export only the final overridden function node, as shown in the `space/rotate.ts` module.
2. Add appropriate tests in `test/tsl`.
    - Use `testCompute()` for pure math functions.
    - Add fragment specs as we add those TSL nodes to the project.

### Notes when porting to TSL:
- Writing TSL function nodes requires `setLayout` or the second argument to `Fn()`. Prefer using `setLayout` to be more specific.
- Since TSL nodes can be tree-shaken automatically, we don't have to conditionally load them as we do in other shader modules.
  Be aware of this while porting to TSL.


### TSL Resources

See [TSL Guide](https://threejs.org/tsl/#welcome) for details.

## How to import these TSL node in your project

```js
import { rotate } from 'lygia/tsl';
import { Fn, vec3, vec4 } from 'three/tsl';

const testing = Fn(() => {
   rotate(vec4(), vec3())
});
```
