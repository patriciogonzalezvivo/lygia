import { linkBuildExtension } from "wesl-plugin";
import type { ViteUserConfig } from "vitest/config";
import viteWesl from "wesl-plugin/vite";

const config = {
  plugins: [
    viteWesl({
      extensions: [linkBuildExtension],
    }) as any, // vite plugin types change frequently and harmlessly.
  ],
} satisfies ViteUserConfig;

export default config;
