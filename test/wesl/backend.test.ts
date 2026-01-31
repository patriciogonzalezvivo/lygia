import { expect, test } from "vitest";
import { getGPUAdapter, isDeno } from "wgsl-test";

test("identify WebGPU backend", async () => {
  const adapter = await getGPUAdapter();
  const info = adapter.info;

  console.log("==> Runtime:", isDeno ? "Deno" : "Node.js");
  console.log(
    "==> Backend:",
    isDeno ? "wgpu (Deno native)" : "Dawn (webgpu npm)",
  );
  console.log("==> Vendor:", info.vendor);
  console.log("==> Architecture:", info.architecture);
  console.log("==> Device:", info.device);
  console.log("==> Description:", info.description);

  expect(info.vendor).toBeDefined();
});

test("list adapter features", async () => {
  const adapter = await getGPUAdapter();
  const features = [...adapter.features].sort();

  console.log("==> Feature count:", features.length);
  console.log("==> Features:", features.join(", "));

  expect(features.length).toBeGreaterThan(0);
});

test("check adapter limits", async () => {
  const adapter = await getGPUAdapter();
  const limits = adapter.limits;

  console.log("==> Max buffer size:", limits.maxBufferSize);
  console.log("==> Max texture dimension 2D:", limits.maxTextureDimension2D);
  console.log(
    "==> Max compute workgroup size X:",
    limits.maxComputeWorkgroupSizeX,
  );

  expect(limits.maxBufferSize).toBeGreaterThan(0);
});
