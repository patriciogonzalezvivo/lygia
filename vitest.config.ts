import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
   reporters: [
      'default',
      'vitest-image-snapshot/reporter'
   ]
  },
})
