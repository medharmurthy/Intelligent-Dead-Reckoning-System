import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import { VitePWA } from 'vite-plugin-pwa';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: ''
        },
        {
          src: 'node_modules/onnxruntime-web/dist/*.mjs',
          dest: ''
        }
      ]
    }),
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        globPatterns: ['**/*.{js,mjs,css,html,ico,png,svg,wasm,json,onnx}'],
        maximumFileSizeToCacheInBytes: 50 * 1024 * 1024 // 50MB to accommodate ONNX model and WASM
      },
      manifest: {
        name: 'Intelligent Dead Reckoning',
        short_name: 'IDR',
        description: 'Smartphone-based vehicle navigation using ONNX AI',
        theme_color: '#ffffff',
        icons: [
          {
            src: 'icon-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'icon-512x512.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      }
    })
  ]
});
