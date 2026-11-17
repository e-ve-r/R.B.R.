import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

let backendAddress = 'http://127.0.0.1:8000';
if (process.env.API_PROXY_TARGET) {
    backendAddress = process.env.API_PROXY_TARGET;
}

export default defineConfig({
    plugins: [react()],
    server: {
        host: '127.0.0.1',
        proxy: {
            '/api': {
                target: backendAddress,
                changeOrigin: true,
            },
        },
    },
});
