const http = require('node:http');

const apiUrl = process.env.RAGBENCH_API_URL || 'http://localhost:8000/health';
const webUrl = process.env.RAGBENCH_WEB_URL || 'http://localhost:5173';

function smoke() {
  http.get(apiUrl, (res) => {
    if (res.statusCode >= 200 && res.statusCode < 500) {
      console.log(`desktop smoke ok: api=${apiUrl} web=${webUrl}`);
      process.exit(0);
    }
    console.error(`desktop smoke failed: ${res.statusCode}`);
    process.exit(1);
  }).on('error', (error) => {
    console.error(`desktop smoke failed: ${error.message}`);
    process.exit(1);
  });
}

if (process.argv.includes('--smoke')) {
  smoke();
} else {
  const { app, BrowserWindow } = require('electron');
  app.whenReady().then(() => {
    const win = new BrowserWindow({ width: 1280, height: 900 });
    win.loadURL(webUrl);
  });
}
