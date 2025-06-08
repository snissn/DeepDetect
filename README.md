# Photo Forensics App

This Next.js application lets you analyze photos directly on your device for signs of editing. The analysis runs entirely in your browser—no images are uploaded anywhere.

## Features

- **Local processing** – images never leave your phone or computer.
- **Highlight suspicious regions** – uses variance, JPEG grid artifacts, PRNU noise, and error level analysis.
- **Responsive design** – works on mobile and desktop.

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the development server:
   ```bash
   npm run dev
   ```
3. Open [http://localhost:3000](http://localhost:3000) to view the app.

## Building for Production

```
npm run build
npm start
```

## Usage

1. Open the app in your browser or on your phone.
2. Tap **Select Image** and choose a photo from your device.
3. The app overlays red boxes on regions that may have been altered.

All analysis happens locally and no data is sent to any server.

## License

MIT
