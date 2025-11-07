# ASCII Real‑Time Camera (MediaPipe)

<p align="center">
  <a href="https://img.shields.io/badge/JavaScript-Vanilla%20JS-F7DF1E?logo=javascript&logoColor=000&labelColor=F7DF1E&color=000"> 
    <img alt="Vanilla JS" src="https://img.shields.io/badge/JavaScript-Vanilla%20JS-F7DF1E?logo=javascript&logoColor=000&labelColor=F7DF1E&color=000" />
  </a>
  <a href="https://img.shields.io/badge/Tailwind-C%20DN-38BDF8?logo=tailwindcss&logoColor=fff&labelColor=06B6D4&color=0B1220">
    <img alt="Tailwind CDN" src="https://img.shields.io/badge/Tailwind-C%20DN-38BDF8?logo=tailwindcss&logoColor=fff&labelColor=06B6D4&color=0B1220" />
  </a>
  <a href="https://img.shields.io/badge/MediaPipe-Selfie%20Segmentation-FF6F00?logo=google&logoColor=fff&labelColor=EA4335&color=0B1220">
    <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-Selfie%20Segmentation-FF6F00?logo=google&logoColor=fff&labelColor=EA4335&color=0B1220" />
  </a>
  <a href="#deploy">
    <img alt="Live demo" src="https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-2ea44f?logo=github" />
  </a>
</p>

Turn your webcam into living ASCII and Emoji art — with optional background removal, smooth controls, and a glassmorphism UI. Built with vanilla JS, Tailwind (CDN), and MediaPipe Selfie Segmentation.

<p align="center">
  <img alt="ASCII camera preview placeholder" src="https://user-images.githubusercontent.com/0000000/placeholder-ascii-preview.gif" width="720" />
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#usage">Usage</a> •
  <a href="#how-it-works">How it works</a> •
  <a href="#performance-tips">Performance</a> •
  <a href="#deploy">Deploy</a>
</p>

---

## Features

- Real‑time webcam → ASCII renderer (selfie‑mirrored)
- Multiple palettes: Basic, Extended, Blocks, Emoji, plus a fully Custom palette
- Optional background removal using MediaPipe Selfie Segmentation
- Dark/Light theme toggle with polished glass UI and responsive layout
- Resolution slider with smooth transition for grid density
- Emoji mode rendered via canvas with per‑glyph caching for alignment and speed
- Crisp pixel aesthetics (image smoothing disabled everywhere)
- Zero build tools — open locally or serve as static files

## Quick start

This app uses the camera via `getUserMedia`, which requires a secure origin. Use one of these local options:

- Easiest: VS Code Live Server extension (right‑click `index.html` → "Open with Live Server").
- Python 3 (built‑in on many systems):

```powershell
# Optional local server on Windows PowerShell
python -m http.server 5173
# Then open http://localhost:5173
```

- Node.js (if installed):

```powershell
npx serve .
# or
npx http-server -p 5173
```

Then visit `http://localhost:<port>`, allow camera access, and you’re in.

## Usage

Top control bar (mobile‑friendly):

- Palette: choose between Basic, Extended, Blocks, Emoji, or Custom
  - Custom prompts you to enter characters ordered from light → dark
- Remove BG: toggles MediaPipe segmentation to keep only the foreground
- Res: controls grid density; higher = more characters, more detail
- Theme: toggles Dark/Light mode (UI adapts)

Display modes:

- ASCII text mode uses a `<pre>` element, dynamically sizing fonts to fill the viewport
- Emoji mode uses a `<canvas>` grid to keep cells perfectly aligned and square

Footer contains creator links. UI uses a subtle animated gradient background with glassmorphism panels.

## How it works

- Camera frames are sampled to an offscreen canvas at a dynamic grid size
- Per‑cell luminance is computed using perceptual luma: `0.2126R + 0.7152G + 0.0722B`
- Each cell maps brightness → a character from the active palette (light → dark)
- Background removal (optional) leverages MediaPipe Selfie Segmentation; mask is applied per cell
- Text mode updates a `<pre>` string; Emoji mode draws cached glyph bitmaps into a canvas
- Scale changes (resolution slider) are eased for smooth visual transitions
- All image smoothing is disabled to keep edges sharp and pixel‑crispy

### Tech stack

- Vanilla JavaScript (no framework)
- Tailwind CSS via CDN
- MediaPipe Selfie Segmentation via CDN

## Performance tips

- Emoji mode is heavier than plain text — reduce resolution on low‑power devices
- Keep the browser tab focused for a stable frame rate
- On laptops, plug in for better sustained performance

## Deploy

Static hosting works anywhere (GitHub Pages, Netlify, Vercel, etc.). For GitHub Pages:

1. Push this repository to GitHub
2. In Settings → Pages, choose the `main` branch and `/ (root)`
3. Save — your site will be live at `https://<user>.github.io/<repo>`

## Privacy & security

- All processing runs in your browser; video never leaves your device
- The page requests camera permission the first time you visit
- External libraries are loaded from trusted CDNs

## Roadmap ideas

- Save/share snapshots and short GIF captures
- Per‑palette brightness curves and dithering options
- Mobile optimizations and touch gestures
- Additional segmentation models and effects

## Acknowledgements

- MediaPipe Selfie Segmentation — real‑time background masks
- Tailwind CSS — utility styling
- Emoji rendering relies on system emoji fonts

## License

No license specified yet. Consider adding a LICENSE file (MIT is a popular choice for open web demos).

---

Made with ❤️ by [Vasu‑Devs](https://github.com/vasu-devs)

