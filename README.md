# Dot Portrait

Upload portraits and get dot-grid halftones (head-and-shoulders crop, background removed, fading bottom edge).
Everything runs in the browser. Photos are never uploaded anywhere. MediaPipe models load from a CDN on first visit.

**Live:** https://sky-seo.github.io/infinite-garden-portraint/

## Run locally

```
python3 -m http.server 8000
```

Then open http://localhost:8000. It has to be served over http(s) because ES modules don't load from `file://`.

Deployed with GitHub Pages from the `main` branch root. Pushing to `main` redeploys within about a minute. There is no build step.

## How it works

1. **Analyze**: MediaPipe selfie segmentation removes the background and face detection finds the face.
2. **Frame**: a square crop is computed from the top of the head and the face, with the head at about 55% of the frame height.
3. **Sample**: the crop is downsampled to the grid, auto-levelled on the subject, and local contrast is boosted so facial features survive.
4. **Quantize**: each cell is mapped to N grey levels with ordered (Bayer) dithering.
5. **Fade**: the bottom rows are thinned out with a Bayer pattern.

On each card: drag to pan, scroll to zoom, double-click to reset the crop.
