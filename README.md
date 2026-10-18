# photo-overview

A single-file Python web app that lists a folder of AI-generated images in a
table, together with the generation info ComfyUI embedded in each file. No
dependencies beyond the Python standard library.

| Preview | Model | LoRAs | Prompt |
|---|---|---|---|
| thumbnail, file name, date, size | checkpoint / UNet | LoRA names + strengths | positive prompt, negative prompt, seed / steps / cfg / sampler |

## Usage

Run it from inside the folder of images (or point it at one with `--folder`):

```bash
python3 photo_overview.py
```

Then open the printed URL (default `http://localhost:8000/`).

```
--port PORT      port to listen on (default 8000)
--host HOST      host to bind (default 0.0.0.0)
--folder PATH    folder of images to serve (default: current folder)
-r, --recursive  include images in subfolders
--open           open the overview in your default browser
```

Try it with the bundled samples:

```bash
python3 photo_overview.py --folder sample_images --open
```

## Features

- Filter box matches file name, model, LoRA and prompt text (all words must match).
- Sort by name, newest, oldest or model; small / medium / large previews.
- Click a preview for a full-size view (click or Escape to close).
- "copy" button on each prompt; toggle negative prompts on/off.
- "Reload" rescans the folder - new images show up without restarting.
  Metadata is cached per file (by mtime/size), so rescans are cheap.

## Where the info comes from

- **ComfyUI PNGs**: the `prompt` text chunk (API-format graph). Models come
  from loader nodes (`ckpt_name`, `unet_name`, ...), LoRAs from `LoraLoader`
  / `LoraLoaderModelOnly` and rgthree's Power Lora Loader, prompts by
  following each sampler's `positive` / `negative` (or a Flux guider's
  `conditioning`) back to the text, including through string primitives and
  conditioning pass-through nodes.
- If only the `workflow` chunk is present, it's read as a fallback.
- Automatic1111 / Forge `parameters` text is also understood.
- WebP / JPEG with ComfyUI's `prompt:{...}` EXIF entries: best effort.
- Images without any of this still appear, marked "No generation info found".
