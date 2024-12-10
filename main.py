import base64
import time
from io import BytesIO

import torch
import requests
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from spandrel import ModelLoader
from tqdm import tqdm


class ModelService:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def load_model(self):
        self.model = ModelLoader().load_from_file('./models/4x-ClearRealityV1.pth')
        self.model.to(self.device)
        self.model.eval()
        print("load model succ")

    def process_image(self, img_tensor):
        if not self.model:
            raise HTTPException(status_code=500, detail="model not loaded")

        tile = 512
        overlap = 32

        while tile >= 128:
            try:
                with torch.no_grad():
                    total_tiles = int(np.ceil(img_tensor.shape[2] / (tile - overlap))) * \
                        int(np.ceil(img_tensor.shape[3] / (tile - overlap)))

                    start_time = time.time()

                    pbar = tqdm(total=total_tiles, desc="upscaling")

                    output = self._tiled_scale(
                        img_tensor.to(self.device),
                        tile_x=tile,
                        tile_y=tile,
                        overlap=overlap,
                        pbar=pbar
                    )
                    pbar.close()

                    return output

            except RuntimeError as e:
                if 'out of memory' not in str(e):
                    raise e
                print(f"short of memory, lower til size: {tile//2}x{tile//2}")
                tile //= 2
                torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500, detail="image too large to process")

    def _tiled_scale(self, img, tile_x=512, tile_y=512, overlap=32, upscale_amount=4, pbar=None):
        batch, channels, height, width = img.shape
        output = torch.zeros(
            (batch, channels, height * upscale_amount, width * upscale_amount), device='cpu')

        x_tiles = int(np.ceil(width / (tile_x - overlap)))
        y_tiles = int(np.ceil(height / (tile_y - overlap)))

        for y in range(y_tiles):
            for x in range(x_tiles):
                start_y = y * (tile_y - overlap)
                end_y = min(start_y + tile_y, height)
                start_x = x * (tile_x - overlap)
                end_x = min(start_x + tile_x, width)

                tile = img[:, :, start_y:end_y, start_x:end_x]
                scaled_tile = self.model(tile)

                out_start_y = int(start_y * upscale_amount)
                out_end_y = int(end_y * upscale_amount)
                out_start_x = int(start_x * upscale_amount)
                out_end_x = int(end_x * upscale_amount)

                output[:, :, out_start_y:out_end_y,
                       out_start_x:out_end_x] = scaled_tile.cpu()
                if pbar:
                    pbar.update(1)

        return output


def prepare_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return transforms.ToTensor()(img).unsqueeze(0)
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"failed to process image: {str(e)}")


def image_to_base64(tensor):
    try:
        img = transforms.ToPILImage()(tensor.squeeze(0).cpu().detach().clamp(0, 1))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"failed to encode image base64: {str(e)}")


app = FastAPI()
model_service = ModelService()


@app.on_event("startup")
async def startup_event():
    model_service.load_model()


class ImageRequest(BaseModel):
    url: str


@app.post("/process_image")
async def process_image(request: ImageRequest):
    start_time = time.time()
    img_tensor = prepare_image(request.url)
    output = model_service.process_image(img_tensor)

    output_pil = transforms.ToPILImage()(output.squeeze(0).cpu().detach().clamp(0, 1))
    buffer = BytesIO()
    output_pil.save(buffer, format="PNG")
    buffer.seek(0)

    process_time = time.time() - start_time
    headers = {
        "Content-Disposition": "attachment; filename=output.png",
        "X-Process-Time": f"{process_time:.2f}s"
    }

    # file response
    return StreamingResponse(
        buffer,
        media_type='image/png',
        headers=headers,
    )


@app.get("/health")
async def health_check():
    return "ok"
