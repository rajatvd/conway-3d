"""Small orthographic, z-buffered mesh previews; no display server needed."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (17, 25, 32)


def checked_jpeg(path):
    """Inspect headers and size before an image is viewed or assembled."""
    path = Path(path)
    with Image.open(path) as image:
        assert image.format == "JPEG", path
        assert max(image.size) <= 640, path
        assert path.stat().st_size <= 180 * 1024, path
        return {
            "format": image.format,
            "pixels": list(image.size),
            "bytes": path.stat().st_size,
        }


def save_jpeg(image, path):
    image.save(path, format="JPEG", quality=91, optimize=True)
    return checked_jpeg(path)


def render(parts, colors, path, title, subtitle, font_path, azimuth=-48, elevation=24):
    """Rasterize the reloaded, registered color STLs, with real depth testing.

    Only the bounded delivery JPEG is saved. Supersampling stays in memory.
    """
    width, height, ss = 640, 640, 2
    w, h = width * ss, height * ss
    vertices = np.vstack([m.vertices for m in parts])
    az, el = np.radians([azimuth, elevation])
    direction = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0])
    up = np.cross(direction, right)
    basis = np.stack([right, up, direction], axis=1)
    projected = vertices @ basis
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    center = (lo + hi) / 2
    factor = min((w - 92 * ss) / (hi[0] - lo[0]), (h - 150 * ss) / (hi[1] - lo[1]))
    depth = np.full((h, w), -np.inf, dtype=np.float32)
    pixels = np.full((h, w, 3), BACKGROUND, dtype=np.uint8)
    light = np.array([-0.45, -0.65, 1.0])
    light /= np.linalg.norm(light)
    for mesh, color in zip(parts, colors):
        rgb = np.array(tuple(bytes.fromhex(color.lstrip("#"))), dtype=float)
        pts = mesh.vertices @ basis
        pts -= center
        pts[:, 0] = pts[:, 0] * factor + w / 2
        pts[:, 1] = -pts[:, 1] * factor + (h + 20 * ss) / 2
        for face, normal in zip(mesh.faces, mesh.face_normals):
            if normal @ direction <= 1e-8:
                continue
            triangle = pts[face]
            x0, y0 = np.maximum(np.floor(triangle[:, :2].min(axis=0)), [0, 0]).astype(
                int
            )
            x1, y1 = np.minimum(
                np.ceil(triangle[:, :2].max(axis=0)), [w - 1, h - 1]
            ).astype(int)
            if x1 < x0 or y1 < y0:
                continue
            a, b, c = triangle
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denominator) < 1e-9:
                continue
            yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
            xx = xx + 0.5
            yy = yy + 0.5
            wa = (
                (b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])
            ) / denominator
            wb = (
                (c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])
            ) / denominator
            wc = 1 - wa - wb
            z = wa * a[2] + wb * b[2] + wc * c[2]
            window = depth[y0 : y1 + 1, x0 : x1 + 1]
            visible = (wa >= -1e-7) & (wb >= -1e-7) & (wc >= -1e-7) & (z > window)
            window[visible] = z[visible]
            shade = 0.48 + 0.52 * max(0, normal @ light)
            pixels[y0 : y1 + 1, x0 : x1 + 1][visible] = np.clip(
                rgb * shade, 0, 255
            ).astype(np.uint8)
    image = Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 22)
    small = ImageFont.truetype(str(font_path), 13)
    draw.text((26, 22), title, font=font, fill=(230, 234, 231))
    draw.text((27, 52), subtitle, font=small, fill=(144, 161, 166))
    for i, color in enumerate(colors):
        draw.rounded_rectangle(
            (27 + i * 23, 605, 42 + i * 23, 610), radius=2, fill=color
        )
    draw.text((110, 599), "TIME UP / mm", font=small, fill=(144, 161, 166))
    return save_jpeg(image, path)


def overview(paths, path, titles, font_path):
    image = Image.new("RGB", (639, 320), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 13)
    small = ImageFont.truetype(str(font_path), 10)
    for i, (source, title) in enumerate(zip(paths, titles)):
        checked_jpeg(source)
        with Image.open(source) as detail:
            # The mesh occupies the central workspace, not the label plane.
            crop = detail.crop((20, 90, 620, 590)).resize(
                (211, 202), Image.Resampling.LANCZOS
            )
            image.paste(crop, (i * 213 + 1, 50))
        draw.text((i * 213 + 12, 17), title, font=font, fill=(230, 234, 231))
        draw.text(
            (i * 213 + 12, 275),
            "EVERGREEN / AMBER / CORAL",
            font=small,
            fill=(144, 161, 166),
        )
        if i:
            draw.line((i * 213, 12, i * 213, 307), fill=(45, 58, 65))
    return save_jpeg(image, path)
