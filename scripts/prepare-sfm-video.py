"""Decode a local video into a bounded, lossless frame bundle for the WebGPU SfM lab.

Requires opencv-python-headless. Native decoding also accepts source dimensions
above browser video/texture limits; only resized analysis frames enter WebGPU.
"""
import argparse
import base64
import json
import math
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--max-edge', type=int, default=512)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float)
    args = parser.parse_args()
    if not 64 <= args.max_edge <= 4096 or not 2 <= args.frames <= 256:
        parser.error('max-edge must be 64..4096 and frames must be 2..256')
    if not math.isfinite(args.start) or args.start < 0:
        parser.error('start must be a finite, nonnegative timestamp')
    if args.output.exists():
        parser.error('output already exists; choose a new output path')
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        parser.error('native decoder cannot open the video or its codec')
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 2:
            parser.error('video must expose a finite frame rate and frame count')
        duration = count / fps
        last_time = (count - 1) / fps
        end = last_time if args.end is None else args.end
        if not math.isfinite(end) or not args.start < end <= last_time:
            parser.error(f'timestamps must satisfy 0 <= start < end <= {last_time:g}')
        frames = []
        source = None
        for index in range(args.frames):
            time = args.start + (end - args.start) * index / (args.frames - 1)
            # Keep one decoded source frame at a time, independent of video length.
            if not capture.set(cv2.CAP_PROP_POS_MSEC, time * 1000):
                raise ValueError(f'decoder cannot seek to {time:g}s')
            ok, image = capture.read()
            if not ok:
                raise ValueError(f'decoder cannot read frame at {time:g}s')
            height, width = image.shape[:2]
            if source is None:
                source = {'name': args.video.name, 'width': width, 'height': height,
                          'duration': duration, 'fps': fps}
            elif width != source['width'] or height != source['height']:
                raise ValueError('changing source dimensions are not supported')
            scale = min(1, args.max_edge / max(width, height))
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            if size[0] * size[1] * args.frames > 64 * 1024 * 1024:
                parser.error('analysis frames exceed the 64-megapixel bundle budget')
            resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            del image
            ok, encoded = cv2.imencode('.png', resized)
            if not ok:
                raise ValueError('PNG encoding failed')
            frames.append({'id': index, 'time': time, 'width': size[0], 'height': size[1],
                           'image': 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')})
            print(f'{index + 1}/{args.frames}: {width}x{height} -> {size[0]}x{size[1]} at {time:g}s')
        bundle = {'format': 'typecv-sfm-video-v1', 'source': source, 'frames': frames}
        text = json.dumps(bundle, separators=(',', ':')) + '\n'
        if len(text) > 64 * 1024 * 1024:
            raise ValueError('encoded frame bundle exceeds 64 MiB; reduce max-edge or frames')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
        print(f'Saved {args.output} ({len(text)} bytes). Open this JSON in the SfM lab.')
    finally:
        capture.release()


if __name__ == '__main__':
    main()
