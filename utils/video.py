import os
import os.path as osp
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Iterable, Union, Optional
import cv2
import av


def mkdirs(path: Union[str, Path]):
    if isinstance(path, str):
        os.makedirs(osp.split(video_path)[0], exist_ok=True)
    else:
        path.parent.mkdir(exist_ok=True)


def dump_vid(video_path: Union[str, Path], frames: Iterable[Image.Image], fps: int = 25,):
    import imageio
    imageio.mimsave(video_path, frames, format='.mp4', quality=8, fps=fps, macro_block_size=1)


class VideoWriterPyAV:
    def __init__(
        self,
        video_path: Union[str, Path],
        fps: int = 25,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        kbitrate: int = 5000,
        codec: str = 'h264',
        pix_fmt: str = 'yuv420p',
        transparency: bool = False,
    ):
        if transparency:
            codec = 'prores_ks'
            pix_fmt = 'yuva444p10le'
            video_path = osp.splitext(video_path)[0] + '.mov'
        
        container = av.open(str(video_path), 'w')
        stream = container.add_stream(codec, rate=fps)
        stream.pix_fmt = pix_fmt
        if kbitrate is not None:
            stream.bit_rate = kbitrate * 1024 # bitrate in kbps
        if image_width is not None and image_height is not None:
            stream.height = image_height
            stream.width = image_width
        self.container = container
        self.stream = stream
        self.writing = False
        self.transparency = transparency
    
    def write(self, image: Union[Image.Image, np.ndarray]):
        
        if isinstance(image, Image.Image):
            if self.transparency:
                image_array = np.array(image.convert('RGBA'))
            else:
                image_array = np.array(image.convert('RGB'))
        else:
            image_array = image
        
        if not self.writing:
            self.writing = True
            self.stream.height = image_array.shape[0]
            self.stream.width = image_array.shape[1]
        
        frame = av.VideoFrame.from_ndarray(image_array, format='rgb24')
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def release(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.writing = False


class VideoWriterOpenCV:

    FOURCC_MAPPING = {
        'mp4v': 'mp4v',
        'ffv1': 'FFV1',
        'png ': 'png ',
        'avc1': 'avc1',
        'h264': 'h264',
    }

    def __init__(
        self,
        video_path: Union[str, Path],
        width: int = None,
        height: int = None,
        fps: int = 25,
        fourcc: str = 'ffv1',
    ):
        self.fps = fps
        fourcc = self.FOURCC_MAPPING[fourcc.lower()]
        video_path = str(video_path)
        if fourcc.lower() in ('ffv1', 'png '):
            video_path = osp.splitext(video_path)[0] + '.avi'
        self.fourcc = int(cv2.VideoWriter_fourcc(*fourcc))
        self.video_path = video_path
        self.video_writer = None  # lazy create
        if width is None or height is None:
            self.size = None
        else:
            self.size = (width, height)
    
    def write(self, image: Image.Image):
        if self.size is None:
            self.size = image.size
        if self.video_writer is None:
            self.video_writer = cv2.VideoWriter(
                filename=self.video_path,
                fourcc=self.fourcc,
                fps=self.fps,
                frameSize=self.size,
                isColor=True,
            )
        image_array = np.array(image.convert('RGB'))[:, :, ::-1]
        self.video_writer.write(image_array)

    def release(self):
        if self.video_writer is not None:
            self.video_writer.release()
        self.size = None


class VideoWriterPIL:
    def __init__(
        self,
        video_path: Union[str, Path],
        fps: int = 25,
    ):
        video_path = str(video_path)
        assert video_path.endswith('.gif')
        self.video_path = video_path
        self.fps = fps
        self.frames = []
    
    def write(self, image: Image.Image):
        self.frames.append(image)
    
    def release(self):
        if len(self.frames) == 0:
            return
        if self.frames[0].mode == 'RGBA':
            disposal = 2
        else:
            disposal = 0
        self.frames[0].save(
            self.video_path,
            format="GIF",
            save_all=True,
            loop=0,
            append_images=self.frames,
            duration=1000 / self.fps,
            disposal=disposal,
        )
        self.frames = []


if __name__ == '__main__':
    
    # Solution 0
    video_path = f'./pyav_h264.mp4'
    video_writer = VideoWriterPyAV(video_path)
    for i in range(25):
        img = Image.fromarray((np.random.rand(512, 512, 3) * 255.0).astype(np.uint8), mode='RGB')
        video_writer.write(img)
    video_writer.release()

    # Solution 1
    # for fourcc in ('ffv1', 'mp4v', 'png ', 'avc1', ):
    # for fourcc in ('avc1', ):
    #     video_path = f'./cv2_{fourcc}.mp4'
    #     video_writer = VideoWriterOpenCV(video_path, fourcc=fourcc)

    #     for i in range(25):
    #         img = Image.fromarray((np.random.rand(512, 512, 3) * 255.0).astype(np.uint8), mode='RGB')
    #         video_writer.write(img)

    #     video_writer.release()

    # Solution 2
    # imgs = []
    # for i in range(50):
    #     img = Image.fromarray((np.random.rand(512, 512, 3) * 255.0).astype(np.uint8), mode='RGB')
    #     imgs.append(img)
    # dump_vid(video_path=video_path, frames=imgs, fps=25)