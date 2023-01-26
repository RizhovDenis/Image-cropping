import os
import csv
import cv2
import numpy as np

from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Process
from pathlib import Path
from typing import List
from threading import Thread


LARGE_DIR = Path("large")
MEDIUM_DIR = Path("medium")
SMALL_DIR = Path("small")
META_DIR = Path("meta")
FULL_DIR = Path("full")


class Tiler:

    def __init__(self, tile_width: int, tile_height: int, out_dir: Path, images: List[Path]):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.out_dir = out_dir
        self.images = images
        self._check_exists_dir()

    def iter_split(self, image: np.array):
        image = image.swapaxes(0, -1)[None]  # HxWxC -> 1xCxWxH

        x_lefts = [x for x in range(
            0, image.shape[3] - self.tile_width + 1, self.tile_width)]
        y_tops = [y for y in range(
            0, image.shape[2] - self.tile_height + 1, self.tile_height)]
        img_tiles = []
        for x_left in x_lefts:
            x_right = x_left + self.tile_width
            for y_top in y_tops:
                y_bottom = y_top + self.tile_height

                tile = image[
                    ...,
                    y_top:y_bottom,  # height
                    x_left:x_right,  # width
                ]

                tile = tile[0].swapaxes(0, -1)  # 1xCxWxH -> HxWxC
                img_tiles.append(tile)
        return img_tiles

    def save_img_tiles(self, img_tiles: List[np.array], img: np.array):
        for idx, img_tile in enumerate(img_tiles):
            img_output_name = self.out_dir / f"{img.stem}_{idx}.jpg"
            cv2.imwrite(str(img_output_name), img_tile)

    def _check_exists_dir(self):
        if not os.path.exists(self.out_dir):
            os.mkdir(path=self.out_dir)


def check_tile_size(tile_size: str, images: List[Path]) -> Tiler:
    if args.debug:
        assert tile_size in {
            "large", "medium", "small"}, "Unknown cutting format. Put one from: large/medium/small"

    size_dictionary = {
        "large": Tiler(500, 500, LARGE_DIR, images),
        "medium": Tiler(300, 500, MEDIUM_DIR, images),
        "small": Tiler(50, 50, SMALL_DIR, images)
    }

    return size_dictionary[tile_size]


def write_meta(time_start: datetime, time_end: datetime, tile_size: str, num_inp_imgs: int, mode: str):
    meta_data = {
        "number_input_imgs": num_inp_imgs,
        "duration": time_end - time_start,
        "tile_size": tile_size,
        "mode": mode
    }
    fieldnames = [key for key, value in meta_data.items()]

    with open(os.path.join(META_DIR, "meta.csv"), "a", encoding="utf-8") as meta_file:
        writer = csv.DictWriter(meta_file, fieldnames=fieldnames)
        writer.writerow(meta_data)


class Mode:

    def __init__(self, mode: str, images: List[Path], tiler: Tiler):
        self.mode = mode
        self.images = images
        self.tiler = tiler
        self.num_workers = os.cpu_count()
        self.batch_size = len(images) // self.num_workers

    def target(self, batch: List[Path], tiler: Tiler):
        for img in batch:
            np_img = cv2.imread(str(img))
            img_tiles = tiler.iter_split(np_img)
            tiler.save_img_tiles(img_tiles, img)

    def iter_loops(self):
        raise NotImplemented


class Simple(Mode):

    def iter_loops(self):
        self.target(self.images, self.tiler)


class Multiproccesing(Mode):
    proccesses = []

    def iter_loops(self):

        for idx in range(self.num_workers):
            batch = self.images[self.batch_size *
                                idx: self.batch_size * (idx+1)]
            self.proccesses.append(Process(target=self.target, args=(
                batch, self.tiler), name=f"Proccess: {idx+1}"))
        for proccess in self.proccesses:
            proccess.start()

        for proccess in self.proccesses:
            proccess.join()


class Multithreading(Mode):
    threads = []

    def iter_loops(self):
        for idx in range(self.num_workers):
            batch = self.images[self.batch_size *
                                idx: self.batch_size * (idx+1)]
            self.threads.append(Thread(target=self.target, args=(
                batch, self.tiler), name=f"Thread: {idx+1}"))
        for tr in self.threads:
            tr.start()

        for tr in self.threads:
            tr.join()


def check_mode(mode: str, images: List[Path], tiler: Tiler) -> Mode:
    if args.debug:
        assert mode in {"simple", "multiproccessing",
                        "multithreading"}, "Unknown cutting mode."

    mode_dictionary = {
        "simple": Simple(mode, images, tiler),
        "multiproccessing": Multiproccesing(mode, images, tiler),
        "multithreading": Multithreading(mode, images, tiler)
    }

    return mode_dictionary[mode]


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--tile_size", default="large",
                            type=str, help="large/medium/small")
    arg_parser.add_argument("--mode", default="simple",
                            type=str, help="simple/multiproccessing/multithreading")
    arg_parser.add_argument("--debug", type=bool,
                            default=False, help="debug mode")
    args = arg_parser.parse_args()

    time_start = datetime.utcnow()
    images = sorted(FULL_DIR.glob("*.jpg"))

    print(f"Found {len(images)} images.")
    tiler = check_tile_size(args.tile_size, images)
    workers = check_mode(args.mode, images, tiler)
    workers.iter_loops()
    time_end = datetime.utcnow()

    write_meta(time_start, time_end, args.tile_size,
               len(images), args.mode)
