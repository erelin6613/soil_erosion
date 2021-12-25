from pathlib import Path

import cv2
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask as riomask
from shapely.geometry.collection import GeometryCollection
from torch.utils.data import Dataset


class ImageryReaderDataset(Dataset):
    def __init__(self, bands=["TCI"], preprocessing=None, transform=None):
        super().__init__()

        self.bands = bands
        self.preprocessing = preprocessing
        self.transform = transform

    def _load_geometry(self, geom_path, dst_crs):
        """Load shapes to create masks from"""

        polygons = gpd.read_file(geom_path)
        try:
            if polygons.crs != dst_crs:
                polygons = polygons.to_crs(dst_crs)
        except Exception:
            print(f"Invalid geometry captured, trying to handle: {geom_path}")
            polygons["geometry"] = polygons.geometry.apply(
                lambda x: x if x else GeometryCollection()
            )

        polygons = polygons.buffer(0)
        polygons = polygons.geometry.tolist()

        if len(polygons) < 1:
            print(f"Warning: {geom_path} as no valid geometries, crs: {polygons.crs}")

        return polygons

    def tile_image(self, image, tile_size, mask=None, out_channels=1):
        x_shape = (
            image.shape[-2] + tile_size
            if image.shape[-2] % tile_size != 0
            else image.shape[-2]
        )
        y_shape = (
            image.shape[-1] + tile_size
            if image.shape[-1] % tile_size != 0
            else image.shape[-1]
        )
        if mask:
            image_mask = np.zeros((out_channels, x_shape, y_shape))
        tiles = []

        for i in range((image.shape[0] // tile_size) + 1):
            for j in range((image.shape[-1] // tile_size) + 1):

                x1, y1 = int(i * tile_size), int(j * tile_size)
                if x1 > image.shape[-2] or y1 > image.shape[-1]:
                    break

                y2 = (
                    y1 + (image.shape[-1] - y1)
                    if image.shape[-1] - y1 < tile_size
                    else y1 + tile_size
                )
                x2 = (
                    x1 + (image.shape[-2] - x1)
                    if image.shape[-2] - x1 < tile_size
                    else x1 + tile_size
                )
                tile = image[:, x1:x2, y1:y2]

                if tile.shape != (image.shape[0], tile_size, tile_size):
                    empty_mask = np.zeros((image.shape[0], tile_size, tile_size))
                    empty_mask[:, : tile.shape[1], : tile.shape[2]] = tile
                    image_tensor = empty_mask
                else:
                    image_tensor = tile

                if mask:
                    image_mask = mask[:, x1 : x1 + tile_size, y1 : y1 + tile_size]
                    tiles.append(image_tensor, image_mask)
                else:
                    tiles.append(image_tensor, None)

        return tiles

    def _extract_s2_tilename(self, filename):
        # pattern =
        pass

    def _write_tiles(self, raster_dir, aoi_file, tiles_dir, masks_dir, mask_aoi=None):

        files = []
        for band in self.bands:
            files.extend(
                [f for f in r_dir.iterdir() if f"_{band}_" in f and ".jp2" in f]
            )
        # return files

        stacks = []
        for file in sorted(files):
            with rasterio.open(file) as dataset:
                # try:
                region, region_tfs = riomask.mask(
                    dataset, aoi.geometry, all_touched=False, crop=True
                )
                # except Exception:
                #     continue
                stacks.append(region)
            if mask_aoi is not None:
                mask_aoi = rasterio.features.rasterize(
                    shapes=mask_aoi.geometry.tolist(),
                    out_shape=(region.shape[-2], region.shape[-1]),
                    transform=region_tfs,
                    default_value=255,
                )

        composit_img = np.stack()
        tiles = self.tile_image(composit_img, self.tile_size, mask_aoi)

        for i, (image, mask) in enumerate(tiles):
            tile_fname = raster_dir



if __name__ == "__main__":
    pass
    # d = ImageryReaderDataset()
    # d._write_tiles("")
