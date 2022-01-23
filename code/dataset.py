import re
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask as riomask
from shapely.geometry.collection import GeometryCollection
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# from utils import crop_raster


class ImageryReaderDataset(Dataset):
    def __init__(
        self,
        aoi_index,
        input_folder,
        out_folder=None,
        bands=["TCI", "B04"],
        tile_size=256,
        preprocessing=None,
        transform=None,
    ):
        super().__init__()

        self.input_folder = Path(input_folder)
        if out_folder is None:
            self.out_folder = Path(f"{self.input_folder}_dataset")
        self.aoi_index = aoi_index

        self.bands = bands
        self.tile_size = tile_size
        self.preprocessing = preprocessing
        self.transform = transform

        #     # self.process_folder()

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
        if mask is not None:
            image_mask = np.zeros((out_channels, x_shape, y_shape))
        tiles = []

        for i in range((image.shape[-1] // tile_size) + 1):
            for j in range((image.shape[-2] // tile_size) + 1):

                x1, y1 = int(i * tile_size), int(j * tile_size)
                if x1 > image.shape[-1] or y1 > image.shape[-2]:
                    break

                y2 = (
                    y1 + (image.shape[-2] - y1)
                    if image.shape[-2] - y1 < tile_size
                    else y1 + tile_size
                )
                x2 = (
                    x1 + (image.shape[-1] - x1)
                    if image.shape[-1] - x1 < tile_size
                    else x1 + tile_size
                )
                tile = image[:, x1:x2, y1:y2]
                if tile.shape[-1] == 0 or tile.shape[-2] == 0:
                    continue

                if tile.shape != (image.shape[0], tile_size, tile_size):
                    empty_image = np.zeros((image.shape[0], tile_size, tile_size))
                    empty_image[:, : tile.shape[-2], : tile.shape[-1]] = tile
                    image_tensor = empty_image

                else:
                    image_tensor = tile

                if mask is not None:
                    # print(mask)
                    if tile.shape != (image.shape[0], tile_size, tile_size):
                        empty_mask = np.zeros((tile_size, tile_size))
                        empty_mask[: tile.shape[-2], : tile.shape[-1]] = mask[
                            x1 : x1 + tile.shape[-2], y1 : y1 + tile.shape[-1]]
                        image_mask = empty_mask
                    else:
                        image_mask = mask[x1 : x1 + tile_size, y1 : y1 + tile_size]
                    tiles.append((image_tensor, image_mask))
                else:
                    tiles.append((image_tensor, None))

        return tiles

    def _extract_s2_tilename(self, filename):
        pattern = r"T[0-9]{2}[A-Z]{3}"
        # print("*", pattern, filename)
        tilename = re.search(pattern, str(filename))
        if tilename is not None:
            return tilename.group(0)
        return None

    def _extract_s2_date(self, filename):
        pattern = r"_[0-9]{8}T"
        # print("*", pattern, filename)
        tilename = re.search(pattern, str(filename))
        if tilename is not None:
            date = tilename.group(0)
            return f"{date[:5]}_{date[5:7]}_{date[7:9]}"
        return None

    def prepare_data(self):
        out_folder = Path(self.input_folder).name + "_dataset"
        out_folder = Path(self.out_folder)

        if out_folder.exists():
            print(f"Dataset {str(self.out_folder)} has been created before")
            return

        out_folder.mkdir(parents=True)

        if isinstance(self.aoi_index, list):
            files = list(Path(self.input_folder).iterdir())
            for folder in tqdm(files):
                if not folder.is_dir():
                    continue

                for aoi in self.aoi_index:
                    try:
                        annot_path = Path(aoi.replace("_aoi.geojson", ".geojson"))
                        annot_geoms = (
                            None
                            if not annot_path.exists()
                            else gpd.read_file(annot_path)
                        )
                        self._write_tiles(
                            folder,
                            aoi,
                            out_folder.joinpath("tiles"),
                            out_folder.joinpath("masks"),
                            annot_geoms,
                        )
                    except Exception: # as e:
                        # print(e)
                        continue

        elif isinstance(self.aoi_index, dict):
            for aoi in tqdm(self.aoi_index.keys(), total=len(aoi_index)):

                self._write_tiles(
                    self.aoi_index[aoi],
                    aoi,
                    out_folder.joinpath("tiles"),
                    out_folder.joinpath("masks"),
                )

    def _write_tiles(self, raster_dir, aoi_file, tiles_dir, masks_dir, mask_aoi=None):

        aoi = gpd.read_file(aoi_file)
        files = []
        for band in self.bands:
            files.extend(
                [
                    f
                    for f in Path(raster_dir).iterdir()
                    if f"_{band}_" in str(f) and ".jp2" in str(f)
                ]
            )

        stacks = []
        for file in sorted(files):
            with rasterio.open(file) as dataset:
                aoi = aoi.to_crs(dataset.crs)
                region, region_tfs = riomask.mask(
                    dataset, aoi.geometry, all_touched=False, crop=True
                )
                for band in range(region.shape[0]):
                    stacks.append(region[band])

            raster_mask = rasterio.features.rasterize(
                shapes=aoi.boundary.geometry.tolist(),
                out_shape=(region.shape[-2], region.shape[-1]),
                transform=region_tfs,
                default_value=0,
            )

            if mask_aoi is not None:
                mask_aoi = mask_aoi.to_crs(dataset.crs)
                mask = rasterio.features.rasterize(
                    shapes=mask_aoi.geometry.tolist(),
                    out_shape=(region.shape[-2], region.shape[-1]),
                    transform=region_tfs,
                    default_value=255,
                )
                raster_mask = np.clip(raster_mask + mask, 0, 255)

        if len(stacks) > 1:
            # print([x.shape for x in stacks])
            composit_img = np.stack(stacks, axis=0)
        else:
            composit_img = stacks[0]

        tiles = self.tile_image(composit_img, self.tile_size, raster_mask)
        tilename = self._extract_s2_tilename(raster_dir)
        tiledate = self._extract_s2_date(raster_dir)

        tiles_dir, masks_dir = Path(tiles_dir), Path(masks_dir)
        tiles_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        for i, (image, mask) in enumerate(tiles):
            f_name = f"{tilename}_{Path(aoi_file).stem}_{tiledate}_{i}.npy"
            img_path, mask_path = tiles_dir.joinpath(f_name), masks_dir.joinpath(f_name)

            # print(img_path, mask_path)
            if mask.sum() == 0:
                continue

            with open(str(img_path), "wb") as f:
                np.save(f, image)
            with open(str(mask_path), "wb") as f:
                np.save(f, mask)

        return tiles


class ChipsDataset(Dataset):
    """The ChipsDataset is a simple torch.utils.data.Dataset
    instance helping to index and supply image mini chips with
    corresponding masks and/or filters
    @param: index_df - pd.DataFrame object with necessary columns:
                        "filter_chip", "mask_chip", "tci_chip",
                        these contain a paths to mask to ignore,
                        label mask and tci images.
    @param: bands - bands to be stacked together as an input image
    """

    def __init__(self, index_df,
                 bands=['tci', 'nir', 'ndvi'],
                 #transfoms=train_tfs
                 transforms=None
                 ):

        self.index_df = index_df
        self.bands = bands

        self.band_cols = self.index_df.filter(regex='|'.join(bands)).columns
        self.transforms = transforms

    def __len__(self):
        return self.index_df.shape[0]

    def __getitem__(self, idx):
        row = self.index_df.loc[idx, :]

        # mask = cv2.imread(row['mask_path'])
        mask = np.load(row["mask_path"])
        # stacks = []
        #
        # # print(mask, row[self.band_cols])
        #
        # for img in row[self.band_cols]:
        #     # chip = cv2.imread(img)
        #     chip = np.load(img)
        #
        #     print(chip.shape, img)
        #     if len(chip.shape) == 2:
        #         stacks.append(chip)
        #     else:
        #         for channel in range(chip.shape[-1]):
        #             stacks.append(chip[:, :, channel])
        #
        # stacks = [img / 255 for img in stacks]
        image = np.load(row["img_path"]) / 255 # np.stack(stacks, axis=-1)

        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                gt_mask=mask,
                filter_mask=mask_ignore)

            # img, mask, mask_ignore = (transformed['image'],
            #                           transformed['gt_mask'],
            #                           transformed['filter_mask'])
            img, mask = (transformed['image'], transformed['gt_mask'])

            mask = mask.unsqueeze(0)/255
            # mask_ignore = np.where(mask_ignore >= 250, True, False)

            # return img.float(), mask, torch.tensor(mask_ignore)
        else:
            img = torch.from_numpy(image)
            mask = torch.from_numpy(mask/255)
        return img.float(), mask.float()

if __name__ == "__main__":

    # aois = [
    #     str(x)
    #     for x in Path("/home/val/soil_erosion/data").iterdir()
    #     if "_aoi" in str(x)
    # ]
    # # print(aois)
    # d = ImageryReaderDataset(aois, "/home/val/soil_erosion/imagery/2018")
    # d.prepare_data()

    import pandas as pd
    df = pd.read_csv("../notebooks/chip_index.csv")

    df["TCI_path"] = df["img_path"]
    dataset = ChipsDataset(df[df.skip == 0].reset_index(), bands=["TCI"])
    print(dataset[0][0].shape, dataset[0][1].shape)
