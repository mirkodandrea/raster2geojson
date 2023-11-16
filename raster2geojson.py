"""
@file main.py
@brief Extracts shapes (connected components) from a raster file and returns them as a GeoDataFrame.
@details Uses a Gaussian filter to smooth out the polygon coordinates.
@note The input raster file must be a GeoTIFF.
@note The output GeoDataFrame is saved as a GeoJSON file.
@author Mirko D'Andrea
@date 16/11/2023
"""

from typing import List, Tuple

import click
import geopandas as gpd
import numpy as np
import rasterio as rio
from affine import Affine
from rasterio.features import rasterize, shapes
from scipy.ndimage import binary_dilation, gaussian_filter1d
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union


def smooth_coords(coords: List[Tuple[float, float]], smooth_sigma: float) -> List[Tuple[float, float]]:
    """
    Smooths the coordinates of a linestring using a Gaussian filter.

    Args:
        p (list): A list of (x, y) tuples representing the coordinates of a linestring.
        smooth_sigma (float): The standard deviation of the Gaussian filter.

    Returns:
        list: A list of (x, y) tuples representing the smoothed coordinates of the linestring.
    """
    x = list(zip(*coords))[0]
    y = list(zip(*coords))[1]

    smooth_x = np.array(
        gaussian_filter1d(
            x,
            smooth_sigma
        ))
    smooth_y = np.array(
        gaussian_filter1d(
            y,
            smooth_sigma
        ))

    # close the linestring
    smooth_y[-1] = smooth_y[0]
    smooth_x[-1] = smooth_x[0]

    smoothed_coords = np.hstack((smooth_x, smooth_y))
    smoothed_coords = zip(smooth_x, smooth_y)
    return list(smoothed_coords)


def smooth_polygon(poly: Polygon, smooth_sigma: float) -> Polygon:
    """
    Uses a gauss filter to smooth out the polygon coordinates.

    Args:
        poly (Polygon): The polygon to be smoothed.
        smooth_sigma (float): The standard deviation of the Gaussian filter.

    Returns:
        MultiPolygon: The smoothed polygon as a MultiPolygon object.
    """
    p = smooth_coords(poly['coordinates'][0], smooth_sigma)
    holes = [smooth_coords(c, smooth_sigma) for c in poly['coordinates'][1:]]

    poly = Polygon(p, holes=holes)

    return poly

# %%


def get_holes_mask(holes_geom: shape, out_shape: Tuple[int, int], transform: Affine) -> np.ndarray:
    """
    Returns a binary mask of the holes in the specified shapefile.

    Args:
        holes_geom (shapely.geometry): The geometry of the holes to cut.
        out_shape (tuple): Shape of the output mask (height, width).
        transform (affine.Affine): Affine transformation to use for the output mask.

    Returns:
        numpy.ndarray: Binary mask of the holes, with the same shape as the input `shape`.
    """

    holes_mask = np.zeros(out_shape, dtype=np.uint8)
    rasterize(
        # list of (geometry, value) tuples
        [(shape(hole), 1) for hole in [holes_geom]],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        out=holes_mask
    )
    return holes_mask


def cut_holes(gdf: gpd.GeoDataFrame, holes_geom: shape) -> gpd.GeoDataFrame:
    """
    Cuts holes in a GeoDataFrame using another GeoDataFrame of holes.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame to cut holes in.
    holes_geom (shapely.geometry): The geometry of the holes to cut.

    Returns:
    GeoDataFrame: The modified GeoDataFrame with holes cut out.
    """
    new_gdf = gdf.copy()
    new_gdf['geometry'] = gdf['geometry'].apply(
        lambda geom: geom.difference(holes_geom))
    return new_gdf


def mask_it(gdf: gpd.GeoDataFrame, mask_geom: shape) -> gpd.GeoDataFrame:
    """
    Masks a GeoDataFrame with a given geometry.
    It will fill the holes between the GeoDataFrame and the mask geometry by buffering the GeoDataFrame by 0.1 and then intersecting it with the mask geometry.

    Parameters:
    gdf (gpd.GeoDataFrame): The GeoDataFrame to be masked.
    mask_geom (shape): The geometry to use as a mask.

    Returns:
    gpd.GeoDataFrame: The masked GeoDataFrame.
    """
    new_gdf = gdf.copy()

    missing = mask_geom.difference(gdf.unary_union)

    new_gdf['geometry'] = gdf['geometry'].apply(
        lambda geom: unary_union(
            [geom, geom.buffer(0.1).intersection(missing)])
    )
    new_gdf['geometry'] = new_gdf['geometry'].apply(
        lambda geom: geom.intersection(mask_geom)
    )

    return new_gdf


def shape_from_raster(
        file_in: str, 
        holes_file: str = None, 
        mask_file: str = None,
        values: List[float] = None,
        smooth: float = 0.8
    ) -> gpd.GeoDataFrame:
    """
    Extracts shapes (connected components) from a raster file and returns them as a GeoDataFrame.

    Args:
        file_in (str): Path to the input raster file.
        holes_file (str, optional): Path to a shapefile containing holes to be cut out from the shapes. Defaults to None.
        values (list, optional): Values to extract. Defaults to None.
        smooth (float, optional): Smoothing factor. Defaults to 0.8.
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the extracted shapes as polygons.
    """
    with rio.open(file_in) as src:
        image = src.read(1)  # Assuming you want the first band

    holes_mask = None
    holes_geom = None
    if holes_file:
        holes_gdf = gpd.read_file(holes_file)
        holes_geom = holes_gdf.unary_union
        holes_mask = get_holes_mask(holes_geom, image.shape, src.transform)

    # Mask out the NaN values
    mask = ~np.isnan(image)

    geoms = []
    # Find shapes (connected components)
    if values is None:
        values = np.unique(image)
        # remove nan
        values = values[~np.isnan(values)]

    for value in values:
        values = ((image >= value) & mask).astype('uint8')

        if holes_mask is not None:
            touching = binary_dilation(values) & binary_dilation(holes_mask)
            values = values | touching

        results = []
        for i, (s, v) in enumerate(shapes(values, mask=values, transform=src.transform)):
            results.append({
                'properties': {
                    'value': value
                },
                'geometry': smooth_polygon(s, smooth) if smooth > 0 else s
            })
        geoms += list(results)
    # Convert to GeoDataFrame

    gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)

    if mask_file:
        mask_gdf = gpd.read_file(mask_file)
        mask_geom = mask_gdf.unary_union
        gdf = mask_it(gdf, mask_geom)

    if holes_geom:
        gdf = cut_holes(gdf, holes_geom)


    return gdf


@click.command('Create a geojson from a raster file, optionally cutting out holes and masking it.')
@click.argument('file_in', type=click.Path(exists=True))
@click.argument('file_out', type=click.Path())
@click.option('--holes', default=None, help='Path to holes file', type=click.Path(exists=True))
@click.option('--mask', default=None, help='Path to mask file', type=click.Path(exists=True))
@click.option('--values', default=None, help='Values to extract', type=click.STRING)
@click.option('--smooth', default=0.8, help='Smoothing factor', type=click.FLOAT)
def main(file_in, file_out, holes, mask, values, smooth):
    values = values.split(',') if values else None
    gdf = shape_from_raster(file_in, holes_file=holes, mask_file=mask, values=values, smooth=smooth)
    gdf.to_file(file_out, driver='GeoJSON')

if __name__ == '__main__':
    main()
