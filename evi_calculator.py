from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from pystac_client import Client
import queue
import rioxarray
import threading
import time
import xarray


class EVICalculator:
    API_URL = 'https://earth-search.aws.element84.com/v0'
    COLLECTION = "sentinel-s2-l2a-cogs"  # Sentinel-2, Level 2A, COGs
    MAX_THREADS = 8

    def __init__(self, polygon_file, n_days_ago, zarr_output_dir):
        """
        The output where the calculated EVI will be stored in Zarr format
        """
        self._zarr_output_dir = zarr_output_dir

        """
        The flag that indicates that all the available scenes have been pushed to the processing queue
        """
        self._querying_finished = False

        """
        We will add the scenes to this queue
        """
        self._processing_queue = queue.Queue(maxsize=self.MAX_THREADS)

        """
        We will need an asynchronous access to the _evi_sums dictionary so we need a lock object
        """
        self._evi_lock = threading.Lock()

        """
        The sums of the calculated EVIs will be saved here
        """
        self._evi_sums = {}

        """
        The coordinate systems of different scenes may be different, so we need to have a single CRS
        to reproject all the scenes to this CRS
        """
        self._crs = None

        """
        The calculated average EVI will be stored here
        """
        self._evi_avg = None

        self._threads = []

        with open(polygon_file, 'r') as fp:
            self._aoi_dict = json.load(fp)['features'][0]['geometry']
        
        self._client = Client.open(self.API_URL)

        self._start_date = (datetime.now() - timedelta(days=n_days_ago)).strftime('%Y-%m-%d')
        self._end_date = datetime.now().strftime('%Y-%m-%d')
    
    def calculate(self):
        search = self._client.search(
            collections=[self.COLLECTION],
            intersects=self._aoi_dict,
            datetime=f'{self._start_date}/{self._end_date}',
            max_items=None,
            limit=100
        )

        self._start_processing_threads()

        for item in search.items():
            self._processing_queue.put(item)
            print('An item is pushed to the queue')
        self._querying_finished = True

        self._wait_for_processing_threads()

        self._calculate_evi_avg()
        self._plot_evi_avg()
    
    def _start_processing_threads(self):
        self._threads = [
            threading.Thread(target=self.processing_thread, args=(self,))

            for _ in range(self.MAX_THREADS)
        ]

        for thread in self._threads:
            thread.start()
    
    def _wait_for_processing_threads(self):
        for thread in self._threads:
            thread.join()
        
    def _process_item(self, item):
        bbox_key = tuple(item.bbox)

        with \
            rioxarray.open_rasterio(item.assets['B08'].href) as nir_asset, \
            rioxarray.open_rasterio(item.assets['B04'].href) as red_asset, \
            rioxarray.open_rasterio(item.assets['B02'].href) as blue_asset:

            if not self._crs:
                self._crs = nir_asset.rio.crs

            if self._crs:
                nir_asset = nir_asset.rio.reproject(self._crs)
                red_asset = red_asset.rio.reproject(self._crs)
                blue_asset = blue_asset.rio.reproject(self._crs)

            nir_clipped = nir_asset.rio.clip(geometries=[self._aoi_dict], crs=4326)
            red_clipped = red_asset.rio.clip(geometries=[self._aoi_dict], crs=4326)
            blue_clipped = blue_asset.rio.clip(geometries=[self._aoi_dict], crs=4326)

            evi = self._calculate_evi(nir=nir_clipped, red=red_clipped, blue=blue_clipped)

            # convert evi to dataset to use xarray.merge() further
            evi = evi.to_dataset(name='calculated_values')

            # the calculated EVI has only one band so we don't need the band dimension anymore
            evi = evi.squeeze(dim='band', drop=True)

        count_mask = xarray.full_like(evi, 1)

        self._evi_lock.acquire()
        
        if bbox_key in self._evi_sums.keys():
            self._evi_sums[bbox_key]['count_mask'] += count_mask
            self._evi_sums[bbox_key]['evi'] += evi
        else:
            self._evi_sums[bbox_key] = {'count_mask': count_mask, 'evi': evi}
            
        self._evi_lock.release()

        print('Processing finished')
    
    def _calculate_evi(self, nir, red, blue):
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    
    def _calculate_zeros_array_merged(self):
        keys = list(self._evi_sums.keys())

        merged = self._evi_sums[keys[0]]['evi']
        for i in range(1, len(keys)):
            merged = xarray.merge([merged, self._evi_sums[keys[i]]['evi']], compat='override')
        
        return xarray.zeros_like(merged)
    
    def _calculate_evi_avg(self):
        zeros_array_merged = self._calculate_zeros_array_merged()
        keys = list(self._evi_sums.keys())

        evi_merged = xarray.merge([self._evi_sums[keys[0]]['evi'], zeros_array_merged], compat='override', fill_value=0.0)
        count_mask_merged = xarray.merge([self._evi_sums[keys[0]]['count_mask'], zeros_array_merged], compat='override', fill_value=0.0)

        for i in range(1, len(keys)):
            evi_merged += xarray.merge([self._evi_sums[keys[i]]['evi'], zeros_array_merged], compat='override', fill_value=0.0)
            count_mask_merged += xarray.merge([self._evi_sums[keys[i]]['count_mask'], zeros_array_merged], compat='override', fill_value=0.0)
        
        self._evi_avg = evi_merged / count_mask_merged
        self._evi_avg.to_zarr(self._zarr_output_dir)
    
    def _plot_evi_avg(self):
        x = self._evi_avg.x.values
        y = self._evi_avg.y.values
        values = self._evi_avg.to_array()[0]

        plt.pcolormesh(x, y, values)
        plt.colorbar()
        plt.show()

    @staticmethod
    def processing_thread(self_instance):
        while True:
            try:
                item = self_instance._processing_queue.get_nowait()
                print('An item is popped from the queue')
                self_instance._process_item(item)
            except queue.Empty:
                if self_instance._querying_finished:
                    return
                time.sleep(0.01) 

