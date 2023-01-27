from evi_calculator import EVICalculator


if __name__ == '__main__':
    calculator = EVICalculator(
        polygon_file='aoi.geojson',
        n_days_ago=5*365,
        zarr_output_dir='output.zarr'
    )
    calculator.calculate()