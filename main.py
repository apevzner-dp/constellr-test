from evi_calculator import EVICalculator


if __name__ == '__main__':
    calculator = EVICalculator(
        polygon_file='aoi.geojson',
        n_days_ago=5*365,
        output_file='output.zarr'
    )
    calculator.calculate()