from pathlib import Path

from bokeh.io import show
from bokeh.plotting import figure

from qusi.experimental.application.tess import TessMissionLightCurve


def main():
    light_curve_path = Path(
        'data/spoc_transit_experiment/train/positives/<add_downloaded_file_name_here>.fits')
    light_curve = TessMissionLightCurve.from_path(light_curve_path)
    light_curve_figure = figure(x_axis_label='Time (BTJD)', y_axis_label='Flux')
    light_curve_figure.scatter(x=light_curve.times, y=light_curve.fluxes)
    light_curve_figure.line(x=light_curve.times, y=light_curve.fluxes, line_alpha=0.3)
    show(light_curve_figure)


if __name__ == '__main__':
    main()
