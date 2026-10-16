"""Rendered regression checks for the stream-NPZ quicklook plot layout."""
import tempfile
from pathlib import Path
import unittest

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import plot_npz


class PlotColorbarLayoutTests(unittest.TestCase):
    def test_sfu_title_is_horizontal_above_colorbar_without_tick_overlap(self):
        for low, high in [(0.4, 2.4), (0.001, 10000.0), (50.0, 1500.0)]:
            with self.subTest(low=low, high=high), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                intensity = np.broadcast_to(np.geomspace(low, high, 768), (32, 768)).copy()
                circular = 0.1 * intensity
                source = root / '2026-10-16_14_26_19.npz'
                np.savez(source, data=np.stack((intensity, circular), axis=-1),
                         mjd=(60000 + np.arange(32) * 0.5 / 86400)[:, None],
                         freq=np.linspace(15, 85, 768))
                fig, output = plot_npz.plot_spectrums([str(source)], output_path=str(root / 'plot.png'))
                self.assertTrue(Path(output).is_file())
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                renderer = canvas.get_renderer()
                bars = [axis for axis in fig.axes if axis.get_title() == '[sfu]']
                self.assertEqual(len(bars), 1, 'The unit must be a horizontal colorbar title')
                bar = bars[0]
                self.assertEqual(bar.get_ylabel(), '', 'Remove the old vertical unit label')
                self.assertEqual(bar.title.get_rotation(), 0)
                unit_box = bar.title.get_window_extent(renderer)
                bar_box = bar.get_window_extent(renderer)
                lower, upper = bar.get_ylim()
                tick_boxes = [label.get_window_extent(renderer)
                              for label in bar.get_yticklabels(which='both')
                              if label.get_visible() and label.get_text()
                              and lower <= label.get_position()[1] <= upper]
                self.assertTrue(tick_boxes)
                self.assertGreaterEqual(unit_box.y0 - bar_box.y1, 5,
                                        'The unit title must sit above the colorbar')
                self.assertFalse(any(unit_box.overlaps(box) for box in tick_boxes))
                self.assertAlmostEqual((unit_box.x0 + unit_box.x1) / 2,
                                       (bar_box.x0 + bar_box.x1) / 2, delta=1)
                self.assertGreaterEqual(unit_box.x0, fig.bbox.x0)
                self.assertLessEqual(unit_box.x1, fig.bbox.x1)
                self.assertLessEqual(unit_box.y1, fig.bbox.y1)
                np.testing.assert_array_equal(np.ma.getdata(fig.axes[0].images[0].get_array()), intensity.T)
                np.testing.assert_allclose(np.ma.getdata(fig.axes[1].images[0].get_array()), circular.T / intensity.T)
                self.assertEqual(fig.axes[1].images[0].get_clim(), (-0.7, 0.7))


if __name__ == '__main__':
    unittest.main()
