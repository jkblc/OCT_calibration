"""Only works with single B-scans with 2D (no volume dimension)"""
import numpy as np
import logging

class cfg:
    input_file = 'dark_not.raw'
    output_file = 'dark_not.npy'

    # Acquisition shape configuration
    bscans_per_volume = 1
    ascans_per_bscan = 1000
    samples_per_ascan = 2752
    channels_per_sample = 2

    # Extraction configuration
    target_channel = 0
    target_bscan = 0
    average_ascans = True

    log_level = 1 # log verbosity (higher number means less verbose)

class RawConverter:
    def __init__(self, cfg):
        # Initialize basic logger to mirror Vortex logging behavior
        logging.basicConfig(format='%(name)s: %(message)s')
        self._log = logging.getLogger('converter')
        self._log.setLevel(logging.WARNING if cfg.log_level > 0 else logging.INFO)

        self.input_file = cfg.input_file
        self.output_file = cfg.output_file

        self.shape = (
            cfg.bscans_per_volume,
            cfg.ascans_per_bscan,
            cfg.samples_per_ascan,
            cfg.channels_per_sample
        )

        self.target_channel = cfg.target_channel
        self.target_bscan = cfg.target_bscan
        self.average_ascans = cfg.average_ascans

    def run(self):
        self._log.info(f'Loading {self.input_file}')

        raw_data = np.fromfile(self.input_file, dtype=np.uint16)

        # The file saved by 'Save Raw B-scan' is strictly 2D: (ascans, samples)
        # Check your specific acquisition parameters. Based on your file size,
        # it is 1000 ascans and 1664 samples.
        actual_ascans = 1000
        actual_samples = 2816
        expected_elements = actual_ascans * actual_samples

        if raw_data.size != expected_elements:
            raise ValueError(f'Size mismatch: File has {raw_data.size} elements, '
                             f'but expected {expected_elements}.')

        # Reshape to 2D
        raw_bscan = raw_data.reshape((actual_ascans, actual_samples))
        self._log.info(f'Reshaped data to {(actual_ascans, actual_samples)}')

        # Process A-scans
        if self.average_ascans:
            processed_spectra = np.mean(raw_bscan, axis=0)
        else:
            processed_spectra = raw_bscan[0, :]

        # The calibration script expects shape (1, 1, samples)
        final_array = processed_spectra.reshape(1, 1, -1)

        self._log.info(f'Saving formatted data with shape {final_array.shape} to {self.output_file}')
        np.save(self.output_file, final_array)

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description='Convert raw binary OCT data to .npy for calibration', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-file', default=cfg.input_file, help='Path to the .raw file')
    parser.add_argument('--output-file', default=cfg.output_file, help='Path to save the .npy file')

    parser.add_argument('--bscans-per-volume', type=int, default=cfg.bscans_per_volume, help='Number of B-scans')
    parser.add_argument('--ascans-per-bscan', type=int, default=cfg.ascans_per_bscan, help='Number of A-scans per B-scan')
    parser.add_argument('--samples-per-ascan', type=int, default=cfg.samples_per_ascan, help='Number of samples per A-scan')
    parser.add_argument('--channels-per-sample', type=int, default=cfg.channels_per_sample, help='Number of acquisition channels')

    parser.add_argument('--target-channel', type=int, default=cfg.target_channel, help='Channel index to extract (0 for ChA, 1 for ChB)')
    parser.add_argument('--target-bscan', type=int, default=cfg.target_bscan, help='Index of the B-scan to extract')
    parser.add_argument('--no-avg', action='store_true', help='Disable A-scan averaging')
    parser.add_argument('--log-level', type=int, default=cfg.log_level, help='Log verbosity (higher is less verbose)')

    args = parser.parse_args()

    cfg.input_file = args.input_file
    cfg.output_file = args.output_file
    cfg.bscans_per_volume = args.bscans_per_volume
    cfg.ascans_per_bscan = args.ascans_per_bscan
    cfg.samples_per_ascan = args.samples_per_ascan
    cfg.channels_per_sample = args.channels_per_sample
    cfg.target_channel = args.target_channel
    cfg.target_bscan = args.target_bscan
    cfg.average_ascans = not args.no_avg
    cfg.log_level = args.log_level

    converter = RawConverter(cfg)
    converter.run()