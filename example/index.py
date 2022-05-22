from kac_drumset import generateDataset, loadDataset
from kac_drumset import TestTone


def main() -> None:
	generateDataset(
		TestTone,
		representation_settings={'output_type': 'end2end'},
		sampler_settings=TestTone.Settings({
			'duration': 1.0,
			'waveshape': 'sin',
			'sample_rate': 48000,
		}),
	)
	loadDataset()


if __name__ == '__main__':
	main()
	exit()
