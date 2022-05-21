from kac_drumset import generateDataset
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


if __name__ == '__main__':
	main()
	exit()
