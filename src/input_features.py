# core
import os

# dependencies
import torch					# pytorch
import torchaudio				# tensor audio manipulation
from tqdm import tqdm			# CLI progress bar

# src
from settings import settings	# creates a project settings object


def fastFourier(x: torch.Tensor) -> torch.Tensor:
	return x


def melSpectro(x: torch.Tensor) -> torch.Tensor:
	return x


def constantQ(x: torch.Tensor) -> torch.Tensor:
	return x


def inputFeatures(data: list[str]) -> torch.Tensor:
	'''
	This method produces the input features to pass to the neural network, by
	first importing a pre-generated .wav file, and return a tensor corresponding
	to a range of input features.
	'''

	tmpList = []

	print('Preprocessing dataset... 📚')
	with tqdm(
		bar_format='{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
		unit=' data samples',
		total=settings['DATASET_SIZE'],
	) as pbar:
		for i in range(settings['DATASET_SIZE']):
			waveform, sr = torchaudio.load(os.path.join(os.getcwd(), data[i]))
			if sr != settings['SAMPLE_RATE']:
				# resample the imported waveform if its sample rate is wrong. This
				# _shouldn't_ happen, hence why a slower resampling function is used.
				waveform = torchaudio.functional.resample(waveform, sr, settings['SAMPLE_RATE'])
			if waveform.shape[0] > 1:
				# convert the imported .wav file to mono if necessary
				mono = torch.zeros(1, waveform.shape[1])
				for i in range(waveform.shape[1]):
					for j in range(waveform.shape[0]):
						mono[0][i] += waveform[j][i] / waveform.shape[0]
				waveform = mono
			if settings['NORMALISE_INPUT']:
				# normalise the audio file
				waveform[0] = waveform[0] * (1.0 / torch.max(waveform[0]))

			# append correct input representation to output tensor
			if settings['INPUT_FEATURES'] == 'end2end':
				tmpList.append(waveform[0])
			if settings['INPUT_FEATURES'] == 'fft':
				tmpList.append(fastFourier(waveform[0]))
			if settings['INPUT_FEATURES'] == 'mel':
				tmpList.append(constantQ(waveform[0]))
			if settings['INPUT_FEATURES'] == 'q':
				tmpList.append(melSpectro(waveform[0]))

			pbar.update(1)

	return torch.stack(tmpList)
