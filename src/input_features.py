# core
import math
import os

# dependencies
import numpy as np 				# maths
import librosa					# numpy audio manipulation
import torch					# pytorch
import torchaudio				# tensor audio manipulation
from tqdm import tqdm			# CLI progress bar

# src
from settings import settings	# creates a project settings object

# tests
import sys
sys.path.insert(1, os.path.join(os.getcwd(), 'test'))
from test_utils import plotSpectrogram, plotWaveform


def inputFeatures(data: list[str]) -> torch.Tensor:
	'''
	This method produces the input features to pass to the neural network, by
	first importing a pre-generated .wav file, returning a tensor corresponding
	to a range of input features.
	'''

	tmpList = []

	# pre-compute spectrogram settings for constant q transform
	if settings['INPUT_FEATURES'] == 'cqt':
		f_min = 22.05
		max_octaves = math.log2((settings['SAMPLE_RATE'] * 0.5) / f_min)
		bins_per_octave = math.floor(settings['SPECTRO_SETTINGS']['n_bins'] / max_octaves)
		n_bins = math.floor(bins_per_octave * max_octaves) # var used for CQT only!
	else:
		f_min = 0.0

	print('Preprocessing dataset... 📚')
	with tqdm(
		bar_format='{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
		unit=' data samples',
		total=settings['DATASET_SIZE'],
	) as pbar:
		for i in range(settings['DATASET_SIZE']):
			waveform, sr = torchaudio.load(os.path.join(os.getcwd(), data[i]))

			# resample the imported waveform if its sample rate is wrong
			if sr != settings['SAMPLE_RATE']:
				waveform = torchaudio.transforms.Resample(sr, settings['SAMPLE_RATE'])(waveform)

			# convert the imported .wav file to mono
			if waveform.shape[0] > 1:
				waveform = torch.mean(waveform, 0)
			else:
				waveform = waveform[0]

			# normalise the audio file
			if settings['NORMALISE_INPUT'] and torch.max(waveform) != 1.0:
				waveform = waveform * (1.0 / torch.max(waveform))

			# append correct input representation to output tensor
			if settings['INPUT_FEATURES'] == 'end2end':
				tmpList.append(waveform)

			if settings['INPUT_FEATURES'] == 'fft':
				tmpList.append(torchaudio.transforms.Spectrogram(
					n_fft=settings['SPECTRO_SETTINGS']['n_bins'],
					win_length=settings['SPECTRO_SETTINGS']['window_length'],
					hop_length=settings['SPECTRO_SETTINGS']['hop_length'],
					power=2.0,
				)(waveform))

			if settings['INPUT_FEATURES'] == 'mel':
				tmpList.append(torchaudio.transforms.MelSpectrogram(
					sample_rate=settings['SAMPLE_RATE'],
					n_mels=settings['SPECTRO_SETTINGS']['n_mels'],
					n_fft=settings['SPECTRO_SETTINGS']['n_bins'],
					win_length=settings['SPECTRO_SETTINGS']['window_length'],
					hop_length=settings['SPECTRO_SETTINGS']['hop_length'],
					power=2.0,
				)(waveform))

			if settings['INPUT_FEATURES'] == 'cqt':
				# TO ADD: see todo.md -> 'Port librosa.vqt() to PyTorch'
				arr = waveform.detach().numpy()
				arr = librosa.cqt(
					arr,
					sr=settings['SAMPLE_RATE'],
					n_bins=n_bins,
					bins_per_octave=bins_per_octave,
					fmin=f_min,
					hop_length=settings['SPECTRO_SETTINGS']['hop_length'],
					dtype=np.float32,
				)
				tmpList.append(torch.abs(torch.from_numpy(arr)))

			pbar.update(1)

	if settings['INPUT_FEATURES'] == 'end2end':
		plotWaveform(tmpList[0].detach().numpy(), settings['SAMPLE_RATE'])
	else:
		plotSpectrogram(
			tmpList[0].detach().numpy(),
			input_type=settings['INPUT_FEATURES'],
			sr=settings['SAMPLE_RATE'],
			hop_length=settings['SPECTRO_SETTINGS']['hop_length'],
			f_min=f_min,
		)

	return torch.stack(tmpList)
