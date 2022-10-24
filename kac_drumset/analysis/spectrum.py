# dependencies
import librosa
import numpy as np
import numpy.typing as npt
from scipy import signal

__all__ = [
	'dominantModes',
]


def dominantModes(
	waveform: npt.NDArray[np.float64],
	sample_rate: int,
	fft_size: int = 2048,
) -> npt.NDArray[np.float64]:
	'''
	Harmonic analysis adapted from `getPitch`:
	https://github.com/beefoo/media-tools/blob/master/lib/audio_utils.py.
	'''

	# increase margin for higher filtering of noise (probably between 1 and 8)
	waveform = librosa.effects.harmonic(waveform, margin=4)
	waveform = np.nan_to_num(waveform)
	pitches, magnitudes = librosa.core.piptrack(y=waveform, sr=sample_rate, n_fft=fft_size)

	# get sum of mags at each time
	magFrames = magnitudes.sum(axis=0) # get the sum of bins at each time frame
	t = magFrames.argmax()

	# get peaks at time t
	magBins = magnitudes[:, t]
	peaks = list(signal.find_peaks(magBins, distance=18, height=np.median(magBins))[0])
	harmonics = pitches[peaks, t]

	return harmonics
