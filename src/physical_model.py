# dependencies
from numba import cuda			# GPU acceleration

if not cuda.is_available():
	print('WARNING❗️ Nvidia GPU support is not available.')
