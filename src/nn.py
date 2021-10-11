'''
'''

# core
import sys

# dependencies
import torch					# pytorch
import torch.nn as nn			# neural network module


class NeuralNet(nn.Module):
	'''
	'''

	def __init__(self) -> None:
		pass


def trainModel(
	trainingDataset: torch.utils.data.DataLoader,
	testingDataset: torch.utils.data.DataLoader,
) -> None:
	# ) -> tuple[nn.Module, float]:
	'''
	'''

	# configure device
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		print(f'WARNING{"" if sys.platform not in ["linux", "darwin"] else "❗️"}')
		print('Nvidia GPU support is not available for training the network.')
		device = torch.device('cpu')

	print(f'Training neural network... {"" if sys.platform not in ["linux", "darwin"] else "🧠"}')
