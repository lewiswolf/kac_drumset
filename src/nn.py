'''
'''

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
		print('WARNING❗️ Nvidia GPU support is not available for training the network.')
		device = torch.device('cpu')

	print('Training neural network... 🧠')
