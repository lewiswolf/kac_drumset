def dataset() -> None:
	'''
	This example demonstrates all of the methods used to generate, load, and modify a dataset.
	'''

	# core
	import os

	# src
	from kac_drumset import (
		# methods
		generateDataset,
		loadDataset,
		transformDataset,
		# classes
		FDTDModel,
		# types
		RepresentationSettings,	# typing for the representation_settings
		TorchDataset,			# the dataset class
	)

	# Generating a dataset takes as its first argument an AudioSampler, each of which has its own customised settings
	# constructor. To configure the representation settings, a dict of type RepresentationSettings is passed to the
	# function. This function also takes an absolute path as an argument to store the dataset.
	dataset_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/data')
	representation_settings: RepresentationSettings = {'output_type': 'end2end'}
	dataset: TorchDataset = generateDataset(
		FDTDModel,
		dataset_dir=dataset_dir,
		dataset_size=10,
		representation_settings=representation_settings,
		sampler_settings=FDTDModel.Settings({
			'duration': 1.0,
			'sample_rate': 48000,
		}),
	)
	# Datasets can be loaded using the method below, which takes only the dataset directory as its argument.
	dataset = loadDataset(dataset_dir)
	# To redefine the input representation, the below method is used. This modifies the metadata for the dataset, such that
	# this method is only executed when the current settings are different from the ones passed to the function.
	representation_settings = {'output_type': 'fft'}
	dataset = transformDataset(dataset, representation_settings)


def geometry() -> None:
	'''
	This example demonstrates all of the methods used to explore geometric analysis of drums.
	'''

	# dependencies
	import numpy as np

	# src
	import kac_drumset.geometry as G

	# Define a square.
	square = G.Polygon(np.array([
		[0., 0.],
		[0., 1.],
		[1., 1.],
		[1., 0.],
	]))
	print(f'This is a square: \n \n {square.vertices} \n')
	print(f'It, of course, has {square.n} sides.')
	# Assess its area.
	assert square.area() == 1.
	print(f"Its area is {square.area()}.")
	# A square does not contain any sides...
	assert not G.isColinear(square.vertices[0: 3])
	# whereas a straight line does.
	assert G.isColinear(np.array([
		[0., 0.],
		[1., 1.],
		[2., 2.],
	]))

	# Define a 5 sided convex polygon.
	polygon = G.Polygon(G.generateConvexPolygon(5))
	# Normalise the polygon to the unit interval, and remove isometric and similarity transformations.
	polygon.vertices = G.convexNormalisation(polygon)
	print(f'\nThis is a {polygon.n} sided polygon: \n \n {polygon.vertices} \n')
	# Assess its area.
	print(f"Its area is {polygon.area()}.")
	# Compute its convexity.
	print(f'It is {G.isConvex(polygon)} that this polygon is convex.')
	# Compute the geometric centroid.
	print(f"The polygon's centroid is at {G.centroid(polygon)}.")
	# Compute its largest vector pair.
	print(f"The length of the polygon's largest vector is {G.largestVector(polygon)[0]}.")
	c = G.largestVector(polygon)[1]
	points = [
		polygon.vertices[c[0], 0],
		polygon.vertices[c[0], 1],
		polygon.vertices[c[1], 0],
		polygon.vertices[c[1], 1],
	]
	print(f"And spans the coordinates (({points[0]}, {points[1]}), ({points[2]}, {points[3]})).")

	# Given two shapes, determine whether they may be isospectral using Weyl's asymptotic law.
	print(
		f"\nFor these two shapes, the square and the polygon, Weyl's asymptotic law is {G.weylCondition(square, polygon)}.",
	)


if __name__ == '__main__':
	dataset()
	print()
	geometry()
	exit()
