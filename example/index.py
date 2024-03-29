def DatasetExample() -> None:
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
			'amplitude': 1.,
			'decay_time': 2.,
			'drum_size': 0.3,
			'duration': 1.,
			'material_density': 0.2,
			'max_vertices': 10,
			'sample_rate': 48000,
			'strike_width': 0.01,
			'tension': 2000.,
		}),
	)
	# Datasets can be loaded using the method below, which takes only the dataset directory as its argument.
	dataset = loadDataset(dataset_dir)
	# To redefine the input representation, the below method is used. This modifies the metadata for the dataset, such that
	# this method is only executed when the current settings are different from the ones passed to the function.
	representation_settings = {'output_type': 'fft'}
	dataset = transformDataset(dataset, representation_settings)


def GeometryExample() -> None:
	'''
	This example demonstrates all of the methods used to explore geometric analysis of drums.
	'''

	# dependencies
	import numpy as np

	# src
	import kac_drumset.geometry as G

	# Define a circle
	circle = G.Circle()
	print(f'\nA circle with radius {circle.r} has an area of {circle.area}.\n')

	# Define a square.
	square = G.Polygon(np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]]))
	print(f'This is a square: \n \n {square.vertices} \n')
	print(f'It, of course, has {square.N} sides.')
	# Assess its area.
	print(f'Its area is {square.area}.')
	print(
		f'A square {"does" if G.isColinear(square.vertices[0: 3]) else "does not"} contain any points that are colinear.',
	)
	print(
		'The points [[0., 0.], [1., 1.], [2., 2.]], however,',
		f'{"are" if G.isColinear(np.array([[0., 0.], [1., 1.], [2., 2.]])) else "are not"} colinear.',
	)
	# Define a 5 sided convex polygon.
	polygon = G.Polygon(G.generateConvexPolygon(5))
	# Normalise the polygon to the unit interval, and remove isometric and similarity transformations.
	polygon.vertices = G.convexNormalisation(polygon)
	print(f'\nThis is a {polygon.N} sided polygon: \n \n {polygon.vertices} \n')
	# Assess its area.
	print(f"Its area is {polygon.area}.")
	# Compute its convexity.
	print(f'It is {polygon.convex} that this polygon is convex.')
	# Compute the geometric centroid.
	print(f"This polygon's centroid is at {G.centroid(polygon)}.")
	# Determine if an arbitrary point is within the polygon.
	print(f'It is {G.isPointInsidePolygon((0.5, 0.5), polygon)} that the point (0.5, 0.5) is inside of this polygon.')
	# Compute its largest vector pair.
	print(f"The length of this polygon's largest vector is {G.largestVector(polygon)[0]}.")
	c = G.largestVector(polygon)[1]
	print(
		f'And spans the coordinates [({polygon.vertices[c[0], 0]}, {polygon.vertices[c[0], 1]}),',
		f'({polygon.vertices[c[1], 0]}, {polygon.vertices[c[1], 1]})].',
	)

	# Given two shapes, determine whether they may be isospectral using Weyl's asymptotic law.
	print(
		f"\nFor these two shapes, the square and the polygon, Weyl's asymptotic law is {G.weylCondition(square, polygon)}.",
	)


if __name__ == '__main__':
	DatasetExample()
	GeometryExample()
	exit()
