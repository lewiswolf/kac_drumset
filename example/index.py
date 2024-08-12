'''
A series of example scripts demonstrating the functionality of the kac_drumset package.
'''


def DatasetExample() -> None:
	'''
	This example demonstrates all of the methods used to generate, load, and modify a dataset.
	'''

	# core
	import os

	# src
	from kac_drumset.geometry import (
		ConvexPolygon
		Ellipse,
		Shape,
	)
	from kac_drumset.samplers import (
		BesselModel,
		FDTDModel,
		LaméModel,
		PoissonModel,
	)
	from kac_prediction.dataset import (
		# methods
		generateDataset,
		loadDataset,
		transformDataset,
		# types
		RepresentationSettings,
		TorchDataset,
	)

	# Default variables for this script.
	dataset: TorchDataset
	dataset_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/data')
	representation_settings: RepresentationSettings = {'output_type': 'end2end'}

	# Generate a dataset of 2D physical models of varying geometric design.
	# Here we use the FDTDModel, parametrised using objects of the class Shape.
	shapes: list[type[Shape]] = [ConvexPolygon, Ellipse]
	for shape in shapes:
		dataset = generateDataset(
			FDTDModel,
			dataset_dir=f'{dataset_dir}/{shape.__name__}',
			dataset_size=10,
			representation_settings=representation_settings,
			sampler_settings=FDTDModel.Settings({
				'amplitude': 1.,
				'arbitrary_shape': shape,
				'decay_time': 2.,
				'drum_size': 0.3,
				'duration': 1.,
				'material_density': 0.2,
				'sample_rate': 48000,
				'shape_settings': Shape.Settings(),
				'strike_width': 0.01,
				'tension': 2000.,
			}),
		)

	# Genereate a dataset of linear models of simple geometric design.
	# Here generateDataset() is parametrised using objects of the class AudioSampler.
	generateDataset(
		BesselModel,
		dataset_dir=f'{dataset_dir}/{BesselModel.__name__}',
		dataset_size=10,
		representation_settings=representation_settings,
		sampler_settings=BesselModel.Settings({
			'M': 10,
			'N': 10,
			'amplitude': 1.,
			'decay_time': 2.,
			'duration': 1.,
			'material_density': 0.2,
			'sample_rate': 48000,
			'tension': 2000.,
		}),
	)
	dataset = generateDataset(
		LaméModel,
		dataset_dir=f'{dataset_dir}/{LaméModel.__name__}',
		dataset_size=10,
		representation_settings=representation_settings,
		sampler_settings=LaméModel.Settings({
			'M': 10,
			'N': 10,
			'amplitude': 1.,
			'decay_time': 2.,
			'duration': 1.,
			'material_density': 0.2,
			'sample_rate': 48000,
			'tension': 2000.,
		}),
	)
	generateDataset(
		PoissonModel,
		dataset_dir=f'{dataset_dir}/{PoissonModel.__name__}',
		dataset_size=10,
		representation_settings=representation_settings,
		sampler_settings=PoissonModel.Settings({
			'M': 10,
			'N': 10,
			'amplitude': 1.,
			'decay_time': 2.,
			'duration': 1.,
			'material_density': 0.2,
			'sample_rate': 48000,
			'tension': 2000.,
		}),
	)

	# Datasets can be loaded using the method below, which takes only the dataset directory as its argument.
	dataset = loadDataset(f'{dataset_dir}/{Ellipse.__name__}')
	# To redefine the input representation, the below method is used. This modifies the metadata for the dataset, such
	# that this method is only executed when the current settings are different from the ones passed to the function.
	representation_settings = {'output_type': 'fft'}
	dataset = transformDataset(dataset, representation_settings)


def GeometryExample() -> None:
	'''
	This example demonstrates all of the methods used to explore geometric analysis of drums.
	'''

	# dependencies
	import numpy as np

	# src
	from kac_drumset.geometry import (
		isColinear,
		largestVector,
		lineIntersection,
		weylCondition,
		Circle,
		ConvexPolygon,
		UnitRectangle,
	)

	# Define a circle
	circle = Circle()
	print(f'\nA circle with radius {circle.r} has an area of {circle.area}.\n')

	# Define a square.
	square = UnitRectangle(1.)
	print(f'This is a square: \n \n {square.vertices} \n')
	print(f'It, of course, has {square.N} sides.')
	# Assess its area.
	print(f'Its area is {square.area}.')
	print(
		f'A square {"does" if isColinear(square.vertices[0: 3]) else "does not"} contain any points that are colinear.',
	)
	print(
		'The points [[0., 0.], [1., 1.], [2., 2.]], however,',
		f'{"are" if isColinear(np.array([[0., 0.], [1., 1.], [2., 2.]])) else "are not"} colinear.',
	)
	# Define a 5 sided convex polygon.
	polygon = ConvexPolygon(5)
	print(f'\nThis is a {polygon.N} sided polygon: \n \n {polygon.vertices} \n')
	# Assess its area.
	print(f'Its area is {polygon.area}.')
	# Compute its simplicity.
	print(f'It is {polygon.isSimple()} that this polygon is simple.')
	# Compute its convexity.
	print(f'It is {polygon.convex} that this polygon is convex.')
	# Compute the geometric centroid.
	print(f"This polygon's centroid is at {polygon.centroid}.")
	# Determine if an arbitrary point is within the polygon.
	print(f'It is {polygon.isPointInside((0.5, 0.5))} that the point (0.5, 0.5) is inside of this polygon.')
	# Compute its largest vector pair.
	print(f"The length of this polygon's largest vector is {largestVector(polygon.vertices)[0]}.")
	c = largestVector(polygon.vertices)[1]
	print(
		f'And spans the coordinates [({polygon.vertices[c[0], 0]}, {polygon.vertices[c[0], 1]}),',
		f'({polygon.vertices[c[1], 0]}, {polygon.vertices[c[1], 1]})].',
	)

	# Define a line
	line_a = np.array([[0., 0.], [1., 0.]])
	line_b = np.array([[0., 1.], [1., 1.]])
	do_they_intersect, and_where = lineIntersection(line_a, line_b)
	print(
		'\nThe two lines, [[0., 0.], [1., 0.]] and [[0., 1.], [1., 1.]],',
		f'{"" if do_they_intersect else "do not "}intersect at point',
		f'({and_where[0]}, {and_where[1]}).',
	)

	# Given two shapes, determine whether they may be isospectral using Weyl's asymptotic law.
	print(
		f"\nFor these two shapes, the square and the polygon, Weyl's asymptotic law is {weylCondition(square, polygon)}.",
	)


if __name__ == '__main__':
	DatasetExample()
	GeometryExample()
	exit()
