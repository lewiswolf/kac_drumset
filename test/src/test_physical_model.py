# core
from unittest import TestCase


class PhysicalModelTests(TestCase):
	'''
	Tests used in conjunction with `physical_model.py`.
	'''

	pass

# 	def test_properties(self) -> None:
# 		'''
# 		Stress test multiple properties of the class DrumModel.
# 		'''

# 		drum_size = [0.3] # [0.9, 0.7, 0.5, 0.3, 0.1]
# 		material_density = [0.26] # [0.75, 0.5, 0.25, 0.125, 0.0625]
# 		tension = [2000.0] # [3000.0, 2000.0, 1500.0, 1000.0]
# 		for i in range(len(drum_size)):
# 			for j in range(len(material_density)):
# 				for k in range(len(tension)):
# 					drum = DrumModel(
# 						allow_concave=False, # this should be true, but concave shapes are not properly managed yet
# 						decay_time=1.0,
# 						max_vertices=10,
# 						drum_size=drum_size[i],
# 						material_density=material_density[j],
# 						tension=tension[k],
# 					)
# 					drum.updateProperties()

# 					# This test asserts that The Courant number λ = γk/h, which is used to confirm
# 					# that the CFL stability criterion is upheld. If λ > 1, the resultant simulation
# 					# will be unstable.
# 					# For a 1D simulation
# 					# self.assertLessEqual(self.drum.cfl, 1.0)
# 					# For a 2D simulation
# 					self.assertLessEqual(drum.cfl, 1 / (2 ** 0.5))

# 					# This test asserts that the conservation law of energy is upheld. This is here
# 					# naively tested, using the waveform itself, but should also be confirmed by
# 					# comparing expected bounds on the Hamiltonian energy throughout the simulation.
# 					drum.length = 1000 # very short simulation
# 					drum.generateWaveform()
# 					self.assertLessEqual(np.max(drum.waveform), 1.0)
# 					self.assertGreaterEqual(np.min(drum.waveform), -1.0)

# 	def test_raised_cosine(self) -> None:
# 		'''
# 		The raised cosine transform is used as the activation function for a physical model. These
# 		tests assert that the raised cosine works as intended, both in the 1 and 2 dimensional case.
# 		'''

# 		# This test asserts that the one dimensional case has the correct peaks.
# 		rc = raisedCosine((100, ), (50, ), sigma=10)
# 		self.assertEqual(np.max(rc), 1.0)
# 		self.assertEqual(np.min(rc), 0.0)

# 		# This test asserts that the two dimensional case has the correct peaks.
# 		rc = raisedCosine((100, 100), (50, 50), sigma=10)
# 		self.assertEqual(np.max(rc), 1.0)
# 		self.assertEqual(np.min(rc), 0.0)
