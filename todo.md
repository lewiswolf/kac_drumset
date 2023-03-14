## General Codebase

-	**EasyInstallDeprecationWarning**

	When running `pipenv run build`, the above warning is raised due to the now outdated `setup.py develop` command being called. It is recommended to instead run `pip install -e .`, which works the same, however this command runs `setup.py develop` implicitly, raising the exact same warning.

## `input_features.py`

-   **Port `librosa.vqt()` to PyTorch**

    The `librosa.vqt` function is written solely in python, whereas a pytorch version would be written with a c++ backend, using python bindings. This would be much quicker to use, as the CQT/VQT is a slow algorithm to begin with. This would also remove the need to import librosa altogether. Librosa also depends on libsndfile to be installed, which is not the default in Linux. 

## geometry

-   **concaveNormalisation**

    Concave normalisation is currently not supported. The method currently available does not account for isometric or more complicated transformations, as well as the order of the vertices. Upon completion, the below tests can be applied for all randomly generated polygons.
	```python
	# This test asserts that the largest vector lies across the x-axis.
	self.assertTrue(polygon.vertices[LV[1][0]][0] == 0.)
	self.assertTrue(polygon.vertices[LV[1][1]][0] == 1.)
	```

-	**Add Support for ellipses**

	Currently, only simple polygons are supported by this library, and there is a need to extend this to include elliptical shapes as well. This would involve creating a new type, as well as updating the geometry library - functions such as `area()` and `centroid()` - so as to support this alternative geometric construction.

-   **Add a method to scale a polygon to a target area**

	```python
	# This test asserts that the area can be accurately scaled to any size.
	from random import random
	target_area = random()
	self.assertEqual(
		target_area,
		polygon.area() * ((target_area / polygon.area()) ** 0.5),
	)
	```

-   **isPointInsidePolygon**

	```python
	assert isConvex(P), 'isPointInsidePolygon() does not currently support concave shapes.'
	```

## Physics

-   **Equilateral amplitudes?**

These seem to be missing the `m` component.
