## General Codebase

-	**EasyInstallDeprecationWarning**

	When running `pipenv run build`, the above warning is raised due to the now outdated `setup.py develop` command being called. It is recommended to instead run `pip install -e .`, which works the same, however this command runs `setup.py develop` implicitly, raising the exact same warning.

## `input_features.py`

-   **Port `librosa.vqt()` to PyTorch**

    The `librosa.vqt` function is written solely in python, whereas a pytorch version would be written with a c++ backend, using python bindings. This would be much quicker to use, as the CQT/VQT is a slow algorithm to begin with. This would also remove the need to import librosa altogether. Librosa also depends on libsndfile to be installed, which is not the default in Linux. 

## `geometry.py`

-   **Missing a reliable algorithm to generate all concave shapes**

    Currently, convex shapes can be deterministically created with an efficient algorithm, as well as some concave shapes. However, due to the algorithm chosen to create concave shapes, not _all_ possible concave shapes can be created. The concave algorithm works by centring a collection of random points around the origin, and connecting them according to their polar angle. More complex concave shapes do not share this property, as it is possible for concave shapes to not obey this ordering, whilst maintaining that there are no line crossings. The best solution I have found so far is [scikit-geometry's](https://github.com/scikit-geometry/scikit-geometry) python wrapper around CGAL's [random_polygon_2()](https://doc.cgal.org/latest/Generator/group__PkgGeneratorsRef.html#gaa8cb58e4cc9ab9e225808799b1a61174), which uses a 2-opt approach to configuring the polygon. Due to scikit-geometry needing to be built from source, this has not yet been implemented.

-   **concaveNormalisation**

    Concave normalisation is currently not supported. The method currently available does not account for isometric or more complicated transformations, as well as the order of the vertices. Upon completion, the below tests can be applied for all randomly generated polygons.
	```python
	# This test asserts that the largest vector lies across the x-axis.
	self.assertTrue(polygon.vertices[LV[1][0]][0] == 0.)
	self.assertTrue(polygon.vertices[LV[1][1]][0] == 1.)
	```

-	**Add Support for ellipses**

	Currently, only simple polygons are supported by this library, and there is a need to extend this to include elliptical shapes as well. This would involve creating a new type, as well as updating the geometry library - functions such as `area()` and `centroid()` - so as to support this alternative geometric construction.
