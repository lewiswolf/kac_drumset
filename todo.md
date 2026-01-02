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

-   **UnitTriangle**

## Physics

-   **Equilateral amplitudes?**

These seem to be missing the `m` component.
