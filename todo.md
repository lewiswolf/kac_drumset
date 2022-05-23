## General Codebase

-	**Dependency conflicts**

	torchaudio does not support python 3.10.

-   **Internal types for nested lists, numpy arrays and pytroch tensors**

    So far, most datatypes have been well documented throughout this codebase. However, when it comes to various 'array-like' datatypes, ensuring the correct _internal_ datatype for each array has quickly become a complex and difficult task.

    For nested lists, it is very easy to declare the correct internal datatype using `list[float]` or `lint[Union[int, None]]` etc. However, this declaration cannot be extended to include arbitrarily nested lists. Given a multi dimensional list, its correct typing will be `list[list[list[...list[float]]]]` where the number of occurences of `list` equals the dimensionality of the data. The shorthand for this, which is necessary to use if the dimensionality of the data is variable or not known beforehand, is to simply specify `list` on its own. This latter declaration however says nothing about its internal datatypes.

    Numpy arrays do not share this problem, as their specific type declarations, i.e `npt.NDArray[np.float64]`, function irrespective of dimensionality. The caveat in this case, is that if the precision of the internal type needed to be changed, this would have to be performed across the entire codebase, and missing any one function will likely render this effort pointless. There is no way of replacing this with something along the lines of `npt.NDArray[np.genericFloat]` and setting the precision elsewhere - numpy requires that each declaration and invocation specifies the correct internal datatype.

    PyTorch and its related libraries again have their own problem, as their typing system revolves solely around the use of `torch.Tensor`, which includes no specificity of the internal datatypes whatsoever. If one tensor needed to be ints, and one floats, there is no way to differentiate between these two objects. As a consolation, PyTorch does offer the function `torch.set_default_dtype(dtype)` which can used to set the default precision across the whole project. This specific function is necessary to run certain operations, as demonstrated by the code shown below (and whether or not this is a bug in the torchaudio library, I am unsure).

    ```python
    import numpy as np
    import torch
    import torchaudio

    waveform = torch.from_numpy(np.sin(2 * np.pi * 440.0 * (np.arange(44100) / 44100)))

    # regular spectrograms work fine
    spectrogram_float = torchaudio.transforms.Spectrogram()(waveform.float())
    spectrogram_double = torchaudio.transforms.Spectrogram()(waveform.double())

    # mel spectrograms however...
    spectrogram_float = torchaudio.transforms.MelSpectrogram(
    	sample_rate=44100,
    )(waveform.float())

    # without this line:
    torch.set_default_dtype(torch.float64)
    # RuntimeError: expected scalar type Double but found Float
    spectrogram_double = torchaudio.transforms.MelSpectrogram(
    	sample_rate=44100,
    )(waveform.double())
    ```

    This issue, specifically with pytorch, has also affected testing, as placing `torch.set_default_dtype(dtype)` somewhere within the code affects every function and file thereafter. I have not been able to find a way of running dedcated tests on idividual files, in an attempt to minimise these issues in the event that the library was to be separated/partially reused, without having to split the tests across multiple files and run them individually.

## `input_features.py`

-   **Port `librosa.vqt()` to PyTorch**

    The `librosa.vqt` function is written solely in python, whereas a pytorch version would be written with a c++ backend, using python bindings. This would be much quicker to use, as the CQT/VQT is a slow algorithm to begin with. This would also remove the need to import librosa altogether. Librosa also depends on libsndfile to be installed, which is not the default in Linux. 

## `geometry.py`

-   **Missing a reliable algorithm to generate all concave shapes**

    Currently, convex shapes can be deterministically created with an efficient algorithm, as well as some concave shapes. However, due to the algorithm chosen to create concave shapes, not _all_ possible concave shapes can be created. The concave algorithm works by centering a collection of random points around the origin, and connecting them according to their polar angle. More complex concave shapes do not share this property, as it is possible for concave shapes to not obey this ordering, whilst maintaining that there are no line crossings. The best solution I have found so far is [scikit-geometry's](https://github.com/scikit-geometry/scikit-geometry) python wrapper around CGAL's [random_polygon_2()](https://doc.cgal.org/latest/Generator/group__PkgGeneratorsRef.html#gaa8cb58e4cc9ab9e225808799b1a61174), which uses a 2-opt approach to configuring the polygon. Due to scikit-geometry needing to be built from source, this has not yet been implemented.

-   **groupNormalisation**

    Group normalisation is meant to be a function that normalises polygons according to group theory, so as to remove translated variations of polygons. At the moment, for convex shapes, this works to some degree, but shapes can still be arbitrarily flipped across both the y-axis and x-axis. The way to fix this is to construct an algorithm that initially sets the longest vector equal to 1.0 with angle π/2, serving to remove any rotational transformations. Then, the four quadrants of the polygon are compared and given a set translation, which serves to remove any reflective transformations. For concave shapes, the algorithm is more complex, and is currently undetermined.

-	**Add Support for ellipses**

	Currently, only simple polygons are supported by this library, and there is a need to extend this to include elliptical shapes as well. This would involve creating a new type, as well as updating the geometry library - functions such as `area()` and `centroid()` - so as to support this alternative geometric construction.
