## General Codebase

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

-   **`pydantic.create_model_from_typeddict` has an incompatible type error**

    See [here](https://github.com/samuelcolvin/pydantic/issues/3008) for details.

## `dataset.py`

-   **Extendable way to loop over TypedDict keys**

    When using the `TypedDict`, it is sometimes necessary to access its various properties using a variable. This methodology allows for extendable code which loops over the object properties contained within the dict. However, when type checked with mypy, this will return the error `TypedDict key must be a string literal`. This issue is well documented [here](https://github.com/python/mypy/issues/6262), and can be recreated using the code shown below on both an instantiated and an uninstantiated dict. In my mind, this issue is trivial, as of course, when used correctly, we can ensure that a variable key can only ever contain the correct values.

    ```python
    from typing import TypedDict

    class SomeDict(TypedDict):
    	key1: int
    	ket2: float
    	key3: str

    test: SomeDict = {
    	key1: 1,
    	key2: 3.1415,
    	key3: 'test',
    }

    for key in test.keys():
    	print(test[key])
    ```

## `input_features.py`

-   **Port `librosa.vqt()` to PyTorch**

    The `librosa.vqt` function is written solely in python, whereas a pytorch version would be written with a c++ backend, using python bindings. This would be much quicker to use, as the CQT/VQT is a slow algorithm to begin with. This would also remove the need to import librosa altogether.

## `geometry.py`

-   **Missing a reliable algorithm to generate all concave shapes**

    Currently, convex shapes can be deterministically created with an efficient algorithm, as well as some concave shapes. However, due to the algorithm chosen to create concave shapes, not _all_ possible concave shapes can be created. The concave algorithm works by centering a collection of random points around the origin, and connecting them according to their polar angle. More complex concave shapes do not share this property, as it is possible for concave shapes to not obey this ordering. An algorithm that can create these shapes however, whilst maintaining that there are no line crossings. Perhaps a random walk/travelling salesman algorithm is required?
