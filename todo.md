## General Codebase

-   **Internal types for nested lists, numpy arrays and pytroch tensors**

    So far, most datatypes have been well documented. When it comes to numpy and tensors however, it is not very simple to set the internal datatypes. In my research so for, I have assertained that numpy cannot be declared with generic types, such that the declaration `npt.NDArray[np.float64]` cannot be replaced by `npt.NDArray[np.genericFloat]` or anything similar. Numpy does not support such a feature, although I imagine it would be possible to import a custom type, that can be used to specify the types for all numpy arrays across the project, but this seems like a complex solution, as all files within the project will be dependent to the file in which this global type is defined. Pytorch has its own set of problems, as the only way to annotate a tensor is with `torch.Tensor`, without any means to specifiy an internal datatype. Pytorch does offer `torch.set_default_dtype(dtype)`, but this has the same complexity issue as the proposed numpy solution. This problem also persists for nested lists, which require explicit typing with respect to dimensionality. As an example, consider a 3-dimensional numpy array, which we then convert to a list; 'np.zeros((3, 3, 3)).tolist()'. The type for this can either be a generic list, `list`, or an explicit type, `list[list[list[float]]]`. I am yet to find a way of defining a list with a specific internal type, that is type checked irrespective of dimensionality.

-   **`pydantic.create_model_from_typeddict` has an incompatible type error**

    See [here](https://github.com/samuelcolvin/pydantic/issues/3008) for details.

## `dataset.py`

-   **Extendable way to loop over TypedDict keys**

    There is a section of code where two TypedDicts must be compared, asessing a boolean relation between corresponding `<key: value>` pairs. The obvious solution to this kind of problem is to use:

    ```python
    for key in [key1, key2, ..., keyN]:
    	dict[key] == other_dict[key]
    ```

    However with a `TypedDict`, this code produces the error `TypedDict key to be string literal`. This issue is well documented [here](https://github.com/python/mypy/issues/6262). The current solution is not very extensible, as adding new keys to corresponding dictionaries necessetates that the code that currently performs this comparison is updated.

## `input_features.py`

-   **Port `librosa.vqt()` to PyTorch**

    The `librosa.vqt` function is written solely in python, whereas a pytorch version would be written with a c++ backend, using python bindings. This would be much quicker to use, as the CQT/VQT is a slow algorithm to begin with. This would also remove the need to import librosa altogether, as well as convert the current function, which works torch.Tensor -> numpy.ndarray -> torch.Tensor, to torch.Tensor -> torch.Tensor. To use terminology borrowed from category theory, this proposed function would be the 'unique morphism' from _a_ to _b_.
