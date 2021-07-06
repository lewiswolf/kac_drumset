## General Codebase

-   **Internal types for both numpy arrays and pytroch tensors**

    So far, most internal types have been well documented. When it comes to numpy and tensors however, it is not very simple to set the internal datatypes. In my research so for, I have assertained that numpy cannot be declared with generic types, such that the declaration `npt.NDArray[np.float64]` cannot be replaced by `npt.NDArray[np.genericFloat]` or anything similar. Numpy does not support such a feature, although I imagine it would be possible to import a custom type, that can be used to specify the types for all numpy arrays across the project, but this seems like a complex solution, as all files within the project will be dependent to the file in which this global type is defined. Pytorch has its own set of problems, as the only way to annotate a tensor is with `torch.Tensor`, without any means to specifiy an internal datatype. Pytorch does offer `torch.set_default_dtype(dtype)`, but this has the same complexity issue as the proposed numpy solution.

## dataset.py

-   **Type check imported json file at runtime**

    When importing the file _metadata.json_, it is expected to be of the type `DatasetMetadata`, which can generally be assumed because of the way the object has been constructed. However, if the json file is somehow corrupted by the time it comes to importing it, there is no way for the system to reliably catch these errors, except for specifying them directly, such as with KeyError. A non-hacky version of [this](https://stackoverflow.com/questions/66665336)?

-   **Extendable way to loop over TypedDict keys**

    There is a section of code where two TypedDicts must be compared, asessing a boolean relation between corresponding `<key: value>` pairs. The obvious solution to this kind of problem is to use:

    ```python
    for key in dict.keys():
    	dict[key] == other_dict[key]
    ```

    However with a `TypedDict`, this code produces the error `TypedDict key to be string literal`. This issue is well documented [here](https://github.com/python/mypy/issues/6262). The current solution is not very extensible, as adding new keys to corresponding dictionaries necessetates that the code that currently performs this comparison is updated.
