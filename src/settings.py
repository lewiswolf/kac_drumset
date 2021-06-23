'''
This file is used to define and configure the project settings.
'''

from typing import TypedDict

# type declarations
class Settings(TypedDict):
    test: int

# the configurable object
settings: Settings = {
	'test': 100
}
