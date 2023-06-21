#!/usr/bin/env python

import os

import setuptools


# GIT_TAG=$(git describe --tags)
def get_version():
    try:
        git_tag = os.getenv("GIT_TAG")
        version = git_tag.lstrip("v")
    except Exception:
        version = "0.0.0"

    return version


if __name__ == "__main__":
    # Use the get_version() function to retrieve the version
    version = get_version()

    # Rest of your setup.py code...
    setuptools.setup(
        name="latentis",
        version=version,
        # Include other relevant setup configurations
        # ...
    )
