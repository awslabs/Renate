# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import toml
from typing import List
import shutil


def generate_requirements_file_for_sagemaker_training_jobs(
    toml_file: str, keys: List[str], output_path: str
):
    dependencies = []

    # create a copy of the main requirements.txt file
    shutil.copyfile("./requirements.txt", "test/integration_tests/requirements.txt")

    # Parse the .toml file
    with open(toml_file, "r") as file:
        pyproject_toml = toml.load(file)

    for key in keys:
        value = pyproject_toml["project"]["optional-dependencies"][key]
        dependencies += value

    # Write the values into the output file
    with open(f"{output_path}/requirements.txt", "a") as file:
        for dependency in dependencies:
            file.write(f"{dependency}\n")


if __name__ == "__main__":
    generate_requirements_file_for_sagemaker_training_jobs(
        "./pyproject.toml", ["avalanche", "benchmark", "dev"], "test/integration_tests"
    )
