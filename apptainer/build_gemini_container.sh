#!/bin/bash

module load apptainer


# Go to home
cd ~/

# container version
version="0.33.1"

apptainer build --sandbox "gemini_sandbox_${version}/" "docker://us-docker.pkg.dev/gemini-code-dev/gemini-cli/sandbox:${version}"
