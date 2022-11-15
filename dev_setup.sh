#!/bin/sh

# update location of Git hooks from default (.git/hooks) to the versioned folder .github/hooks
git config core.hooksPath ".github/hooks"
