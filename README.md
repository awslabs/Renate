# Coverage data

This branch is just here to hold coverage data. It's part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action.

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/awslabs/Renate/python-coverage-comment-action-data/badge.svg)](https://github.com/awslabs/Renate/tree/python-coverage-comment-action-data)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/awslabs/Renate/python-coverage-comment-action-data/endpoint.json)](https://github.com/awslabs/Renate/tree/python-coverage-comment-action-data)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fawslabs%2FRenate%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://github.com/awslabs/Renate/tree/python-coverage-comment-action-data)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.