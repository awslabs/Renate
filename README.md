# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/awslabs/Renate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                              |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| src/renate/\_\_init\_\_.py                                        |       10 |        0 |    100% |           |
| src/renate/benchmark/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| src/renate/benchmark/datasets/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| src/renate/benchmark/datasets/nlp\_datasets.py                    |       26 |       10 |     62% |53, 61-71, 75-78 |
| src/renate/benchmark/datasets/vision\_datasets.py                 |      108 |       68 |     37% |45-52, 56-57, 68-72, 76-97, 150, 157-162, 170-178, 238-243, 254-260, 265, 269-292 |
| src/renate/benchmark/experiment\_config.py                        |       71 |        2 |     97% |     79-80 |
| src/renate/benchmark/experimentation.py                           |      104 |        6 |     94% |47, 65, 202, 259, 371-374 |
| src/renate/benchmark/models/\_\_init\_\_.py                       |        4 |        0 |    100% |           |
| src/renate/benchmark/models/mlp.py                                |       30 |        1 |     97% |        62 |
| src/renate/benchmark/models/resnet.py                             |       47 |        0 |    100% |           |
| src/renate/benchmark/models/vision\_transformer.py                |       44 |        1 |     98% |        91 |
| src/renate/benchmark/scenarios.py                                 |       93 |        1 |     99% |        58 |
| src/renate/cli/parsing\_functions.py                              |      149 |       59 |     60% |51-94, 96, 109, 246, 251, 267-279, 284-305, 310-350, 355-434, 439, 470, 488, 500 |
| src/renate/cli/run\_training.py                                   |       77 |        1 |     99% |       250 |
| src/renate/data/\_\_init\_\_.py                                   |        3 |        0 |    100% |           |
| src/renate/data/data\_module.py                                   |       66 |        5 |     92% |67, 72, 90, 145, 160 |
| src/renate/data/datasets.py                                       |       50 |        1 |     98% |        36 |
| src/renate/defaults.py                                            |       95 |        0 |    100% |           |
| src/renate/evaluation/\_\_init\_\_.py                             |        0 |        0 |    100% |           |
| src/renate/evaluation/evaluator.py                                |       58 |        2 |     97% |   76, 141 |
| src/renate/evaluation/metrics/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| src/renate/evaluation/metrics/classification.py                   |       22 |        0 |    100% |           |
| src/renate/evaluation/metrics/performance\_regression\_metrics.py |       68 |        0 |    100% |           |
| src/renate/evaluation/metrics/utils.py                            |       13 |        0 |    100% |           |
| src/renate/memory/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| src/renate/memory/buffer.py                                       |      155 |       10 |     94% |97, 104, 109, 114, 161, 186, 202, 205, 208, 211 |
| src/renate/models/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| src/renate/models/layers/\_\_init\_\_.py                          |        1 |        0 |    100% |           |
| src/renate/models/layers/cn.py                                    |       11 |        4 |     64% |41-44, 47, 59 |
| src/renate/models/renate\_module.py                               |       75 |       17 |     77% |104, 114, 126, 168-188, 193-197, 207-208, 213 |
| src/renate/tuning/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| src/renate/tuning/config\_spaces.py                               |       15 |       15 |      0% |      3-75 |
| src/renate/tuning/tuning.py                                       |      191 |       29 |     85% |167, 204-221, 370-371, 545, 582-583, 616-634 |
| src/renate/updaters/\_\_init\_\_.py                               |        0 |        0 |    100% |           |
| src/renate/updaters/experimental/\_\_init\_\_.py                  |        0 |        0 |    100% |           |
| src/renate/updaters/experimental/er.py                            |      237 |       66 |     72% |73, 173-174, 197, 209, 309-313, 363-370, 441-458, 574-604, 722-740, 801-820, 884-906, 976-1004 |
| src/renate/updaters/experimental/gdumb.py                         |       34 |        2 |     94% |   117-130 |
| src/renate/updaters/experimental/joint.py                         |       40 |        2 |     95% |   110-121 |
| src/renate/updaters/experimental/offline\_er.py                   |       62 |       10 |     84% |47, 70, 92, 103-107, 157-171 |
| src/renate/updaters/experimental/repeated\_distill.py             |       68 |        0 |    100% |           |
| src/renate/updaters/learner.py                                    |      180 |        1 |     99% |       154 |
| src/renate/updaters/learner\_components/\_\_init\_\_.py           |        0 |        0 |    100% |           |
| src/renate/updaters/learner\_components/component.py              |       27 |        1 |     96% |        42 |
| src/renate/updaters/learner\_components/losses.py                 |      155 |       69 |     55% |55, 69, 111-114, 188, 192-193, 202-240, 243-244, 247-248, 259-265, 365-383, 395-399, 404-417, 422-427, 430-435, 438-443, 446-451 |
| src/renate/updaters/learner\_components/reinitialization.py       |       33 |       10 |     70% |17, 55-59, 62-65, 68-69 |
| src/renate/updaters/model\_updater.py                             |      127 |        1 |     99% |       228 |
| src/renate/utils/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| src/renate/utils/file.py                                          |      120 |       61 |     49% |21, 26-27, 38, 62, 87-91, 95, 112, 116, 123-124, 138-145, 164-176, 189-195, 211-218, 228-229, 234-235, 247-253, 269-270 |
| src/renate/utils/module.py                                        |       42 |        2 |     95% |   83, 101 |
| src/renate/utils/optimizer.py                                     |       17 |        0 |    100% |           |
| src/renate/utils/pytorch.py                                       |       28 |        2 |     93% |     32-33 |
| src/renate/utils/syne\_tune.py                                    |       46 |        5 |     89% |38-39, 46, 80, 88 |
|                                                         **TOTAL** | **2808** |  **464** | **83%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/awslabs/Renate/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/awslabs/Renate/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/awslabs/Renate/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/awslabs/Renate/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fawslabs%2FRenate%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/awslabs/Renate/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.