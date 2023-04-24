# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/awslabs/Renate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                              |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| src/renate/\_\_init\_\_.py                                        |       10 |        0 |    100% |           |
| src/renate/benchmark/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| src/renate/benchmark/datasets/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| src/renate/benchmark/datasets/nlp\_datasets.py                    |       54 |        6 |     89% |22, 83, 85, 98-102, 124 |
| src/renate/benchmark/datasets/vision\_datasets.py                 |      108 |       68 |     37% |45-52, 56-57, 68-72, 76-97, 150, 157-162, 170-178, 238-243, 254-260, 265, 269-292 |
| src/renate/benchmark/experiment\_config.py                        |       67 |        3 |     96% | 70, 82-83 |
| src/renate/benchmark/experimentation.py                           |      104 |        6 |     94% |47, 65, 206, 265, 376-379 |
| src/renate/benchmark/models/\_\_init\_\_.py                       |        4 |        0 |    100% |           |
| src/renate/benchmark/models/base.py                               |       40 |        1 |     98% |        61 |
| src/renate/benchmark/models/mlp.py                                |       20 |        1 |     95% |        71 |
| src/renate/benchmark/models/resnet.py                             |       36 |        0 |    100% |           |
| src/renate/benchmark/models/vision\_transformer.py                |       33 |        1 |     97% |        99 |
| src/renate/benchmark/scenarios.py                                 |      142 |       10 |     93% |58, 291, 299-300, 373-380 |
| src/renate/cli/parsing\_functions.py                              |      242 |       52 |     79% |49, 73-74, 76-77, 79-87, 89-103, 105-106, 108-109, 111-112, 114-115, 117-118, 120-123, 125-128, 130-133, 140, 331, 444, 449, 454, 458-459, 486-500, 505-527, 532-571, 576-648, 653, 667, 680, 738, 780, 808, 816, 846, 886 |
| src/renate/cli/run\_training.py                                   |       55 |        2 |     96% |   87, 144 |
| src/renate/data/\_\_init\_\_.py                                   |        3 |        0 |    100% |           |
| src/renate/data/data\_module.py                                   |       66 |        5 |     92% |67, 72, 90, 145, 160 |
| src/renate/data/datasets.py                                       |       82 |        3 |     96% |38, 86, 101 |
| src/renate/defaults.py                                            |      102 |        1 |     99% |       120 |
| src/renate/evaluation/\_\_init\_\_.py                             |        0 |        0 |    100% |           |
| src/renate/evaluation/evaluator.py                                |       58 |        2 |     97% |   76, 141 |
| src/renate/evaluation/metrics/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| src/renate/evaluation/metrics/classification.py                   |       22 |        0 |    100% |           |
| src/renate/evaluation/metrics/performance\_regression\_metrics.py |       68 |        0 |    100% |           |
| src/renate/evaluation/metrics/utils.py                            |       13 |        0 |    100% |           |
| src/renate/memory/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| src/renate/memory/buffer.py                                       |      187 |       11 |     94% |117, 124, 126, 149, 209, 213, 222, 224, 230, 232, 336 |
| src/renate/memory/storage.py                                      |       64 |        6 |     91% |27, 30, 33, 78, 89, 112 |
| src/renate/models/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| src/renate/models/layers/\_\_init\_\_.py                          |        1 |        0 |    100% |           |
| src/renate/models/layers/cn.py                                    |       11 |        4 |     64% |41-44, 47, 59 |
| src/renate/models/prediction\_strategies.py                       |       14 |        5 |     64% | 13, 18-21 |
| src/renate/models/renate\_module.py                               |       85 |       23 |     73% |108, 118, 130, 173-193, 198-202, 212-213, 218, 255-260, 263 |
| src/renate/training/\_\_init\_\_.py                               |        2 |        0 |    100% |           |
| src/renate/training/training.py                                   |      195 |       29 |     85% |173, 211-228, 377-378, 551, 587-588, 621-639 |
| src/renate/types.py                                               |        3 |        0 |    100% |           |
| src/renate/updaters/\_\_init\_\_.py                               |        0 |        0 |    100% |           |
| src/renate/updaters/avalanche/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| src/renate/updaters/avalanche/learner.py                          |       73 |        2 |     97% |  138, 148 |
| src/renate/updaters/avalanche/model\_updater.py                   |      120 |        7 |     94% |54, 102, 104, 320-332, 386-399 |
| src/renate/updaters/avalanche/plugins.py                          |       25 |        1 |     96% |        43 |
| src/renate/updaters/experimental/\_\_init\_\_.py                  |        0 |        0 |    100% |           |
| src/renate/updaters/experimental/er.py                            |      235 |       66 |     72% |75, 180-181, 199, 211, 313-317, 367-374, 445-462, 578-608, 729-747, 810-829, 895-917, 989-1017 |
| src/renate/updaters/experimental/fine\_tuning.py                  |       11 |        0 |    100% |           |
| src/renate/updaters/experimental/gdumb.py                         |       36 |        2 |     94% |   125-138 |
| src/renate/updaters/experimental/joint.py                         |       51 |        2 |     96% |   129-140 |
| src/renate/updaters/experimental/offline\_er.py                   |       64 |       10 |     84% |48, 72, 93, 104-108, 159-173 |
| src/renate/updaters/experimental/repeated\_distill.py             |       67 |        0 |    100% |           |
| src/renate/updaters/learner.py                                    |      200 |        1 |     99% |       155 |
| src/renate/updaters/learner\_components/\_\_init\_\_.py           |        0 |        0 |    100% |           |
| src/renate/updaters/learner\_components/component.py              |       27 |        1 |     96% |        42 |
| src/renate/updaters/learner\_components/losses.py                 |      155 |       69 |     55% |55, 69, 111-114, 188, 192-193, 202-240, 243-244, 247-248, 259-265, 365-375, 387-391, 396-409, 414-419, 422-427, 430-435, 438-443 |
| src/renate/updaters/learner\_components/reinitialization.py       |       33 |       10 |     70% |17, 55-59, 62-65, 68-69 |
| src/renate/updaters/model\_updater.py                             |      144 |        4 |     97% |107, 140-141, 253 |
| src/renate/utils/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| src/renate/utils/avalanche.py                                     |       85 |        0 |    100% |           |
| src/renate/utils/config\_spaces.py                                |       15 |       15 |      0% |      3-75 |
| src/renate/utils/file.py                                          |      120 |       61 |     49% |21, 26-27, 38, 62, 87-91, 95, 112, 116, 123-124, 138-145, 164-176, 189-195, 211-218, 228-229, 234-235, 247-253, 269-270 |
| src/renate/utils/module.py                                        |       42 |        2 |     95% |   83, 101 |
| src/renate/utils/optimizer.py                                     |       17 |        0 |    100% |           |
| src/renate/utils/pytorch.py                                       |       40 |        4 |     90% |34-35, 78, 86 |
| src/renate/utils/syne\_tune.py                                    |       46 |        5 |     89% |38-39, 46, 80, 88 |
|                                                         **TOTAL** | **3501** |  **501** | **86%** |           |


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