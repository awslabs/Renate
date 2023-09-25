# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/awslabs/Renate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                              |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| src/renate/\_\_init\_\_.py                                        |       10 |        0 |    100% |           |
| src/renate/benchmark/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| src/renate/benchmark/datasets/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| src/renate/benchmark/datasets/base.py                             |        9 |        0 |    100% |           |
| src/renate/benchmark/datasets/nlp\_datasets.py                    |      102 |       30 |     71% |25, 87, 89, 113-117, 139, 198, 202, 209, 217-218, 222-272 |
| src/renate/benchmark/datasets/vision\_datasets.py                 |      136 |       76 |     44% |47-54, 58-59, 70-74, 78-99, 152, 164, 244-249, 260-265, 269-290, 361-376, 387-391, 395-404 |
| src/renate/benchmark/datasets/wild\_time\_data.py                 |       33 |       18 |     45% |64-78, 86-103 |
| src/renate/benchmark/experiment\_config.py                        |      144 |        7 |     95% |105-110, 114-115, 369, 428 |
| src/renate/benchmark/experimentation.py                           |      106 |        5 |     95% |49, 228, 298, 422-425 |
| src/renate/benchmark/models/\_\_init\_\_.py                       |        5 |        0 |    100% |           |
| src/renate/benchmark/models/base.py                               |       41 |        1 |     98% |        59 |
| src/renate/benchmark/models/l2p.py                                |      122 |       10 |     92% |63, 65, 104-105, 206, 214, 361-363, 372, 374 |
| src/renate/benchmark/models/mlp.py                                |       20 |        1 |     95% |        68 |
| src/renate/benchmark/models/resnet.py                             |       38 |        0 |    100% |           |
| src/renate/benchmark/models/transformer.py                        |       22 |        0 |    100% |           |
| src/renate/benchmark/models/vision\_transformer.py                |       43 |        1 |     98% |        42 |
| src/renate/benchmark/scenarios.py                                 |      183 |        9 |     95% |298, 306-307, 380-387 |
| src/renate/cli/parsing\_functions.py                              |      262 |       59 |     77% |54, 68-69, 71-72, 74-75, 77-78, 80-88, 90-104, 106-107, 109-110, 112-113, 115-116, 118-119, 121-124, 126-129, 131-134, 141, 372, 474, 487-488, 493-494, 499, 514, 518-519, 546-560, 565-587, 592-631, 636-708, 713, 727, 740, 811, 862, 890, 898, 968 |
| src/renate/cli/run\_training.py                                   |       65 |        2 |     97% |   96, 191 |
| src/renate/data/\_\_init\_\_.py                                   |        3 |        0 |    100% |           |
| src/renate/data/data\_module.py                                   |       75 |        5 |     93% |70, 75, 105, 160, 175 |
| src/renate/data/datasets.py                                       |       94 |        3 |     97% |37, 85, 100 |
| src/renate/defaults.py                                            |      106 |        1 |     99% |       126 |
| src/renate/evaluation/\_\_init\_\_.py                             |        0 |        0 |    100% |           |
| src/renate/evaluation/evaluator.py                                |       57 |        2 |     96% |   84, 154 |
| src/renate/evaluation/metrics/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| src/renate/evaluation/metrics/classification.py                   |       25 |        0 |    100% |           |
| src/renate/evaluation/metrics/performance\_regression\_metrics.py |       68 |        0 |    100% |           |
| src/renate/memory/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| src/renate/memory/buffer.py                                       |      184 |       11 |     94% |118, 125, 127, 143, 194, 198, 207, 209, 215, 217, 321 |
| src/renate/memory/storage.py                                      |       84 |       45 |     46% |18-25, 35, 38, 41, 44, 65-73, 77-96, 100-107, 111, 115-130, 133-136 |
| src/renate/models/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| src/renate/models/layers/\_\_init\_\_.py                          |        1 |        0 |    100% |           |
| src/renate/models/layers/cn.py                                    |       11 |        4 |     64% |41-44, 47, 59 |
| src/renate/models/prediction\_strategies.py                       |       14 |        5 |     64% | 13, 18-21 |
| src/renate/models/renate\_module.py                               |       87 |       23 |     74% |111, 121, 133, 176-196, 201-205, 215-216, 221, 257-262, 265 |
| src/renate/shift/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| src/renate/shift/detector.py                                      |       39 |        4 |     90% |39, 43, 109, 112 |
| src/renate/shift/kernels.py                                       |       19 |        0 |    100% |           |
| src/renate/shift/ks\_detector.py                                  |       11 |        0 |    100% |           |
| src/renate/shift/mmd\_detectors.py                                |       14 |        0 |    100% |           |
| src/renate/shift/mmd\_helpers.py                                  |       30 |        0 |    100% |           |
| src/renate/training/\_\_init\_\_.py                               |        2 |        0 |    100% |           |
| src/renate/training/training.py                                   |      200 |       30 |     85% |192, 237-259, 408-409, 501, 588, 624-625, 659-678 |
| src/renate/types.py                                               |        3 |        0 |    100% |           |
| src/renate/updaters/\_\_init\_\_.py                               |        0 |        0 |    100% |           |
| src/renate/updaters/avalanche/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| src/renate/updaters/avalanche/learner.py                          |       76 |        2 |     97% |  142, 152 |
| src/renate/updaters/avalanche/model\_updater.py                   |      134 |        8 |     94% |53, 64, 123, 125, 359-364, 429-435 |
| src/renate/updaters/avalanche/plugins.py                          |       25 |        1 |     96% |        43 |
| src/renate/updaters/experimental/\_\_init\_\_.py                  |        0 |        0 |    100% |           |
| src/renate/updaters/experimental/er.py                            |      165 |       13 |     92% |76, 166-167, 185, 197, 650-661, 735-747, 824-839, 922-943 |
| src/renate/updaters/experimental/fine\_tuning.py                  |       15 |        0 |    100% |           |
| src/renate/updaters/experimental/gdumb.py                         |       34 |        2 |     94% |   137-143 |
| src/renate/updaters/experimental/joint.py                         |       46 |        2 |     96% |   126-130 |
| src/renate/updaters/experimental/l2p.py                           |       73 |       33 |     55% |56-60, 95-137, 175-181, 251-259 |
| src/renate/updaters/experimental/offline\_er.py                   |       73 |       13 |     82% |42, 75-82, 109, 116, 126-130, 182-189 |
| src/renate/updaters/experimental/repeated\_distill.py             |       70 |        0 |    100% |           |
| src/renate/updaters/learner.py                                    |      199 |        8 |     96% |164, 168, 178-179, 235-236, 477, 521 |
| src/renate/updaters/learner\_components/\_\_init\_\_.py           |        0 |        0 |    100% |           |
| src/renate/updaters/learner\_components/component.py              |       22 |        1 |     95% |        46 |
| src/renate/updaters/learner\_components/losses.py                 |      129 |       49 |     62% |33, 47, 87-90, 131, 135-136, 145-183, 194-200, 265-275 |
| src/renate/updaters/learner\_components/reinitialization.py       |       23 |        1 |     96% |        17 |
| src/renate/updaters/model\_updater.py                             |      174 |        8 |     95% |116, 183-186, 201-202, 336 |
| src/renate/utils/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| src/renate/utils/avalanche.py                                     |       87 |        1 |     99% |        27 |
| src/renate/utils/config\_spaces.py                                |       15 |       15 |      0% |      3-75 |
| src/renate/utils/deepspeed.py                                     |       42 |       28 |     33% |21-33, 40-42, 69-100 |
| src/renate/utils/distributed\_strategies.py                       |       22 |        2 |     91% |    34, 45 |
| src/renate/utils/file.py                                          |      144 |       70 |     51% |22, 27-28, 39, 72, 99-107, 127, 131, 172-173, 187-194, 213-225, 238-244, 264-276, 286-287, 292-293, 305-314, 330-331, 354 |
| src/renate/utils/hf\_utils.py                                     |       30 |        1 |     97% |        47 |
| src/renate/utils/misc.py                                          |       14 |        0 |    100% |           |
| src/renate/utils/module.py                                        |       64 |        3 |     95% |94-98, 153 |
| src/renate/utils/optimizer.py                                     |       12 |        0 |    100% |           |
| src/renate/utils/pytorch.py                                       |      103 |        4 |     96% |35-36, 79, 87 |
| src/renate/utils/syne\_tune.py                                    |       46 |        5 |     89% |38-39, 46, 80, 88 |
|                                                         **TOTAL** | **4309** |  **622** | **86%** |           |


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