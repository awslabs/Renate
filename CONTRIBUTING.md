# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.

## Questions and Discussion Topics

Questions and discussion topics can be proposed by using GitHub [issue tracker](https://github.com/awslabs/renate/issues)
and tagging the issue with `[discussion]` in the title.

## Reporting Bugs/Feature Requests

We welcome you to use the GitHub [issue tracker](https://github.com/awslabs/renate/issues/new/choose) to report bugs or suggest features.

When filing an issue, please check [existing open](https://github.com/awslabs/renate/issues), or [recently closed](https://github.com/awslabs/renate/issues?utf8=%E2%9C%93&q=is%3Aissue%20is%3Aclosed%20), issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or short series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment 

## Contributing via Pull Requests

Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the `dev` branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. If you want to contribute some major changes, you open an issue and discuss the matter in advance to better coordinate and use your time efficiently.

To send us a pull request, please:
0. Fork the repository and clone it locally.
1. Run `bash ./dev_setup.sh` to setup the git pre-commit hooks
2. Modify the source and format it with [black](https://black.readthedocs.io/en/stable/) (this optional since black will run when you commit but running beforehand helps to avoid issues)
3. Ensure local tests pass by executing `pytest`.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

## Finding Contributions to Work On

Looking at the existing issues is a great way to find something to contribute on: issues labeled with
['good first issue'](https://github.com/awslabs/renate/issues/labels/good%20first%20issue)
are a great place to start.

## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.

## Security Issues Notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.

## Licensing

This project is released under the Apache 2.0 license. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.
