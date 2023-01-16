Supported Algorithms
********************

Renate provides implementations of various continual learning methods. The following table provides
an overview with links to the documentation, and a short description. When initiating model updates
using Renate (e.g., using :py:func:`renate.tuning.tuning.execute_tuning_job`; see
:doc:`how_to_run_training`), a method may be selected using the shorthand provided below.

.. list-table:: Supported Algorithms
   :header-rows: 1

   * - Shorthand
     - Implementation
     - Description
   * - ``"ER"``
     - :py:class:`ExperienceReplayLearner <renate.updaters.experimental.er.ExperienceReplayLearner>`
     - A simple replay-based method, where the model is finetuned using minibatches combining new data and points sampled from a rehearsal memory. The memory is updated after each minibatch. [`Paper <https://arxiv.org/abs/1902.10486>`__]
   * - ``"DER"``
     - :py:class:`DarkExperienceReplayLearner <renate.updaters.experimental.er.DarkExperienceReplayLearner>`
     - A version of experience replay which augments the loss by a distillation term using logits produced by previous model states. The implementation also includes the DER++ variant of the algorithm. [`Paper <https://arxiv.org/abs/2004.07211>`__]
   * - ``"Super-ER"``
     - :py:class:`SuperExperienceReplayLearner <renate.updaters.experimental.er.SuperExperienceReplayLearner>`
     - An experimental method combining various of the ER variants listed above.
   * - ``"OfflineER"``
     - :py:class:`OfflineExperienceReplayLearner <renate.updaters.experimental.offline_er.OfflineExperienceReplayLearner>`
     - An offline version of experience replay, where the rehearsal memory is only updated at the end of training.
   * - ``"RD"``
     - :py:class:`RepeatedDistillationLearner <renate.updaters.experimental.repeated_distill.RepeatedDistillationLearner>`
     - A distillation-based method inspired by (but not identical to) Deep Model Consolidation. An expert model is trained on the new data and then combined with the previous model state in a distillation phase. [`DMC Paper <https://arxiv.org/abs/1903.07864>`__]
   * - ``"GDumb"``
     - :py:class:`GDumbLearner <renate.updaters.experimental.gdumb.GDumbLearner>`
     - A strong baseline that trains the model from scratch on a memory, which is maintained using a greedy class-balancing strategy. [`Paper <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf>`__]
   * - ``"Joint"``
     - :py:class:`JointLearner <renate.updaters.experimental.joint.JointLearner>`
     - Retraining from scratch on all data seen so far. Used as an "upper bound" in experiments, inefficient for practical use.
   * - ``"FineTuning"``
     - :py:class:`Learner <renate.updaters.experimental.learner.Learner>`
     - Fine-tuning the existing model using the new data without any sort of mitigation for forgetting. Users as "lower bound" baseline in the experiments.
