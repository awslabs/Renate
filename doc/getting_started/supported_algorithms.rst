Supported Algorithms
********************

Renate provides implementations of various continual learning methods. The following table provides
an overview with links to the documentation, and a short description. When initiating model updates
using Renate (see :doc:`how_to_run_training`), a method may be selected using the shorthand
provided below.

.. list-table:: Title
   :header-rows: 1

   * - Shorthand
     - Method
     - Description
   * - ``"ER"``
     - :py:class:`ExperienceReplayLearner <renate.updaters.experimental.er.ExperienceReplayLearner>`
     - The most basic replay-based method. The model is finetuned using minibatches combining new data and points sampled from a rehearsal memory. The memory is updated after each minibatch.
   * - ``"DER"``
     - :py:class:`DarkExperienceReplayLearner <renate.updaters.experimental.er.DarkExperienceReplayLearner>`
     - A version of experience replay which augments the loss by a distillation term using logits produced by previous model states.
   * - ``"POD-ER"``
     - :py:class:`PooledOutputDistillationExperienceReplayLearner <renate.updaters.experimental.er.PooledOutputDistillationExperienceReplayLearner>`
     - Experience replay with distillation terms for intermediate layer activations.
   * - ``"CLS-ER"``
     - :py:class:`CLSExperienceReplayLearner <renate.updaters.experimental.er.CLSExperienceReplayLearner>`
     - A version of experience replay that maintains to copies of the model: A "fast" and a "slow" learner.
   * - ``"Super-ER"``
     - :py:class:`SuperExperienceReplayLearner <renate.updaters.experimental.er.SuperExperienceReplayLearner>`
     - An experimental method combining various of the ER variants listed above.
   * - ``"OfflineER"``
     - :py:class:`OfflineExperienceReplayLearner <renate.updaters.experimental.offline_er.OfflineExperienceReplayLearner>`
     - An offline version of experience replay, where the rehearsal memory is only updated at the end of training.
   * - ``"RD"``
     - :py:class:`RepeatedDistillationLearner <renate.updaters.experimental.repeated_distill.RepeatedDistillationLearner>`
     - A distillation-based method inspired by Deep Model Consolidation (DMC). An expert model is trained on the new data and then combined with the previous model state in a distillation phase.
   * - ``"GDumb"``
     - :py:class:`GDumbLearner <renate.updaters.experimental.gdumb.GDumbLearner>`
     - A strong baseline that trains the model from scratch on a memory, which is maintained using a greedy class-balancing strategy.
   * - ``"Joint"``
     - :py:class:`JointLearner <renate.updaters.experimental.joint.JointLearner>`
     - Retraining from scratch on all data seen so far. Used as an "upper bound" in experiments, inefficient for practical use.
