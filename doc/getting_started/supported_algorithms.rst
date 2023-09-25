Supported Algorithms
********************

Renate provides implementations of various continual learning methods. The following table provides
an overview with links to the documentation, and a short description. When initiating model updates
using Renate (e.g., using :py:func:`~renate.training.training.run_training_job`; see
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
   * - ``"Offline-ER"``
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
     - This method retrains a randomly initialized model each time from scratch on all data seen so far. Used as "upper bound" in experiments, inefficient for practical use.
   * - ``"FineTuning"``
     - :py:class:`Learner <renate.updaters.learner.Learner>`
     - A simple method which trains the current model on only the new data without any sort of mitigation for forgetting. Used as "lower bound" baseline in experiments.
   * - ``"LearningToPrompt"`` 
     - :py:class:`LearningToPromptLearner <renate.updaters.experimental.l2p.LearningToPromptLearner>`
     - A class that implements a Learning to Prompt method for ViTs. The methods trains only the input prompts that are sampled from a prompt pool in an input dependent fashion.
   * - ``"LearningToPromptReplay"`` 
     - :py:class:`LearningToPromptLearner <renate.updaters.experimental.l2p.LearningToPromptReplayLearner>`
     - A class that extends the Learning to Prompt method to use a memory replay method like "Offline-ER"
   * - ``"Avalanche-ER"``
     - :py:class:`AvalancheReplayLearner <renate.updaters.avalanche.learner.AvalancheReplayLearner>`
     - A wrapper which gives access to Experience Replay as implemented in the Avalanche library. This method is the equivalent to our Offline-ER.
   * - ``"Avalanche-EWC"``
     - :py:class:`AvalancheEWCLearner <renate.updaters.avalanche.learner.AvalancheEWCLearner>`
     - A wrapper which gives access to Elastic Weight Consolidation as implemented in the Avalanche library. EWC updates the model in such a way that the parameters after the update remain close to the parameters before the update to avoid catastrophic forgetting. [`Paper <https://arxiv.org/abs/1612.00796>`__]
   * - ``"Avalanche-LwF"``
     - :py:class:`AvalancheLwFLearner <renate.updaters.avalanche.learner.AvalancheLwFLearner>`
     - A wrapper which gives access to Learning without Forgetting as implemented in the Avalanche library. LwF does not require to retain old data. It assumes that each new data chunk is its own task. A common backbone is shared across all task and each task has its own prediction head. [`Paper <https://arxiv.org/abs/1606.09282>`__]
   * - ``"Avalanche-iCaRL"``
     - :py:class:`AvalancheICaRLLearner <renate.updaters.avalanche.learner.AvalancheICaRLLearner>`
     - A wrapper which gives access to iCaRL as implemented in the Avalanche library. This method is limited to class-incremental learning and combines knowledge distillation with nearest neighbors classification. [`Paper <https://arxiv.org/abs/1611.07725>`__]
