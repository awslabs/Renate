# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from renate.models import RenateModule
from renate.types import NestedTensors
from renate.updaters.learner_components.component import Component


class WeightedLossComponent(Component, ABC):
    """The abstract class implementing a weighted loss function.

    This is an abstract class from which each other loss should inherit from.

    Args:
        weight: A scaling coefficient which should scale the loss which gets returned.
        sample_new_memory_batch: Whether a new batch of data should be sampled from the memory
            buffer when the loss is calculated.
    """

    def _verify_attributes(self) -> None:
        """Verify if attributes have valid values."""
        super()._verify_attributes()
        assert self.weight >= 0, "Weight must be larger than 0."

    def loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[Tuple[NestedTensors, torch.Tensor], Dict[str, torch.Tensor]],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        if self.weight == 0:
            return torch.tensor(0.0)
        return self._loss(
            outputs_memory=outputs_memory,
            batch_memory=batch_memory,
            intermediate_representation_memory=intermediate_representation_memory,
        )

    @abstractmethod
    def _loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[Tuple[NestedTensors, torch.Tensor], Dict[str, torch.Tensor]],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        pass


class WeightedCustomLossComponent(WeightedLossComponent):
    """Adds a (weighted) user-provided custom loss contribution.

    Args:
        loss_fn: The loss function to apply.
        weight: A scaling coefficient which should scale the loss which gets returned.
        sample_new_memory_batch: Whether a new batch of data should be sampled from the memory
            buffer when the loss is calculated.
    """

    def __init__(self, loss_fn: Callable, weight: float, sample_new_memory_batch: bool) -> None:
        super().__init__(weight=weight, sample_new_memory_batch=sample_new_memory_batch)
        self._loss_fn = loss_fn

    def _loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[Tuple[NestedTensors, torch.Tensor], Dict[str, torch.Tensor]],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Returns user-provided loss evaluated on memory batch."""
        (_, targets_memory), _ = batch_memory
        return self.weight * self._loss_fn(outputs_memory, targets_memory)


class WeightedMeanSquaredErrorLossComponent(WeightedLossComponent):
    """Mean squared error between the current and previous logits computed with respect to the
    memory sample.
    """

    def _loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[Tuple[NestedTensors, torch.Tensor], Dict[str, torch.Tensor]],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Mean-squared error between current and previous logits on memory."""
        logits = outputs_memory
        _, meta_data = batch_memory
        previous_logits = meta_data["outputs"]
        return self.weight * F.mse_loss(logits, previous_logits, reduction="mean")


class WeightedPooledOutputDistillationLossComponent(WeightedLossComponent):
    """Pooled output feature distillation with respect to intermediate network features.

    As described in: Douillard, Arthur, et al. "Podnet: Pooled outputs distillation for small-tasks
    incremental learning." European Conference on Computer Vision. Springer, Cham, 2020.

    Given the intermediate representations collected at different parts of the network, minimise
    their Euclidean distance with respect to the cached representation. There are different
    `distillation_type`s trading-off plasticity and stability of the resultant representations.
    `normalize` enables the user to normalize the resultant feature representations to ensure that
    they are less affected by their magnitude.

    Args:
        weight: Scaling coefficient which scales the loss with respect to all intermediate
            representations.
        sample_new_memory_batch: Whether a new batch of data should be sampled from the memory
            buffer when the loss is calculated.
        distillation_type: Which distillation type to apply with respect to all intermediate
            representations.
        normalize: Whether to normalize both the current and cached features before computing the
            Frobenius norm.
    """

    def __init__(
        self,
        weight: float,
        sample_new_memory_batch: bool,
        distillation_type: str = "spatial",
        normalize: bool = True,
    ) -> None:
        self._distillation_type = distillation_type
        super().__init__(weight=weight, sample_new_memory_batch=sample_new_memory_batch)
        self._normalize = normalize

    def _verify_attributes(self) -> None:
        """Verify if attributes have valid values."""
        super()._verify_attributes()
        if self._distillation_type not in ["pixel", "channel", "width", "height", "gap", "spatial"]:
            raise ValueError(f"Invalid distillation type: {self._distillation_type}")

    def _sum_reshape(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Sum the tensor according to specific dimension and reshape."""
        batch_size = x.shape[0]
        return x.sum(dim=dim).reshape(batch_size, -1)

    def _pod(self, features: torch.Tensor, features_memory: torch.Tensor) -> torch.Tensor:
        """Pooled output distillation with respect to intermediate and cached intermediate features.

        Args:
            features: Current intermediate features.
            features_memory: Cached intermediate features.
        """
        if features.shape != features_memory.shape:
            raise ValueError(
                "The shape of the features and the cached features should be the same: "
                f"{features.shape}, and: {features_memory.shape}"
            )

        features = features.pow(2)
        features_memory = features_memory.pow(2)

        if self._distillation_type == "channels":
            features, features_memory = self._sum_reshape(features, 1), self._sum_reshape(
                features_memory, 1
            )
        elif self._distillation_type == "width":
            features, features_memory = self._sum_reshape(features, 2), self._sum_reshape(
                features_memory, 2
            )
        elif self._distillation_type == "height":
            features, features_memory = self._sum_reshape(features, 3), self._sum_reshape(
                features_memory, 3
            )
        elif self._distillation_type == "gap":
            features = F.adaptive_avg_pool2d(features, (1, 1))[..., 0, 0]
            features_memory = F.adaptive_avg_pool2d(features_memory, (1, 1))[..., 0, 0]
        elif self._distillation_type == "spatial":
            features_h, features_memory_h = self._sum_reshape(features, 3), self._sum_reshape(
                features_memory, 3
            )
            features_w, features_memory_w = self._sum_reshape(features, 2), self._sum_reshape(
                features_memory, 2
            )
            features = torch.cat([features_h, features_w], dim=-1)
            features_memory = torch.cat([features_memory_h, features_memory_w], dim=-1)

        if self._normalize:
            features = F.normalize(features, dim=1, p=2)
            features_memory = F.normalize(features_memory, dim=1, p=2)

        return torch.frobenius_norm(features - features_memory, dim=-1).mean(dim=0)

    def _loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[Tuple[NestedTensors, torch.Tensor], Dict[str, torch.Tensor]],
        intermediate_representation_memory: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the pooled output with respect to current and cached intermediate outputs from
        memory.
        """
        loss = 0.0
        _, meta_data = batch_memory
        for n in range(len(intermediate_representation_memory)):
            features = intermediate_representation_memory[n]
            features_memory = meta_data[f"intermediate_representation_{n}"]
            loss += self._pod(features, features_memory)
        return (self.weight * loss) / len(intermediate_representation_memory)


"""
Expected:
INFO:renate.benchmark.experimentation:  Task ID  accuracy          
             Task 1    Task 2
0       1  0.999527  0.000000
1       2  0.985816  0.975514
INFO:renate.benchmark.experimentation:### Cumulative results: ###
INFO:renate.benchmark.experimentation:   Task ID  Average Accuracy  Forgetting  Forward Transfer  Backward Transfer
0        1          0.999527    0.000000          0.000000           0.000000
1        2          0.980665    0.013712         -0.001469          -0.013712

INFO:renate.benchmark.experimentation:Starting Update 2/2.
INFO:renate.training.training:Start updating the model.
WARNING:syne_tune.backend.local_backend:num_gpus_per_trial = 1 is too large, reducing to 0
INFO:renate.utils.syne_tune:Epoch 1/5
-----------------  ---------
train_accuracy     0.842789
train_base_loss    0.537081
train_loss         0.518919
train_memory_loss  0.121193
train_cls_loss     0.171997
val_accuracy       0.981197
val_loss           0.0947309
-----------------  ---------
INFO:renate.utils.syne_tune:Epoch 2/5
-----------------  ---------
train_accuracy     0.979782
train_base_loss    0.0975023
train_loss         0.149422
train_memory_loss  0.0308922
train_cls_loss     0.110681
val_accuracy       0.96782
val_loss           0.130574
-----------------  ---------
INFO:renate.utils.syne_tune:Epoch 3/5
-----------------  ---------
train_accuracy     0.989542
train_base_loss    0.0680644
train_loss         0.122773
train_memory_loss  0.0270025
train_cls_loss     0.10137
val_accuracy       0.966211
val_loss           0.141664
-----------------  ---------
INFO:renate.utils.syne_tune:Epoch 4/5
-----------------  ---------
train_accuracy     0.99268
train_base_loss    0.0535827
train_loss         0.104172
train_memory_loss  0.0262224
train_cls_loss     0.0868703
val_accuracy       0.962993
val_loss           0.151753
-----------------  ---------
INFO:renate.utils.syne_tune:Epoch 5/5
-----------------  ---------
train_accuracy     0.996166
train_base_loss    0.0399887
train_loss         0.0861045
train_memory_loss  0.024808
train_cls_loss     0.0729706
val_accuracy       0.962188
val_loss           0.16872
-----------------  ---------
WARNING:syne_tune.optimizer.schedulers.searchers.random_grid_searcher:Failed to sample a configuration not already chosen before. Exclusion list has size 1. Configuration space has size 1.
Tuning is finishing as the whole configuration space got exhausted.
--------------------
Resource summary (last result is reported):
 trial_id    status  iter  val_size            scenario_name class_groupings  num_tasks  max_epochs           model_name  num_hidden_layers  hidden_size updater optimizer  learning_rate  momentum  weight_decay  batch_size  memory_batch_size  memory_size  alpha  beta  stable_model_update_weight  plastic_model_update_weight  stable_model_update_probability  plastic_model_update_probability dataset_name  num_inputs  num_outputs                                                                      config_file  prepare_data  chunk_id      task_id  working_directory  seed accelerator  devices strategy precision  deterministic_trainer                input_state_url       metric mode  train_accuracy  train_base_loss  train_loss  train_memory_loss  train_cls_loss  val_accuracy  val_loss  step  epoch  worker-time
        0 Completed     5      0.05 ClassIncrementalScenario   [[0,1],[2,3]]          2           5 MultiLayerPerceptron                  1          200  CLS-ER      Adam           0.01       0.0           0.0         256                256          500    0.5   0.1                       0.999                        0.999                              0.7                               0.9        MNIST         784           10 /Users/marwistu/PycharmProjects/renate/src/renate/benchmark/experiment_config.py         False         1 default_task renate_working_dir     0        auto        1      ddp        32                   True renate_working_dir/input_state val_accuracy  max        0.996166         0.039989    0.086105           0.024808        0.072971      0.962188   0.16872     4      5    25.648655
0 trials running, 1 finished (1 until the end), 55.10s wallclock-time

val_accuracy: best 0.9811965823173523 for trial-id 0
--------------------
INFO:renate.training.training:All training is completed. Saving state...
Multi Objective Optimization dependencies are not imported since dependencies are missing. You can install them with
   pip install 'syne-tune[moo]'
or (for everything)
   pip install 'syne-tune[extra]'
memory_loss 0.005096937995404005
cls_loss 0.011621142737567425
memory_loss 0.20032952725887299
cls_loss 0.10943064838647842
memory_loss 0.4728430211544037
cls_loss 0.19654035568237305
memory_loss 0.5549020767211914
cls_loss 0.23473529517650604
memory_loss 0.5682501792907715
cls_loss 0.2689952552318573
memory_loss 0.5233345627784729
cls_loss 0.35401710867881775
memory_loss 0.4623183310031891
cls_loss 0.4478920102119446
memory_loss 0.4077729284763336
cls_loss 0.4824923574924469
memory_loss 0.2875109612941742
cls_loss 0.46580180525779724
memory_loss 0.19320343434810638
cls_loss 0.3559134304523468
memory_loss 0.12103022634983063
cls_loss 0.223443865776062
memory_loss 0.08937011659145355
cls_loss 0.12197346985340118
memory_loss 0.07794462889432907
cls_loss 0.12221729010343552
memory_loss 0.06015016511082649
cls_loss 0.12211334705352783
memory_loss 0.06662067025899887
cls_loss 0.15294979512691498
memory_loss 0.06573906540870667
cls_loss 0.19558070600032806
memory_loss 0.09320586174726486
cls_loss 0.18661770224571228
memory_loss 0.07525616884231567
cls_loss 0.20424316823482513
memory_loss 0.04947834089398384
cls_loss 0.17002616822719574
memory_loss 0.03639943152666092
cls_loss 0.1626826971769333
memory_loss 0.04118342697620392
cls_loss 0.15303920209407806
memory_loss 0.045653391629457474
cls_loss 0.16208966076374054
memory_loss 0.0655428096652031
cls_loss 0.1314183920621872
memory_loss 0.06860221922397614
cls_loss 0.1377023160457611
memory_loss 0.05714713782072067
cls_loss 0.14713464677333832
memory_loss 0.052203770726919174
cls_loss 0.1424962729215622
memory_loss 0.04310104623436928
cls_loss 0.11800438165664673
memory_loss 0.03671986982226372
cls_loss 0.11984870582818985
memory_loss 0.04364239051938057
cls_loss 0.13547246158123016
memory_loss 0.031782545149326324
cls_loss 0.1348702609539032
memory_loss 0.03158913180232048
cls_loss 0.15066979825496674
memory_loss 0.031875129789114
cls_loss 0.13376343250274658
memory_loss 0.03779058903455734
cls_loss 0.12336807698011398
memory_loss 0.043563663959503174
cls_loss 0.10693468153476715
memory_loss 0.04872112348675728
cls_loss 0.12007945030927658
memory_loss 0.04167334362864494
cls_loss 0.09854646027088165
memory_loss 0.039086807519197464
cls_loss 0.10811477154493332
memory_loss 0.04313311353325844
cls_loss 0.10291007906198502
memory_loss 0.04317881539463997
cls_loss 0.10963457822799683
memory_loss 0.03929790481925011
cls_loss 0.13826455175876617
memory_loss 0.030823025852441788
cls_loss 0.12586431205272675
memory_loss 0.033869337290525436
cls_loss 0.10837904363870621
memory_loss 0.03395329788327217
cls_loss 0.1201445460319519
memory_loss 0.028043514117598534
cls_loss 0.11138121038675308
memory_loss 0.03073180466890335
cls_loss 0.11044913530349731
[tune-metric]: {"train_accuracy": 0.8427886962890625, "train_base_loss": 0.537081241607666, "train_loss": 0.5189193487167358, "train_memory_loss": 0.12119260430335999, "train_cls_loss": 0.1719970405101776, "val_accuracy": 0.9811965823173523, "val_loss": 0.0947309210896492, "step": 0, "epoch": 1, "st_worker_timestamp": 1690189409.3551612, "st_worker_time": 6.5192155450000016, "st_worker_iter": 0}
memory_loss 0.03142643719911575
cls_loss 0.10047958046197891
memory_loss 0.033861204981803894
cls_loss 0.1138586774468422
memory_loss 0.04231885075569153
cls_loss 0.123374342918396
memory_loss 0.034157685935497284
cls_loss 0.1306399554014206
memory_loss 0.038097601383924484
cls_loss 0.11710162460803986
memory_loss 0.03610220551490784
cls_loss 0.12385059893131256
memory_loss 0.03376802057027817
cls_loss 0.10300033539533615
memory_loss 0.034662894904613495
cls_loss 0.10879959911108017
memory_loss 0.03256558999419212
cls_loss 0.10165603458881378
memory_loss 0.03195364028215408
cls_loss 0.12261426448822021
memory_loss 0.029570696875452995
cls_loss 0.11882586777210236
memory_loss 0.031699832528829575
cls_loss 0.11137288063764572
memory_loss 0.03638574481010437
cls_loss 0.12108957767486572
memory_loss 0.03768584877252579
cls_loss 0.1235426589846611
memory_loss 0.03137720003724098
cls_loss 0.10822445154190063
memory_loss 0.026350218802690506
cls_loss 0.11504647880792618
memory_loss 0.026217233389616013
cls_loss 0.1179238110780716
memory_loss 0.02908398024737835
cls_loss 0.11894917488098145
memory_loss 0.03407808020710945
cls_loss 0.10916395485401154
memory_loss 0.03167244419455528
cls_loss 0.1102699562907219
memory_loss 0.02921290509402752
cls_loss 0.10716251283884048
memory_loss 0.029865393415093422
cls_loss 0.10765993595123291
memory_loss 0.027536876499652863
cls_loss 0.10460250824689865
memory_loss 0.026029545813798904
cls_loss 0.11236733198165894
memory_loss 0.027880754321813583
cls_loss 0.1060718521475792
memory_loss 0.02750752866268158
cls_loss 0.11239083111286163
memory_loss 0.031068570911884308
cls_loss 0.10594209283590317
memory_loss 0.030217517167329788
cls_loss 0.10755126923322678
memory_loss 0.03067559190094471
cls_loss 0.10126985609531403
memory_loss 0.02850094623863697
cls_loss 0.10542165488004684
memory_loss 0.02729480154812336
cls_loss 0.10569441318511963
memory_loss 0.02296915277838707
cls_loss 0.11407144367694855
memory_loss 0.029397381469607353
cls_loss 0.11893775314092636
memory_loss 0.03377053141593933
cls_loss 0.11320473998785019
memory_loss 0.03236519172787666
cls_loss 0.10240794718265533
memory_loss 0.0306183360517025
cls_loss 0.10079161077737808
memory_loss 0.027243725955486298
cls_loss 0.11000652611255646
memory_loss 0.029726630076766014
cls_loss 0.10571954399347305
memory_loss 0.02860950119793415
cls_loss 0.11302683502435684
memory_loss 0.028799589723348618
cls_loss 0.11393711715936661
memory_loss 0.03302883729338646
cls_loss 0.09766394644975662
memory_loss 0.03080224245786667
cls_loss 0.1049857884645462
memory_loss 0.02948610670864582
cls_loss 0.10358121246099472
memory_loss 0.027021009474992752
cls_loss 0.1051313653588295
memory_loss 0.02748274616897106
cls_loss 0.10126461088657379
[tune-metric]: {"train_accuracy": 0.9797821640968323, "train_base_loss": 0.09750225394964218, "train_loss": 0.14942218363285065, "train_memory_loss": 0.030892150476574898, "train_cls_loss": 0.11068105697631836, "val_accuracy": 0.9678198099136353, "val_loss": 0.13057437539100647, "step": 1, "epoch": 2, "st_worker_timestamp": 1690189414.8335412, "st_worker_time": 11.997535555000002, "st_worker_iter": 1}
memory_loss 0.023708917200565338
cls_loss 0.1158745288848877
memory_loss 0.02270503155887127
cls_loss 0.12211745232343674
memory_loss 0.026006806641817093
cls_loss 0.11078055948019028
memory_loss 0.028124582022428513
cls_loss 0.10880191624164581
memory_loss 0.02971644140779972
cls_loss 0.0961754247546196
memory_loss 0.02982393652200699
cls_loss 0.09899589419364929
memory_loss 0.029273036867380142
cls_loss 0.10009286552667618
memory_loss 0.02619074657559395
cls_loss 0.11131796985864639
memory_loss 0.02603677660226822
cls_loss 0.10903707891702652
memory_loss 0.027880873531103134
cls_loss 0.11123187839984894
memory_loss 0.030767466872930527
cls_loss 0.09607623517513275
memory_loss 0.033495429903268814
cls_loss 0.09253574907779694
memory_loss 0.028926733881235123
cls_loss 0.09459688514471054
memory_loss 0.023005010560154915
cls_loss 0.10424597561359406
memory_loss 0.02125435322523117
cls_loss 0.11951201409101486
memory_loss 0.02360229194164276
cls_loss 0.11468211561441422
memory_loss 0.025408118963241577
cls_loss 0.10113594681024551
memory_loss 0.028793105855584145
cls_loss 0.0963871106505394
memory_loss 0.03000554069876671
cls_loss 0.09141623228788376
memory_loss 0.026477057486772537
cls_loss 0.1022544875741005
memory_loss 0.025196362286806107
cls_loss 0.10351502895355225
memory_loss 0.024907777085900307
cls_loss 0.10489320755004883
memory_loss 0.027984634041786194
cls_loss 0.09614207595586777
memory_loss 0.03005402907729149
cls_loss 0.10009580105543137
memory_loss 0.029243120923638344
cls_loss 0.09846150130033493
memory_loss 0.02809566631913185
cls_loss 0.09563172608613968
memory_loss 0.025886591523885727
cls_loss 0.0982494056224823
memory_loss 0.024031635373830795
cls_loss 0.10969215631484985
memory_loss 0.022867772728204727
cls_loss 0.10913436859846115
memory_loss 0.023835591971874237
cls_loss 0.10202407836914062
memory_loss 0.024767301976680756
cls_loss 0.09983525425195694
memory_loss 0.0281270332634449
cls_loss 0.09319104999303818
memory_loss 0.027996230870485306
cls_loss 0.09123041480779648
memory_loss 0.027327150106430054
cls_loss 0.09536315500736237
memory_loss 0.026326635852456093
cls_loss 0.10167019814252853
memory_loss 0.027425259351730347
cls_loss 0.09514310210943222
memory_loss 0.025691993534564972
cls_loss 0.10204102098941803
memory_loss 0.027371635660529137
cls_loss 0.09607652574777603
memory_loss 0.028437891975045204
cls_loss 0.08821220695972443
memory_loss 0.0239059217274189
cls_loss 0.09969867765903473
memory_loss 0.027263019233942032
cls_loss 0.09685896337032318
memory_loss 0.03128825128078461
cls_loss 0.10141211003065109
memory_loss 0.033528443425893784
cls_loss 0.09473077952861786
memory_loss 0.026932576671242714
cls_loss 0.08946263045072556
memory_loss 0.025419648736715317
cls_loss 0.10162526369094849
[tune-metric]: {"train_accuracy": 0.9895424842834473, "train_base_loss": 0.06806435436010361, "train_loss": 0.1227731704711914, "train_memory_loss": 0.027002541348338127, "train_cls_loss": 0.10137020796537399, "val_accuracy": 0.9662107825279236, "val_loss": 0.14166419208049774, "step": 2, "epoch": 3, "st_worker_timestamp": 1690189419.6290839, "st_worker_time": 16.793026518, "st_worker_iter": 2}
memory_loss 0.025063438341021538
cls_loss 0.1033722311258316
memory_loss 0.025233285501599312
cls_loss 0.09908288717269897
memory_loss 0.024254515767097473
cls_loss 0.09944113343954086
memory_loss 0.026736516505479813
cls_loss 0.10334336012601852
memory_loss 0.03182400390505791
cls_loss 0.08947847783565521
memory_loss 0.030263042077422142
cls_loss 0.08232385665178299
memory_loss 0.026406843215227127
cls_loss 0.08976585417985916
memory_loss 0.025020984932780266
cls_loss 0.09183184057474136
memory_loss 0.023864751681685448
cls_loss 0.09121925383806229
memory_loss 0.023834969848394394
cls_loss 0.09192675352096558
memory_loss 0.025527674704790115
cls_loss 0.09360817819833755
memory_loss 0.027827247977256775
cls_loss 0.0932571217417717
memory_loss 0.028980985283851624
cls_loss 0.08740826696157455
memory_loss 0.029161576181650162
cls_loss 0.07762821763753891
memory_loss 0.0261984970420599
cls_loss 0.08411375433206558
memory_loss 0.0258796289563179
cls_loss 0.08672080188989639
memory_loss 0.024220086634159088
cls_loss 0.09223809093236923
memory_loss 0.02422352135181427
cls_loss 0.08623208105564117
memory_loss 0.026560012251138687
cls_loss 0.08234129101037979
memory_loss 0.02602509595453739
cls_loss 0.08801987767219543
memory_loss 0.024969851598143578
cls_loss 0.08569307625293732
memory_loss 0.022627145051956177
cls_loss 0.09440828859806061
memory_loss 0.024575918912887573
cls_loss 0.08343591541051865
memory_loss 0.0255789365619421
cls_loss 0.08083441108465195
memory_loss 0.02942073345184326
cls_loss 0.08058051019906998
memory_loss 0.029328132048249245
cls_loss 0.07995720952749252
memory_loss 0.028167275711894035
cls_loss 0.08051745593547821
memory_loss 0.025289874523878098
cls_loss 0.08999417722225189
memory_loss 0.02414354309439659
cls_loss 0.09018846601247787
memory_loss 0.02708926983177662
cls_loss 0.0842500701546669
memory_loss 0.025830049067735672
cls_loss 0.08992566168308258
memory_loss 0.026203099638223648
cls_loss 0.08161565661430359
memory_loss 0.026581188663840294
cls_loss 0.08191414177417755
memory_loss 0.027509743347764015
cls_loss 0.08364943414926529
memory_loss 0.023729704320430756
cls_loss 0.09123406559228897
memory_loss 0.027930522337555885
cls_loss 0.07797911763191223
memory_loss 0.02647518366575241
cls_loss 0.07612299174070358
memory_loss 0.025520440191030502
cls_loss 0.08453754335641861
memory_loss 0.022795792669057846
cls_loss 0.09114160388708115
memory_loss 0.02525647170841694
cls_loss 0.08137119561433792
memory_loss 0.028043797239661217
cls_loss 0.07714136689901352
memory_loss 0.030877383425831795
cls_loss 0.07644788920879364
memory_loss 0.026447445154190063
cls_loss 0.07946842908859253
memory_loss 0.025793103501200676
cls_loss 0.0812065526843071
memory_loss 0.02271602302789688
cls_loss 0.09219452738761902
[tune-metric]: {"train_accuracy": 0.9926797151565552, "train_base_loss": 0.053582679480314255, "train_loss": 0.10417209565639496, "train_memory_loss": 0.026222379878163338, "train_cls_loss": 0.08687029778957367, "val_accuracy": 0.962992787361145, "val_loss": 0.15175284445285797, "step": 3, "epoch": 4, "st_worker_timestamp": 1690189423.99071, "st_worker_time": 21.154605231999998, "st_worker_iter": 3}
memory_loss 0.02131875604391098
cls_loss 0.08853524923324585
memory_loss 0.024328652769327164
cls_loss 0.08132126182317734
memory_loss 0.02787037193775177
cls_loss 0.07537757605314255
memory_loss 0.02712433785200119
cls_loss 0.07546831667423248
memory_loss 0.02529773861169815
cls_loss 0.08097022771835327
memory_loss 0.02470446564257145
cls_loss 0.08119384944438934
memory_loss 0.02453131228685379
cls_loss 0.07916950434446335
memory_loss 0.02849007397890091
cls_loss 0.07195796072483063
memory_loss 0.026922767981886864
cls_loss 0.07199811935424805
memory_loss 0.0230726208537817
cls_loss 0.07962853461503983
memory_loss 0.021758822724223137
cls_loss 0.08448648452758789
memory_loss 0.02201094664633274
cls_loss 0.0855870172381401
memory_loss 0.024102577939629555
cls_loss 0.07757371664047241
memory_loss 0.02510441280901432
cls_loss 0.07607495784759521
memory_loss 0.024494187906384468
cls_loss 0.07174258679151535
memory_loss 0.0240015871822834
cls_loss 0.07764805853366852
memory_loss 0.024043120443820953
cls_loss 0.07787082344293594
memory_loss 0.026391206309199333
cls_loss 0.07642712444067001
memory_loss 0.026196276769042015
cls_loss 0.07152962684631348
memory_loss 0.026946203783154488
cls_loss 0.06740490347146988
memory_loss 0.026636721566319466
cls_loss 0.07307334244251251
memory_loss 0.02567407488822937
cls_loss 0.07205270975828171
memory_loss 0.026562390848994255
cls_loss 0.06838282197713852
memory_loss 0.02509806863963604
cls_loss 0.07486782968044281
memory_loss 0.025397542864084244
cls_loss 0.07172515988349915
memory_loss 0.026678677648305893
cls_loss 0.07167255878448486
memory_loss 0.025431223213672638
cls_loss 0.07092078775167465
memory_loss 0.024940604344010353
cls_loss 0.06887412071228027
memory_loss 0.02534811571240425
cls_loss 0.06876903772354126
memory_loss 0.023082101717591286
cls_loss 0.0739656388759613
memory_loss 0.025102419778704643
cls_loss 0.0683949813246727
memory_loss 0.0246223583817482
cls_loss 0.06961037963628769
memory_loss 0.02443557046353817
cls_loss 0.06853964179754257
memory_loss 0.02406812272965908
cls_loss 0.07033005356788635
memory_loss 0.025684691965579987
cls_loss 0.06700237095355988
memory_loss 0.02433246374130249
cls_loss 0.06852589547634125
memory_loss 0.023646295070648193
cls_loss 0.06517907232046127
memory_loss 0.024175167083740234
cls_loss 0.06793385744094849
memory_loss 0.023594612255692482
cls_loss 0.06730979681015015
memory_loss 0.024111634120345116
cls_loss 0.07158677279949188
memory_loss 0.024095231667160988
cls_loss 0.06283599138259888
memory_loss 0.02414807304739952
cls_loss 0.06636973470449448
memory_loss 0.0226968415081501
cls_loss 0.06911434233188629
memory_loss 0.022398345172405243
cls_loss 0.07258976995944977
memory_loss 0.025690041482448578
cls_loss 0.0620834119617939
[tune-metric]: {"train_accuracy": 0.9961655735969543, "train_base_loss": 0.0399886816740036, "train_loss": 0.08610454946756363, "train_memory_loss": 0.024808043614029884, "train_cls_loss": 0.07297057658433914, "val_accuracy": 0.9621882438659668, "val_loss": 0.16871953010559082, "step": 4, "epoch": 5, "st_worker_timestamp": 1690189428.484808, "st_worker_time": 25.648654577, "st_worker_iter": 4}


Observed:

INFO:renate.benchmark.experimentation:  Task ID  accuracy          
             Task 1    Task 2
0       1  0.999527  0.000000
1       2  0.984397  0.978453
INFO:renate.benchmark.experimentation:### Cumulative results: ###
INFO:renate.benchmark.experimentation:   Task ID  Average Accuracy  Forgetting  Forward Transfer  Backward Transfer
0        1          0.999527     0.00000          0.000000            0.00000
1        2          0.981425     0.01513         -0.001469           -0.01513

INFO:renate.benchmark.experimentation:Starting Update 2/2.
INFO:renate.training.training:Start updating the model.
WARNING:syne_tune.backend.local_backend:num_gpus_per_trial = 1 is too large, reducing to 0
INFO:renate.utils.syne_tune:Epoch 1/5
-----------------  ---------
train_accuracy     0.85281
train_base_loss    0.495211
train_loss         0.410285
train_memory_loss  0.0906021
train_cls_loss     0.0706438
val_accuracy       0.984046
val_loss           0.0690224
-----------------  ---------
INFO:renate.utils.syne_tune:Epoch 2/5
-----------------  ---------
train_accuracy     0.982571
train_base_loss    0.0624524
train_loss         0.061879
train_memory_loss  0.0174406
train_cls_loss     0.0191133
val_accuracy       0.973451
val_loss           0.100062
-----------------  ---------
INFO:renate.utils.syne_tune:Epoch 3/5
-----------------  ---------
train_accuracy     0.990414
train_base_loss    0.0403503
train_loss         0.0410895
train_memory_loss  0.011296
train_cls_loss     0.0140969
val_accuracy       0.962993
val_loss           0.132334
-----------------  ---------
INFO:renate.utils.syne_tune:Epoch 4/5
-----------------  ----------
train_accuracy     0.993987
train_base_loss    0.0285595
train_loss         0.0301703
train_memory_loss  0.00899642
train_cls_loss     0.0107165
val_accuracy       0.96782
val_loss           0.129036
-----------------  ----------
INFO:renate.utils.syne_tune:Epoch 5/5
-----------------  ----------
train_accuracy     0.996078
train_base_loss    0.0214255
train_loss         0.0240065
train_memory_loss  0.00762377
train_cls_loss     0.00936104
val_accuracy       0.961384
val_loss           0.160481
-----------------  ----------
WARNING:syne_tune.optimizer.schedulers.searchers.random_grid_searcher:Failed to sample a configuration not already chosen before. Exclusion list has size 1. Configuration space has size 1.
Tuning is finishing as the whole configuration space got exhausted.
--------------------
Resource summary (last result is reported):
 trial_id    status  iter  val_size            scenario_name class_groupings  num_tasks  max_epochs           model_name  num_hidden_layers  hidden_size updater optimizer  learning_rate  momentum  weight_decay  batch_size  memory_batch_size  memory_size  alpha  beta  stable_model_update_weight  plastic_model_update_weight  stable_model_update_probability  plastic_model_update_probability dataset_name  num_inputs  num_outputs                                                                      config_file  prepare_data  chunk_id      task_id  working_directory  seed accelerator  devices strategy precision  deterministic_trainer                input_state_url       metric mode  train_accuracy  train_base_loss  train_loss  train_memory_loss  train_cls_loss  val_accuracy  val_loss  step  epoch  worker-time
        0 Completed     5      0.05 ClassIncrementalScenario   [[0,1],[2,3]]          2           5 MultiLayerPerceptron                  1          200  CLS-ER      Adam           0.01       0.0           0.0         256                256          500    0.5   0.1                       0.999                        0.999                              0.7                               0.9        MNIST         784           10 /Users/marwistu/PycharmProjects/renate/src/renate/benchmark/experiment_config.py         False         1 default_task renate_working_dir     0        auto        1      ddp        32                   True renate_working_dir/input_state val_accuracy  max        0.996078         0.021426    0.024006           0.007624        0.009361      0.961384  0.160481     4      5    26.976227
0 trials running, 1 finished (1 until the end), 55.11s wallclock-time

val_accuracy: best 0.9840455651283264 for trial-id 0
--------------------
INFO:renate.training.training:All training is completed. Saving state...
Multi Objective Optimization dependencies are not imported since dependencies are missing. You can install them with
   pip install 'syne-tune[moo]'
or (for everything)
   pip install 'syne-tune[extra]'
memory_loss 0.005096937995404005
cls_loss 0.0
memory_loss 0.14620403945446014
cls_loss 0.02572334185242653
memory_loss 0.34193021059036255
cls_loss 0.0282069593667984
memory_loss 0.3718527555465698
cls_loss 0.04066632315516472
memory_loss 0.340764582157135
cls_loss 0.1053737923502922
memory_loss 0.271002858877182
cls_loss 0.21564479172229767
memory_loss 0.2654852271080017
cls_loss 0.22174052894115448
memory_loss 0.30037975311279297
cls_loss 0.16517524421215057
memory_loss 0.2779679596424103
cls_loss 0.13358645141124725
memory_loss 0.15971282124519348
cls_loss 0.11255566030740738
memory_loss 0.10579990595579147
cls_loss 0.11310428380966187
memory_loss 0.08017642050981522
cls_loss 0.1039714589715004
memory_loss 0.08255822211503983
cls_loss 0.11694584041833878
memory_loss 0.07952206581830978
cls_loss 0.10731837898492813
memory_loss 0.08284983783960342
cls_loss 0.1047186627984047
memory_loss 0.07362327724695206
cls_loss 0.11259625107049942
memory_loss 0.07893781363964081
cls_loss 0.09878376126289368
memory_loss 0.060299571603536606
cls_loss 0.09971838444471359
memory_loss 0.04013681411743164
cls_loss 0.09182211011648178
memory_loss 0.03978605195879936
cls_loss 0.09690503031015396
memory_loss 0.05058618262410164
cls_loss 0.09320685267448425
memory_loss 0.04922085255384445
cls_loss 0.09932448714971542
memory_loss 0.05567532032728195
cls_loss 0.0768149271607399
memory_loss 0.05580673739314079
cls_loss 0.07489845156669617
memory_loss 0.03573012351989746
cls_loss 0.06763892620801926
memory_loss 0.033818308264017105
cls_loss 0.054518330842256546
memory_loss 0.039408616721630096
cls_loss 0.03900536894798279
memory_loss 0.03901783376932144
cls_loss 0.03611285239458084
memory_loss 0.045370303094387054
cls_loss 0.036053366959095
memory_loss 0.03680703043937683
cls_loss 0.03345030918717384
memory_loss 0.03211863711476326
cls_loss 0.04142998903989792
memory_loss 0.026977647095918655
cls_loss 0.038694169372320175
memory_loss 0.03258942812681198
cls_loss 0.03634694218635559
memory_loss 0.037133850157260895
cls_loss 0.03187950327992439
memory_loss 0.03502855449914932
cls_loss 0.0316765271127224
memory_loss 0.02733113057911396
cls_loss 0.028107691556215286
memory_loss 0.024504145607352257
cls_loss 0.027779851108789444
memory_loss 0.03102208487689495
cls_loss 0.026175400242209435
memory_loss 0.03293240815401077
cls_loss 0.026011938229203224
memory_loss 0.029917124658823013
cls_loss 0.03280177339911461
memory_loss 0.024764416739344597
cls_loss 0.03102237544953823
memory_loss 0.02774086408317089
cls_loss 0.02854827046394348
memory_loss 0.027336690574884415
cls_loss 0.03254498168826103
memory_loss 0.020688045769929886
cls_loss 0.029985100030899048
memory_loss 0.021483002230525017
cls_loss 0.030383994802832603
[tune-metric]: {"train_accuracy": 0.8528104424476624, "train_base_loss": 0.4952106475830078, "train_loss": 0.41028523445129395, "train_memory_loss": 0.09060212969779968, "train_cls_loss": 0.07064377516508102, "val_accuracy": 0.9840455651283264, "val_loss": 0.06902240961790085, "step": 0, "epoch": 1, "st_worker_timestamp": 1690188896.356616, "st_worker_time": 6.894997802999999, "st_worker_iter": 0}
memory_loss 0.019215647131204605
cls_loss 0.029441634193062782
memory_loss 0.02306513860821724
cls_loss 0.028401849791407585
memory_loss 0.028079060837626457
cls_loss 0.02983993850648403
memory_loss 0.022438116371631622
cls_loss 0.030656620860099792
memory_loss 0.025721009820699692
cls_loss 0.026319188997149467
memory_loss 0.02491082064807415
cls_loss 0.02553083375096321
memory_loss 0.020872991532087326
cls_loss 0.0202670656144619
memory_loss 0.021065320819616318
cls_loss 0.020804574713110924
memory_loss 0.019218124449253082
cls_loss 0.020206233486533165
memory_loss 0.018903572112321854
cls_loss 0.02174350433051586
memory_loss 0.018866149708628654
cls_loss 0.020888155326247215
memory_loss 0.02184050716459751
cls_loss 0.019363006576895714
memory_loss 0.023492014035582542
cls_loss 0.01869572140276432
memory_loss 0.023898955434560776
cls_loss 0.019448397681117058
memory_loss 0.017983537167310715
cls_loss 0.017314685508608818
memory_loss 0.015626143664121628
cls_loss 0.020433424040675163
memory_loss 0.017185349017381668
cls_loss 0.020013276487588882
memory_loss 0.017290214076638222
cls_loss 0.02110586129128933
memory_loss 0.01910841092467308
cls_loss 0.019698455929756165
memory_loss 0.0155636016279459
cls_loss 0.02044372260570526
memory_loss 0.013651935383677483
cls_loss 0.02147776260972023
memory_loss 0.015368469059467316
cls_loss 0.023463888093829155
memory_loss 0.013352621346712112
cls_loss 0.02167225442826748
memory_loss 0.013185182586312294
cls_loss 0.020423326641321182
memory_loss 0.014313633553683758
cls_loss 0.016810540109872818
memory_loss 0.015426931902766228
cls_loss 0.015804532915353775
memory_loss 0.018050542101264
cls_loss 0.0141306696459651
memory_loss 0.016801128163933754
cls_loss 0.013508756645023823
memory_loss 0.01713835820555687
cls_loss 0.01366842444986105
memory_loss 0.013455417938530445
cls_loss 0.015379066579043865
memory_loss 0.015376821160316467
cls_loss 0.015070955269038677
memory_loss 0.012290013954043388
cls_loss 0.014920582063496113
memory_loss 0.01763325370848179
cls_loss 0.014446568675339222
memory_loss 0.017570286989212036
cls_loss 0.015322329476475716
memory_loss 0.01609482429921627
cls_loss 0.015104131773114204
memory_loss 0.013227120973169804
cls_loss 0.015393554233014584
memory_loss 0.012327236123383045
cls_loss 0.017439723014831543
memory_loss 0.015840187668800354
cls_loss 0.017740463837981224
memory_loss 0.01502937637269497
cls_loss 0.016926391050219536
memory_loss 0.015489554964005947
cls_loss 0.01616474986076355
memory_loss 0.017047835513949394
cls_loss 0.014451560564339161
memory_loss 0.014795091934502125
cls_loss 0.015014528296887875
memory_loss 0.013440870679914951
cls_loss 0.014730962924659252
memory_loss 0.010915654711425304
cls_loss 0.014985633082687855
memory_loss 0.01266059372574091
cls_loss 0.015432926826179028
[tune-metric]: {"train_accuracy": 0.9825708270072937, "train_base_loss": 0.062452442944049835, "train_loss": 0.061878982931375504, "train_memory_loss": 0.017440611496567726, "train_cls_loss": 0.019113343209028244, "val_accuracy": 0.9734513163566589, "val_loss": 0.10006207227706909, "step": 1, "epoch": 2, "st_worker_timestamp": 1690188901.911792, "st_worker_time": 12.450151512000001, "st_worker_iter": 1}
memory_loss 0.011271397583186626
cls_loss 0.017314722761511803
memory_loss 0.010938337072730064
cls_loss 0.01558799296617508
memory_loss 0.012714246287941933
cls_loss 0.013343624770641327
memory_loss 0.01602223329246044
cls_loss 0.012810416519641876
memory_loss 0.014507044106721878
cls_loss 0.011075506918132305
memory_loss 0.012662452645599842
cls_loss 0.012721635401248932
memory_loss 0.013757931999862194
cls_loss 0.01833086833357811
memory_loss 0.012332230806350708
cls_loss 0.020833177492022514
memory_loss 0.011353439651429653
cls_loss 0.01517987996339798
memory_loss 0.01380658894777298
cls_loss 0.015269304625689983
memory_loss 0.016753049567341805
cls_loss 0.017453357577323914
memory_loss 0.012748649343848228
cls_loss 0.01677011512219906
memory_loss 0.00986518431454897
cls_loss 0.01551958080381155
memory_loss 0.009244060143828392
cls_loss 0.019424831494688988
memory_loss 0.008892246522009373
cls_loss 0.02187848649919033
memory_loss 0.010621237568557262
cls_loss 0.01618201471865177
memory_loss 0.012313556857407093
cls_loss 0.012550557032227516
memory_loss 0.013563976623117924
cls_loss 0.011594007723033428
memory_loss 0.013672503642737865
cls_loss 0.011489777825772762
memory_loss 0.011080249212682247
cls_loss 0.010481047444045544
memory_loss 0.009834198281168938
cls_loss 0.012650554068386555
memory_loss 0.009478696621954441
cls_loss 0.014234391041100025
memory_loss 0.010201716795563698
cls_loss 0.011862319894134998
memory_loss 0.01164070051163435
cls_loss 0.011712715961039066
memory_loss 0.011714637279510498
cls_loss 0.01130919810384512
memory_loss 0.011195234954357147
cls_loss 0.011563219130039215
memory_loss 0.009800364263355732
cls_loss 0.012226481921970844
memory_loss 0.009734834544360638
cls_loss 0.013195693492889404
memory_loss 0.009494668804109097
cls_loss 0.013761746697127819
memory_loss 0.009644284844398499
cls_loss 0.012961163185536861
memory_loss 0.00930106546729803
cls_loss 0.014123968780040741
memory_loss 0.010288316756486893
cls_loss 0.014429422095417976
memory_loss 0.010490020737051964
cls_loss 0.011601515114307404
memory_loss 0.010684005916118622
cls_loss 0.01209208369255066
memory_loss 0.011112615466117859
cls_loss 0.011356365866959095
memory_loss 0.01086355373263359
cls_loss 0.010470067150890827
memory_loss 0.01025240495800972
cls_loss 0.011435823515057564
memory_loss 0.010435004718601704
cls_loss 0.01419425755739212
memory_loss 0.012152744457125664
cls_loss 0.013314664363861084
memory_loss 0.008842667564749718
cls_loss 0.013970854692161083
memory_loss 0.011064011603593826
cls_loss 0.012467038817703724
memory_loss 0.012690083123743534
cls_loss 0.014171910472214222
memory_loss 0.012366891838610172
cls_loss 0.013807971961796284
memory_loss 0.008067338727414608
cls_loss 0.014933379366993904
memory_loss 0.008849762380123138
cls_loss 0.0207020603120327
[tune-metric]: {"train_accuracy": 0.9904139637947083, "train_base_loss": 0.04035032540559769, "train_loss": 0.04108951613306999, "train_memory_loss": 0.011296011507511139, "train_cls_loss": 0.014096882194280624, "val_accuracy": 0.962992787361145, "val_loss": 0.13233426213264465, "step": 2, "epoch": 3, "st_worker_timestamp": 1690188906.951042, "st_worker_time": 17.489380888000003, "st_worker_iter": 2}
memory_loss 0.009237128309905529
cls_loss 0.020141158252954483
memory_loss 0.009401068091392517
cls_loss 0.015704747289419174
memory_loss 0.009654521010816097
cls_loss 0.011721618473529816
memory_loss 0.010669615119695663
cls_loss 0.01131129078567028
memory_loss 0.012265537865459919
cls_loss 0.009341408498585224
memory_loss 0.011097087524831295
cls_loss 0.008620147593319416
memory_loss 0.009216006845235825
cls_loss 0.009005575440824032
memory_loss 0.007642671931535006
cls_loss 0.010856742039322853
memory_loss 0.007788891904056072
cls_loss 0.01198359951376915
memory_loss 0.00871710479259491
cls_loss 0.010774070397019386
memory_loss 0.008802024647593498
cls_loss 0.009225375019013882
memory_loss 0.009991978295147419
cls_loss 0.010595201514661312
memory_loss 0.009311119094491005
cls_loss 0.009283659979701042
memory_loss 0.009443577378988266
cls_loss 0.009696691296994686
memory_loss 0.008419865742325783
cls_loss 0.010851956903934479
memory_loss 0.008354109711945057
cls_loss 0.010653005912899971
memory_loss 0.007875199429690838
cls_loss 0.01256980188190937
memory_loss 0.008066135458648205
cls_loss 0.011349665932357311
memory_loss 0.00967130996286869
cls_loss 0.010177765972912312
memory_loss 0.00891067087650299
cls_loss 0.010176497511565685
memory_loss 0.008243443444371223
cls_loss 0.009341391734778881
memory_loss 0.007859020493924618
cls_loss 0.009480799548327923
memory_loss 0.008320627734065056
cls_loss 0.009389578364789486
memory_loss 0.00827388372272253
cls_loss 0.009696407243609428
memory_loss 0.009202218614518642
cls_loss 0.009506743401288986
memory_loss 0.010011132806539536
cls_loss 0.009369658306241035
memory_loss 0.009256809018552303
cls_loss 0.009336833842098713
memory_loss 0.008605985902249813
cls_loss 0.011053832247853279
memory_loss 0.008759291842579842
cls_loss 0.011967782862484455
memory_loss 0.009192824363708496
cls_loss 0.011318343691527843
memory_loss 0.007515942212194204
cls_loss 0.010113745927810669
memory_loss 0.00900026224553585
cls_loss 0.00918826274573803
memory_loss 0.00931930635124445
cls_loss 0.009719355963170528
memory_loss 0.00962812453508377
cls_loss 0.008925535716116428
memory_loss 0.008036208339035511
cls_loss 0.009756435640156269
memory_loss 0.008447226136922836
cls_loss 0.011058489792048931
memory_loss 0.007674442604184151
cls_loss 0.010289909318089485
memory_loss 0.009239185601472855
cls_loss 0.011009658686816692
memory_loss 0.008441714569926262
cls_loss 0.010913415811955929
memory_loss 0.010172621347010136
cls_loss 0.010093780234456062
memory_loss 0.009558280929923058
cls_loss 0.009327985346317291
memory_loss 0.010093037970364094
cls_loss 0.009326467290520668
memory_loss 0.008252174593508244
cls_loss 0.01136835478246212
memory_loss 0.009132754057645798
cls_loss 0.013717119581997395
memory_loss 0.008066561073064804
cls_loss 0.012934060767292976
[tune-metric]: {"train_accuracy": 0.9939869046211243, "train_base_loss": 0.028559479862451553, "train_loss": 0.030170265585184097, "train_memory_loss": 0.008996416814625263, "train_cls_loss": 0.010716530494391918, "val_accuracy": 0.9678198099136353, "val_loss": 0.12903554737567902, "step": 3, "epoch": 4, "st_worker_timestamp": 1690188911.847706, "st_worker_time": 22.386024377000002, "st_worker_iter": 3}
memory_loss 0.007966110482811928
cls_loss 0.011906461790204048
memory_loss 0.008280068635940552
cls_loss 0.011871234513819218
memory_loss 0.00835714116692543
cls_loss 0.011643032543361187
memory_loss 0.00800559762865305
cls_loss 0.010738964192569256
memory_loss 0.007410835474729538
cls_loss 0.010789744555950165
memory_loss 0.007721062749624252
cls_loss 0.010128580033779144
memory_loss 0.007339009083807468
cls_loss 0.008823535405099392
memory_loss 0.008877472952008247
cls_loss 0.007776074111461639
memory_loss 0.008506884798407555
cls_loss 0.007659942843019962
memory_loss 0.008444688282907009
cls_loss 0.008157560601830482
memory_loss 0.00809918250888586
cls_loss 0.00953651126474142
memory_loss 0.007136956322938204
cls_loss 0.009925872087478638
memory_loss 0.007394029758870602
cls_loss 0.011425919830799103
memory_loss 0.007027920801192522
cls_loss 0.010126371867954731
memory_loss 0.007220755331218243
cls_loss 0.010571050457656384
memory_loss 0.007352539338171482
cls_loss 0.010480373166501522
memory_loss 0.007458531763404608
cls_loss 0.009921378456056118
memory_loss 0.00783049501478672
cls_loss 0.011467387899756432
memory_loss 0.006990163121372461
cls_loss 0.010478494688868523
memory_loss 0.007226827088743448
cls_loss 0.009227091446518898
memory_loss 0.0072150882333517075
cls_loss 0.008477606810629368
memory_loss 0.007577372249215841
cls_loss 0.007645490113645792
memory_loss 0.00790786650031805
cls_loss 0.0073738121427595615
memory_loss 0.007502428721636534
cls_loss 0.00860894750803709
memory_loss 0.00770726939663291
cls_loss 0.009601321071386337
memory_loss 0.007958874106407166
cls_loss 0.009553740732371807
memory_loss 0.007589365355670452
cls_loss 0.008556181564927101
memory_loss 0.007981826551258564
cls_loss 0.007907689549028873
memory_loss 0.008358214050531387
cls_loss 0.007782870437949896
memory_loss 0.007855058647692204
cls_loss 0.008981524035334587
memory_loss 0.0081067169085145
cls_loss 0.009707767516374588
memory_loss 0.007114439737051725
cls_loss 0.01006033644080162
memory_loss 0.007371365092694759
cls_loss 0.010136212222278118
memory_loss 0.006937857251614332
cls_loss 0.009067792445421219
memory_loss 0.007753558456897736
cls_loss 0.00815556664019823
memory_loss 0.007472215220332146
cls_loss 0.007331406231969595
memory_loss 0.007039936259388924
cls_loss 0.007440761663019657
memory_loss 0.006777855101972818
cls_loss 0.007499107625335455
memory_loss 0.006720053963363171
cls_loss 0.008698253892362118
memory_loss 0.007269847672432661
cls_loss 0.009827653877437115
memory_loss 0.007474203128367662
cls_loss 0.007692898157984018
memory_loss 0.008391721174120903
cls_loss 0.008449717424809933
memory_loss 0.00803714245557785
cls_loss 0.008648477494716644
memory_loss 0.007492929697036743
cls_loss 0.009850370697677135
memory_loss 0.006809972692281008
cls_loss 0.011535851284861565
[tune-metric]: {"train_accuracy": 0.9960784316062927, "train_base_loss": 0.021425524726510048, "train_loss": 0.02400645986199379, "train_memory_loss": 0.007623766548931599, "train_cls_loss": 0.009361043572425842, "val_accuracy": 0.9613837599754333, "val_loss": 0.1604812890291214, "step": 4, "epoch": 5, "st_worker_timestamp": 1690188916.437928, "st_worker_time": 26.976227234000003, "st_worker_iter": 4}
"""


class WeightedCLSLossComponent(WeightedLossComponent):
    """Complementary Learning Systems Based Experience Replay.

    Arani, Elahe, Fahad Sarfraz, and Bahram Zonooz.
    "Learning fast, learning slow: A general continual learning method based on complementary
    learning system." arXiv preprint arXiv:2201.12604 (2022).

    The implementation follows the Algorithm 1 in the respective paper. The complete `Learner`
    implementing this loss, is the `CLSExperienceReplayLearner`.

    Args:
        weight: A scaling coefficient which should scale the loss which gets returned.
        sample_new_memory_batch: Whether a new batch of data should be sampled from the memory
            buffer when the loss is calculated.
        model: The model that is being trained.
        stable_model_update_weight: The weight used in the update of the stable model.
        plastic_model_update_weight:  The weight used in the update of the plastic model.
        stable_model_update_probability:  The probability of updating the stable model at each
            training step.
        plastic_model_update_probability:  The probability of updating the plastic model at each
            training step.
    """

    def __init__(
        self,
        weight: float,
        sample_new_memory_batch: bool,
        model: RenateModule,
        stable_model_update_weight: float,
        plastic_model_update_weight: float,
        stable_model_update_probability: float,
        plastic_model_update_probability: float,
    ) -> None:
        self._stable_model_update_weight = torch.tensor(
            stable_model_update_weight, dtype=torch.float32
        )
        self._stable_model_update_weight = torch.tensor(
            stable_model_update_weight, dtype=torch.float32
        )
        self._plastic_model_update_weight = torch.tensor(
            plastic_model_update_weight, dtype=torch.float32
        )
        self._stable_model_update_probability = torch.tensor(
            stable_model_update_probability, dtype=torch.float32
        )
        self._plastic_model_update_probability = torch.tensor(
            plastic_model_update_probability, dtype=torch.float32
        )
        self._iteration = torch.tensor(0, dtype=torch.int64)
        super().__init__(weight=weight, sample_new_memory_batch=sample_new_memory_batch)
        self._plastic_model: RenateModule = copy.deepcopy(model)
        self._stable_model: RenateModule = copy.deepcopy(model)
        self._plastic_model.deregister_hooks()
        self._stable_model.deregister_hooks()
        # print("INIT")

    def _register_parameters(
        self,
        weight: float,
        sample_new_memory_batch: bool,
        stable_model_update_weight: float,
        plastic_model_update_weight: float,
        stable_model_update_probability: float,
        plastic_model_update_probability: float,
        iteration: int,
    ) -> None:
        """Register the parameters of the loss component."""
        super()._register_parameters(
            weight=weight,
            sample_new_memory_batch=sample_new_memory_batch,
        )
        self.register_buffer(
            "_stable_model_update_weight",
            torch.tensor(stable_model_update_weight, dtype=torch.float32),
        )
        self.register_buffer(
            "_plastic_model_update_weight",
            torch.tensor(plastic_model_update_weight, dtype=torch.float32),
        )
        self.register_buffer(
            "_stable_model_update_probability",
            torch.tensor(stable_model_update_probability, dtype=torch.float32),
        )
        self.register_buffer(
            "_plastic_model_update_probability",
            torch.tensor(plastic_model_update_probability, dtype=torch.float32),
        )
        self.register_buffer("_iteration", torch.tensor(iteration, dtype=torch.int64))

    def _verify_attributes(self) -> None:
        """Verify if attributes have valid values."""
        super()._verify_attributes()
        assert 0.0 <= self._stable_model_update_weight
        assert 0.0 <= self._plastic_model_update_weight
        assert 0.0 <= self._stable_model_update_probability <= 1.0
        assert 0.0 <= self._plastic_model_update_probability <= 1.0
        assert self._plastic_model_update_probability > self._stable_model_update_probability
        assert self._plastic_model_update_weight <= self._stable_model_update_weight

    def _loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[Tuple[NestedTensors, torch.Tensor], Dict[str, torch.Tensor]],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Computes the consistency loss with respect to averaged plastic and stable models."""
        (inputs_memory, targets_memory), _ = batch_memory
        with torch.no_grad():
            outputs_plastic = self._plastic_model(inputs_memory)
            outputs_stable = self._plastic_model(inputs_memory)
            probs_plastic = F.softmax(outputs_plastic, dim=-1)
            probs_stable = F.softmax(outputs_stable, dim=-1)
            label_mask = F.one_hot(targets_memory, num_classes=outputs_stable.shape[-1]) > 0
            idx = (probs_stable[label_mask] > probs_plastic[label_mask]).unsqueeze(1)
            outputs = torch.where(idx, outputs_stable, outputs_plastic)
        consistency_loss = F.mse_loss(outputs_memory, outputs.detach(), reduction="mean")
        print(
            float(consistency_loss.detach()),
            self.weight,
            float(outputs_memory.detach()[0][0]),
            float(outputs.detach()[0][0]),
            float(outputs_plastic[0][0]),
            float(outputs_stable[0][0]),
        )
        # for i, p in enumerate(self._plastic_model.parameters()):
        #    while len(p.shape) > 1:
        #        p = p[0]
        #    print("Plastic Model", i, float(p[0]))
        return self.weight * consistency_loss

    @torch.no_grad()
    def _update_model_variables(
        self, model: RenateModule, original_model: RenateModule, weight: torch.Tensor
    ) -> None:
        """Performs exponential moving average on the stored model copies.

        Args:
            model: Whether the plastic or the stable model is updated.
            weight: The minimum weight used in the exponential moving average to update the model.
        """
        alpha = min(
            1.0 - torch.tensor(1.0, device=self._iteration.device) / (self._iteration + 1), weight
        )
        for ema_p, p in zip(model.parameters(), original_model.parameters()):
            ema_p.data.mul_(alpha).add_(p.data, alpha=1 - alpha)

    def on_train_batch_end(self, model: RenateModule) -> None:
        """Updates the model copies with the current weights,
        given the specified probabilities of update, and increments iteration counter."""
        self._iteration += 1
        if (
            torch.rand(1, device=self._plastic_model_update_probability.device)
            < self._plastic_model_update_probability
        ):
            self._update_model_variables(
                self._plastic_model, model, self._plastic_model_update_weight
            )

        if (
            torch.rand(1, device=self._stable_model_update_probability.device)
            < self._stable_model_update_probability
        ):
            self._update_model_variables(
                self._stable_model, model, self._stable_model_update_weight
            )

    def set_stable_model_update_weight(self, stable_model_update_weight: float) -> None:
        self._stable_model_update_weight.data = torch.tensor(
            stable_model_update_weight,
            dtype=self._stable_model_update_weight.dtype,
            device=self._stable_model_update_weight.device,
        )
        self._verify_attributes()

    def set_plastic_model_update_weight(self, plastic_model_update_weight: float) -> None:
        self._plastic_model_update_weight.data = torch.tensor(
            plastic_model_update_weight,
            dtype=self._plastic_model_update_weight.dtype,
            device=self._plastic_model_update_weight.device,
        )
        self._verify_attributes()

    def set_stable_model_update_probability(self, stable_model_update_probability: float) -> None:
        self._stable_model_update_probability.data = torch.tensor(
            stable_model_update_probability,
            dtype=self._stable_model_update_probability.dtype,
            device=self._stable_model_update_probability.device,
        )
        self._verify_attributes()

    def set_plastic_model_update_probability(self, plastic_model_update_probability: float) -> None:
        self._plastic_model_update_probability.data = torch.tensor(
            plastic_model_update_probability,
            dtype=self._plastic_model_update_probability.dtype,
            device=self._plastic_model_update_probability.device,
        )
        self._verify_attributes()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load relevant information from checkpoint."""
        super().on_load_checkpoint(checkpoint)
        print("LOAD")
        self._plastic_model = checkpoint["component-cls-plastic-model"]
        self._stable_model = checkpoint["component-cls-stable-model"]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Add plastic and stable model to checkpoint."""
        super().on_save_checkpoint(checkpoint)
        print("SAVE")
        checkpoint["component-cls-plastic-model"] = self._plastic_model
        checkpoint["component-cls-stable-model"] = self._stable_model
