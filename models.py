import ot
import numpy
import utils


class GromovWasserstein(object):
    def __init__(self, source_cost_calculator, target_cost_calculator):
        # 関数渡しでC1, C2を計算するための関数への参照を受け取る
        # 関数の引数はそれぞれsource_variableまたはtaget_variableの1変数の関数orメソッド
        self._source_cost_calculator = source_cost_calculator
        self._target_cost_calculator = target_cost_calculator
        self._label_converter = utils.LabelConverter()

    def fit(self, source_variables, target_variables, source_labels, confidence_threshold=None):
        # 特徴量は sample x n_featureの形で持つようにする
        if source_variables.ndim == 1:
            source_variables = source_variables[:, None]
        if target_variables.ndim == 1:
            target_variables = target_variables[:, None]

        self._source_variables = source_variables
        self._target_variables = target_variables
        self._source_labels = source_labels
        hs = numpy.ones(source_variables.shape[0]) / source_variables.shape[0]
        ht = numpy.ones(target_variables.shape[0]) / target_variables.shape[0]

        # source_labels_onehot = self._label_converter.encode(source_labels)
        C1 = self._source_cost_calculator(source_variables)
        C2 = self._source_cost_calculator(target_variables)

        self._transport_plan = ot.gromov.gromov_wasserstein(C1, C2, hs, ht)

        # 重心写像でターゲットの特にラベルを推定
        self._target_labels_est = self.barycentric_mapping()
        # ラベルの確信度
        self._confidence = self.calc_confidence()

        # 確信度でフィルタリングせずにhard coded labelを利用したい場合はconfidence thresholdを0にして運用
        if confidence_threshold is not None:
            X, labels = self._get_hard_labels(
                self._target_variables, self._target_labels_est,
                self._confidence, confidence_threshold)

        self._confident_target_samples = (X, labels)

    def calc_confidence(self, labels=None):
        if labels is None:
            labels = self.barycentric_mapping()

        confidence = numpy.ones(len(labels))
        labels += 1e-12
        labels /= labels.sum(1)[:, None]
        confidence -= -(labels * numpy.log2(labels)).sum(1) / numpy.log2(labels.shape[1])

        return confidence

    def _get_hard_labels(self, X, soft_labels, confidence, confidence_threshold=0):
        confidence = self.calc_confidence()
        decoded_labels = self.predict()
        Xout = X[confidence >= confidence_threshold]
        yout = decoded_labels[confidence >= confidence_threshold]

        return Xout, yout

    def barycentric_mapping(self):
        one_hot_labels = self._label_converter.encode(self._source_labels)
        # source側ラベルの重心を計算し，これをtestの各点のラベルとして利用
        # 理論上はtarget_data数をtransport_mapにかければ良いのだけど，数値誤差がある
        label_barycenter = \
            (self._transport_plan / self._transport_plan.sum(0)).T.\
            dot(one_hot_labels)

        return label_barycenter

    def predict(self):
        labels = self.barycentric_mapping()
        decoded_labels = self._label_converter.hard_decode(labels).astype(int)

        return decoded_labels


class FusedGromovWasserstein(GromovWasserstein):
    def __init__(self, source_cost_calculator, target_cost_calculator,
                 label_cost_calculator,
                 source_target_cost_calculator,
                 alpha=1e-4, beta=0.5):
        super().__init__(source_cost_calculator, target_cost_calculator)
        self._source_target_cost_calculator = source_target_cost_calculator
        self._label_cost_calculator = label_cost_calculator
        self._alpha = alpha
        self._beta = beta

    def fit(self, source_variables, target_variables, source_labels,
            confidence_threshold=None):

        # 特徴量は sample x n_featureの形で持つようにする
        if source_variables.ndim == 1:
            source_variables = source_variables[:, None]
        if target_variables.ndim == 1:
            target_variables = target_variables[:, None]

        self._source_variables = source_variables
        self._target_variables = target_variables
        self._source_labels = source_labels
        hs = numpy.ones(source_variables.shape[0]) / source_variables.shape[0]
        ht = numpy.ones(target_variables.shape[0]) / target_variables.shape[0]

        source_labels_onehot = self._label_converter.encode(source_labels)

        C1 = self._source_cost_calculator(source_variables)
        C1 /= numpy.max(C1)
        C2 = self._target_cost_calculator(target_variables)
        # C2 /= numpy.max(C2)
        labelloss = self._label_cost_calculator(source_labels_onehot)
        labelloss /= numpy.max(labelloss)
        # M = sklearn.metrics.pairwise_distances(
        #     source_labels_onehot, source_labels_onehot,
        #     metric=self._metrics_list[1])

        # source_variablesの列数=共通変数の数という設定なので
        target_variables_common = target_variables[:, :source_variables.shape[1]]
        if numpy.ndim(target_variables_common) == 1:
            target_variables_common = target_variables_common.reshape(-1, 1)
        # TODO: 距離の正規化をするならば，まずこの距離行列計算時点で正規化をしておくべきな気がする
        M = self._source_target_cost_calculator(
            source_variables, target_variables[:, :source_variables.shape[1]])
        M /= numpy.max(M)

        # self._transport_plan = ot.gromov.fused_gromov_wasserstein(
        #     M, C1, C2, hs, ht, alpha=self._alpha)
        self._transport_plan = ot.gromov.fused_gromov_wasserstein(
            M, (1 - self._beta) * C1 + self._beta * labelloss, C2, hs, ht, alpha=self._alpha)

        # 重心写像でターゲットの特にラベルを推定
        labels = self.barycentric_mapping()
        self._target_labels_est = labels
        # ラベルの確信度
        confidence = self.calc_confidence()

        # 確信度でフィルタリングせずにhard coded labelを利用したい場合はconfidence thresholdを0にして運用
        if confidence_threshold is not None:
            X, labels =\
                self._get_hard_labels(self._target_variables, labels,
                                      confidence, confidence_threshold)

        self._confident_target_samples = (X, labels)


class ClassificationAndMajorityVote(GromovWasserstein):
    def __init__(self, source_target_cost_calculator, n_clusters=2, affinity='nearest_neighbor',
                 n_neighbors=4, gamma=0.1, random_state=None):
        import sklearn.cluster
        self._source_target_cost_calculator = source_target_cost_calculator
        self._label_converter = utils.LabelConverter()

        self._clf = sklearn.cluster.SpectralClustering(
            n_clusters=n_clusters, assign_labels='discretize', affinity=affinity,
            n_neighbors=n_neighbors, gamma=gamma, random_state=random_state)

    def fit(self, source_variables, target_variables, source_labels,
            confidence_threshold=None):
        # 特徴量は sample x n_featureの形で持つようにする
        if source_variables.ndim == 1:
            source_variables = source_variables[:, None]
        if target_variables.ndim == 1:
            target_variables = target_variables[:, None]

        self._source_variables = source_variables
        self._target_variables = target_variables
        self._source_labels = source_labels

        # -------------------- OTの処理 --------------------
        hs = numpy.ones(source_variables.shape[0]) / source_variables.shape[0]
        ht = numpy.ones(target_variables.shape[0]) / target_variables.shape[0]

        M = self._source_target_cost_calculator(
            source_variables, target_variables[:, :source_variables.shape[1]])
        M /= numpy.max(M)

        self._transport_plan = ot.emd(hs, ht, M)
        ot_labels = self.barycentric_mapping()
        # # ラベルの確信度
        self._confidence = self.calc_confidence(ot_labels)

        # if confidence_threshold is not None:
        #     X, ot_labels = self._get_hard_labels(
        #         self._target_variables, ot_labels,
        #         self._confidence, confidence_threshold)

        # self._confident_target_samples = (X, ot_labels)
        # ot_labels = self._label_converter.hard_decode(ot_labels)

        # -------------------- Spectral Clustering --------------------
        cluster_labels = self._clf.fit_predict(self._target_variables)
        self._target_labels_est = cluster_labels

        # -------------------- Decide Cluster Labels by Majority Vote --------------------
        target_labels_est = numpy.zeros_like(ot_labels)
        for label in numpy.unique(cluster_labels):
            labels_in_cluster = ot_labels[cluster_labels == label]
        #     confidence_in_cluster = self._confidence[cluster_labels == label]
        #     labels_in_cluster = labels_in_cluster[confidence_in_cluster > 0.5]
        #     label_list, counts = numpy.unique(labels_in_cluster, return_counts=True)
            print(labels_in_cluster.mean(0))
            target_labels_est[cluster_labels == label] = labels_in_cluster.mean(0)
        self._target_labels_est = self._label_converter.hard_decode(target_labels_est).astype(int)

    def predict(self):
        return self._target_labels_est
