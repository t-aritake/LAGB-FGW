import numpy
import pandas


class LinearDataset(object):
    def __init__(
            self, source_mu_list, source_Sigma_list,
            target_mu_list, target_Sigma_list,
            Ns=1000, Nt=100,
            class_ratio=[0.5, 0.5], common_idx=None):
        self._source_mu_list = source_mu_list
        self._source_Sigma_list = source_Sigma_list
        self._target_mu_list = target_mu_list
        self._target_Sigma_list = target_Sigma_list
        self._Ns = Ns
        self._Nt = Nt
        self._class_ratio = class_ratio
        self._common_idx = common_idx

    def gen_data(self):
        Xs, ys = self._gen_linear_data(
            self._Ns, self._source_mu_list, self._source_Sigma_list)
        Xt, yt = self._gen_linear_data(
            self._Nt, self._target_mu_list, self._target_Sigma_list)

        if self._common_idx is None:
            self._common_idx = numpy.arange(Xs.shape[1])

        return Xs, ys, Xt, yt, self._common_idx

    def _gen_linear_data(self, N, mu_list, Sigma_list):
        # クラスタごとの平均値と分散共分散行列を入力する形
        N_list = (N * numpy.array(self._class_ratio)).astype(int)
        X_list = []
        y_list = []

        for i in range(len(self._class_ratio)):
            X = numpy.random.multivariate_normal(
                mu_list[i], Sigma_list[i], size=N_list[i])
            y = numpy.ones(N_list[i]) * i
            X_list.append(X)
            y_list.append(y)

        return numpy.row_stack(X_list), numpy.concatenate(y_list)


class TwoBallsDataset(object):
    def __init__(self, Ns=1000, Nt=1000, class_ratio=[0.5, 0.5],
                 radius_list=[2, 3], radius_std_list=[0.2, 0.1],
                 source_mu_list=numpy.array([0, 0]),
                 source_scales_list=numpy.array([1, 1]),
                 target_mu_list=numpy.array([3, 1]),
                 target_scales_list=numpy.array([1, 1]),
                 common_idx=None, only_common=False):
        # 2クラスであると考える．radius_listやradius_std_listは2クラス分数があればよい
        # source_mu_listなどは次元に応じた要素数で与えるものとする
        self._Ns = Ns
        self._Nt = Nt
        self._class_ratio = class_ratio
        # radius_list等は要素数2ならばsource, targetで同じ半径を利用
        # そうでなければ要素数4にして，source, targetで異なる半径とする
        if len(radius_list) == 2:
            self._radius_list = numpy.append(radius_list, radius_list)
        else:
            self._radius_list = radius_list
        if len(radius_std_list) == 2:
            self._radius_std_list =\
                numpy.append(radius_std_list, radius_std_list)
        else:
            self._radius_std_list = radius_std_list
        self._source_mu_list = source_mu_list
        self._source_scales_list = source_scales_list
        self._target_mu_list = target_mu_list
        self._target_scales_list = target_scales_list
        self._common_idx = common_idx
        self._only_common = only_common

    def gen_data(self):
        Xs, ys = self._gen_two_balls(
            self._Ns, self._class_ratio,
            self._radius_list[:2], self._radius_std_list[:2],
            self._source_mu_list, self._source_scales_list)
        Xt, yt = self._gen_two_balls(
            self._Nt, self._class_ratio,
            self._radius_list[2:], self._radius_std_list[2:],
            self._target_mu_list, self._target_scales_list)

        if self._common_idx is None:
            self._common_idx = numpy.arange(Xs.shape[1])

        return Xs, ys, Xt, yt, self._common_idx

    def _gen_two_balls(self, N, class_ratio, radius_list, radius_std_list,
                       mu_list, scales_list):
        assert len(mu_list[0]) == len(scales_list[0]), \
            "length of averages and standard deviation should be the same"

        dim = len(mu_list[0])
        num_samples = (N * numpy.array(class_ratio)).astype(int)

        # n次元の場合はn-1個の角度パラメータが必要
        theta1 = numpy.random.uniform(
            0, 2 * numpy.pi, size=(num_samples[0], dim - 1))
        theta2 = numpy.random.uniform(
            0, 2 * numpy.pi, size=(num_samples[1], dim - 1))

        cos1 = numpy.column_stack(
            (numpy.cos(theta1), numpy.ones(num_samples[0])))
        sin1 = numpy.column_stack(
            (numpy.ones(num_samples[0]), numpy.sin(theta1)))
        cos2 = numpy.column_stack(
            (numpy.cos(theta2), numpy.ones(num_samples[1])))
        sin2 = numpy.column_stack(
            (numpy.ones(num_samples[1]), numpy.sin(theta2)))

        X1 = numpy.cumprod(sin1, axis=1) * cos1
        X2 = numpy.cumprod(sin2, axis=1) * cos2

        # 半径はスカラーでOK
        r1 = numpy.random.normal(
            self._radius_list[0], self._radius_std_list[0],
            size=(num_samples[0], 1))
        r2 = numpy.random.normal(
            self._radius_list[1], self._radius_std_list[1],
            size=(num_samples[1], 1))

        X1 = r1 * scales_list[0] * X1 + mu_list[0]
        X2 = r2 * scales_list[1] * X2 + mu_list[1]

        X = numpy.concatenate((X1, X2), axis=0)
        y = numpy.concatenate(
            (numpy.zeros(num_samples[0]), numpy.ones(num_samples[1])))

        return X, y


class TwoSpiralsDataset(object):
    def __init__(self, Ns=1000, Nt=1000, class_ratio=[0.5, 0.5],
                 rotation=2.4, start=0.5, noise=0.2,
                 Xs_center=[0, 0], Xt_center=[0, 0],
                 Xs_scale=[1, 1], Xt_scale=[1, 1],
                 common_idx=None, only_common=False):
        self._Ns = Ns
        self._Nt = Nt
        self._class_ratio = class_ratio
        self._rotation = rotation * numpy.pi
        self._start = start * numpy.pi
        self._noise = noise

        self._Xs_center = Xs_center
        self._Xt_center = Xt_center
        self._Xs_scale = Xs_scale
        self._Xt_scale = Xt_scale
        self._common_idx = common_idx
        self._only_common = only_common

    def gen_data(self):
        # ここで生成したデータを複数のモデルで使いたいタイミングがあるかもしれないがとりあえず考えない
        Xs, ys = self._gen_two_spirals(self._Ns, self._class_ratio)
        Xt, yt = self._gen_two_spirals(self._Nt, self._class_ratio)

        Xs = Xs * self._Xs_scale + self._Xs_center
        Xt = Xt * self._Xt_scale + self._Xt_center

        if self._common_idx is None:
            self._common_idx = numpy.arange(Xs.shape[1])

        return Xs, ys, Xt, yt, self._common_idx

    def _gen_two_spirals(self, N, class_ratio):
        num_samples = (N * numpy.array(self._class_ratio)).astype(int)
        # 円の半径が小さい方が密になるようにsqrtを一様分布からのサンプルにかけている
        theta = self._start + numpy.sqrt(numpy.random.random(size=num_samples[0]))\
            * self._rotation
        d1 = numpy.array([
            -numpy.cos(theta) * theta + numpy.random.random(size=num_samples[0]) * self._noise,
            numpy.sin(theta) * theta + numpy.random.random(size=num_samples[0]) * self._noise]).T
        y1 = numpy.zeros(shape=(d1.shape[0]))

        theta = self._start + numpy.sqrt(numpy.random.random(size=num_samples[1]))\
            * self._rotation
        d2 = numpy.array([
            numpy.cos(theta) * theta + numpy.random.random(size=num_samples[1]) * self._noise,
            -numpy.sin(theta) * theta + numpy.random.random(size=num_samples[1]) * self._noise]).T
        y2 = numpy.ones(shape=(d2.shape[0]))

        X = numpy.concatenate((d1, d2), axis=0)
        y = numpy.concatenate((y1, y2), axis=0)

        return X, y


class TwoMoonsDataset(object):
    def __init__(self, Ns=1000, Nt=1000, class_ratio=[0.5, 0.5],
                 radius=0.5, noise=0.1,
                 Xs_shift=[0, 0], Xt_shift=[0, 0],
                 Xs_scale=[1, 1], Xt_scale=[1, 1],
                 rotation=None, common_idx=None, only_common=False):
        self._Ns = Ns
        self._Nt = Nt
        self._class_ratio = class_ratio
        self._radius = radius
        self._noise = noise

        self._Xs_shift = Xs_shift
        self._Xt_shift = Xt_shift
        self._Xs_scale = Xs_scale
        self._Xt_scale = Xt_scale
        self._rotation = rotation
        self._common_idx = common_idx
        self._only_common = only_common

    def gen_data(self):
        # ここで生成したデータを複数のモデルで使いたいタイミングがあるかもしれないがとりあえず考えない
        Xs, ys = self._gen_two_moons(self._Ns)
        Xt, yt = self._gen_two_moons(self._Nt)

        Xs = Xs * self._Xs_scale + self._Xs_shift
        Xt = Xt * self._Xt_scale + self._Xt_shift

        return Xs, ys, Xt, yt, self._common_idx

    def _gen_two_moons(self, N):
        num_samples = (N * numpy.array(self._class_ratio)).astype(int)
        # 円の半径が小さい方が密になるようにsqrtを一様分布からのサンプルにかけている
        theta = numpy.linspace(0, numpy.pi, num=num_samples[0])
        d1 = numpy.array([
            self._radius * numpy.cos(theta) +
            numpy.random.normal(size=num_samples[0]) * self._noise,
            self._radius * numpy.sin(theta) +
            numpy.random.normal(size=num_samples[0]) * self._noise]).T
        y1 = numpy.zeros(shape=(d1.shape[0]))

        theta = numpy.linspace(0, numpy.pi, num=num_samples[1])
        d2 = numpy.array([
            self._radius * numpy.cos(theta) +
            numpy.random.normal(size=num_samples[1]) * self._noise,
            -self._radius * numpy.sin(theta) +
            numpy.random.normal(size=num_samples[1]) * self._noise]).T
        y2 = numpy.ones(shape=(d2.shape[0]))

        d2 += [self._radius, self._radius / 2]

        X = numpy.concatenate((d1, d2), axis=0)
        y = numpy.concatenate((y1, y2), axis=0)

        if self._rotation is not None:
            rot = self._rotation / 180 * numpy.pi
            rot_matrix = numpy.array([
                [numpy.cos(rot), -numpy.sin(rot)],
                [numpy.sin(rot), numpy.cos(rot)]])
            X = rot_matrix.dot(X.T).T

        return X, y


class HAR70ArtificialShift(object):
    def __init__(self, target_id, test_ratio=0.2, random_state=None,
                 valid_labels=[1, 6, 7, 8],
                 num_shift_directions=10,
                 num_shift_iterations=50,
                 shift_mode='random',
                 common_columns=numpy.arange(6),
                 only_common=True, seed=None):
        self._target_id = target_id
        self._test_ratio = test_ratio
        self._random_state = random_state
        self._valid_labels = valid_labels
        self._num_shift_directions = num_shift_directions
        self._num_shift_iterations = num_shift_iterations
        self._shift_mode = shift_mode

        self._common_idx = numpy.array(common_columns)
        self._specific_idx = numpy.setdiff1d(numpy.arange(12), self._common_idx)
        self._only_common = only_common

    def _load_df(self, target_id):
        df = pandas.read_csv(f"./datasets/har70plus/{target_id}.csv")
        df = df.dropna(axis=0)
        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, -1].values.astype(int)

        idx = numpy.isin(y, self._valid_labels)

        X = X[idx]
        y = y[idx]

        return (X, y)

    def gen_data(self):
        import sklearn.model_selection
        X, y = self._load_df(self._target_id)
        Xs, Xt, ys, yt = sklearn.model_selection.train_test_split(
            X, y, test_size=self._test_ratio, random_state=self._random_state,
            shuffle=True,
            stratify=y)
        if self._shift_mode == 'random':
            Xt = ordered_shift(Xt, self._shift_mode,
                               shift_directions=self._num_shift_directions,
                               num_iterations=self._num_shift_iterations,
                               random_state=self._random_state)
        elif self._shift_mode == 'pca':
            Xt = ordered_shift(Xt, self._shift_mode,
                               shift_directions=self._num_shift_directions,
                               num_iterations=self._num_shift_iterations,
                               random_state=self._random_state)
        # Xs = Xs[:, self._common_idx]
        if self._only_common:
            Xt = Xt[:, self._common_idx]
        else:
            Xt = Xt[:, numpy.append(self._common_idx, self._specific_idx)]

        return Xs, ys, Xt, yt, self._common_idx


def ordered_shift(X, shift_mode, shift_directions=1, num_iterations=50, random_state=None):
    if random_state is not None:
        numpy.random.seed(random_state)
    import sklearn.decomposition

    if shift_mode == 'pca':
        pca = sklearn.decomposition.PCA(n_components=shift_directions)
        pca.fit(X)
        W = pca.components_
    elif shift_mode == 'random':
        W = numpy.random.normal(size=(shift_directions, X.shape[1]))
        W /= numpy.linalg.norm(W, 2, axis=1)[:, None]
    X2 = numpy.copy(X)

    for w in W:
        z = w.dot(X2.T)
        idx = numpy.argsort(z)
        # NOTE:データの順番を変えない方が良いかなぁ... 結局は各行に対応するoffsetが得られれば良い
        z = z[idx]
        # ここの端の量のcontrol
        width = z[-1] - z[0]
        z = numpy.append(numpy.append(z[0] - width, z), z[-1] + width)
        offset = numpy.zeros_like(z)
        # offset = numpy.ones(z.shape[0]) * 3

        # ネストが深くなるので後で関数化？
        for i in range(num_iterations):
            for j in numpy.random.permutation(numpy.arange(1, z.shape[0] - 1)):
                # zもoffsetも後で元の順番に戻すのでいいかなぁ...
                pos = z + offset
                low = pos[j - 1] - pos[j]
                high = pos[j + 1] - pos[j]
                offset[j] += numpy.random.uniform(low, high)
        X2[idx] = X2[idx] + offset[1:-1, None] * w

    return X2
