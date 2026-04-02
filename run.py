import hydra
from omegaconf import DictConfig
import datasets
import numpy
import scipy.spatial.distance
import sklearn.neighbors
import models
import networkx
import matplotlib.pyplot as plt
import utils
import mlflow
mlflow.set_tracking_uri(uri="http://localhost:50000")
mlflow.set_experiment("DAEVS_GW")


def create_dataset(name: str, cfg: DictConfig):
    subcfg = getattr(cfg, name)
    if name == 'linear':
        return datasets.LinearDataset(**subcfg)
    elif name == 'two_circles':
        return datasets.TwoBallsDataset(**subcfg)
    elif name == 'two_spirals':
        return datasets.TwoSpiralsDataset(**subcfg)
    elif name == 'two_moons':
        return datasets.TwoMoonsDataset(**subcfg)
    elif name == 'har70_artificial_shift':
        return datasets.HAR70ArtificialShift(**subcfg)


def create_model(name: str, cfg: DictConfig):
    if name == 'simple_baseline':
        subcfg = getattr(cfg, name)
        source_target_cost_calculator = sqeuclidean_cost_calculator
        model = models.ClassificationAndMajorityVote(
            source_target_cost_calculator, n_clusters=subcfg.n_clusters, affinity=subcfg.affinity,
            n_neighbors=subcfg.n_neighbors, gamma=subcfg.gamma)

        return model

    source_cost_calculator = sqeuclidean_cost_calculator
    label_cost_calculator = sqeuclidean_cost_calculator
    source_target_cost_calculator = sqeuclidean_cost_calculator
    target_cost_calculator = sqeuclidean_cost_calculator

    if name == 'OT':
        alpha = 0
        beta = 0
    elif name == 'GWOT':
        alpha = 1
        beta = 0
    elif name == 'FGW':
        alpha = cfg.alpha
        beta = 0
    elif name == 'LA-FGW':
        alpha = cfg.alpha
        beta = cfg.beta
    elif name == 'GB-FGW':
        alpha = cfg.alpha
        beta = 0
        target_cost_calculator = GraphCostCalculator(n_neighbors=cfg.n_neighbors).graph_cost_calculator
    elif name == 'LAGB-FGW':
        alpha = cfg.alpha
        beta = cfg.beta
        target_cost_calculator = GraphCostCalculator(n_neighbors=cfg.n_neighbors).graph_cost_calculator

    model = models.FusedGromovWasserstein(
        source_cost_calculator,
        target_cost_calculator,
        label_cost_calculator,
        source_target_cost_calculator, alpha=alpha, beta=beta)

    return model


def sqeuclidean_cost_calculator(x, y=None):
    if y is None:
        y = x
    # D = sklearn.metrics.pairwise_distances(x, y, metric='sqeuclidean')
    D = scipy.spatial.distance.cdist(x, y, metric='sqeuclidean')

    return D / len(x)


class GraphCostCalculator(object):
    def __init__(self, n_neighbors, max_cost=10):
        self._n_neighbors = n_neighbors
        self._max_cost = max_cost

    def graph_cost_calculator(self, x):
        G = networkx.from_numpy_array(sklearn.neighbors.kneighbors_graph(
            x, n_neighbors=self._n_neighbors, mode='distance', metric='sqeuclidean'))
        C = numpy.ones(shape=(x.shape[0], x.shape[0])) * self._max_cost

        for (k1, v1) in networkx.shortest_path_length(G, weight='weight'):
            for (k2, v2) in v1.items():
                C[k1, k2] = v2

        return C


@hydra.main(config_name="config", version_base=None, config_path=".")
def main(cfg: DictConfig) -> None:
    if 'seed' in cfg:
        numpy.random.seed(cfg.seed)

    ds = create_dataset(name=cfg.dataset, cfg=cfg)

    # source_cost_calculator = sqeuclidean_cost_calculator
    # label_cost_calculator = sqeuclidean_cost_calculator
    # source_target_cost_calculator = sqeuclidean_cost_calculator
    # if cfg.target_cost == 'sqeuclidean':
    #     target_cost_calculator = sqeuclidean_cost_calculator
    # elif cfg.target_cost == 'graph':
    #     target_cost_calculator = GraphCostCalculator(n_neighbors=cfg.n_neighbors).graph_cost_calculator

    # model = models.FusedGromovWasserstein(
    #     source_cost_calculator,
    #     target_cost_calculator,
    #     label_cost_calculator,
    #     source_target_cost_calculator, alpha=cfg.alpha, beta=cfg.beta)
    model = create_model(cfg.model, cfg)
    Xs0, ys, Xt, yt, common_idx = ds.gen_data()
    Xs = Xs0[:, common_idx]

    with mlflow.start_run():
        mlflow.log_params(cfg)

        model.fit(Xs, Xt, ys, confidence_threshold=0)
        yt_est = model.predict()
        acc = numpy.mean(yt == yt_est)
        mlflow.log_metric(key='Accuracy', value=acc)
        numpy.savez("data.npz", Xs0=Xs0, ys=ys, Xt=Xt, yt=yt,
                    common_idx=common_idx, yt_est=yt_est)
        mlflow.log_artifact("data.npz")

        if cfg.make_plot:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111)
            # utils.plot2D(ax, numpy.column_stack((Xs[:, 0], numpy.zeros_like(Xs))), ys)
            utils.plot2D(ax, Xs0, ys)
            # ax.set_ylim([-.1, .1])
            mlflow.log_figure(fig, 'source.png')
            plt.close('all')

            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111)
            utils.plot2D(ax, Xt, yt)
            ax.set_xlabel('$x^c$')
            ax.set_ylabel('$x^a$')
            plt.tight_layout()
            mlflow.log_figure(fig, 'target_gt.pdf')
            plt.close('all')

            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111)
            utils.plot2D(ax, Xt, yt_est)
            ax.set_xlabel('$x^c$')
            ax.set_ylabel('$x^a$')
            plt.tight_layout()
            mlflow.log_figure(fig, 'target_estimated.pdf')
            plt.close('all')


if __name__ == '__main__':
    main()
