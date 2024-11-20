import pandas as pd
import numpy as np

from arpa.plot.graph import draw_tree, draw_components
from arpa.transformer.FeatureSelectionTransformer import ARPA
from arpa.shannon_information import shannon
import networkx as nx



def create_redutant_varaible(A, B, c_1, c_2, noise):
    n_samples = len(A)
    variable = c_1*A + c_2*B + noise * np.random.random_sample((n_samples, ))
    return variable.reshape(n_samples, 1)


if __name__ == '__main__':
    # Set a random seed for reproducibility
    np.random.seed(42)
    n_informative = 2
    n_redundant = 4
    n_repeated = 1
    n_unusefull = 1

    n_features = n_informative + n_redundant + n_repeated + n_unusefull

    # Number of data points for each class
    num_samples = 300

    # Mean and covariance matrix for Class 0
    mean_class0 = np.array([2, 2])
    cov_class0 = np.array([[1, 0.7], [0.7, 1]])

    # Mean and covariance matrix for Class 1
    mean_class1 = np.array([-2, -2])
    cov_class1 = np.array([[1, -0.5], [-0.5, 1]])

    # Generate data for Class 0
    X_class0 = np.random.multivariate_normal(mean_class0, cov_class0,
                                             num_samples)
    y_class0 = np.zeros(num_samples, dtype=int)

    # Generate data for Class 1
    X_class1 = np.random.multivariate_normal(mean_class1, cov_class1,
                                             num_samples)
    y_class1 = np.ones(num_samples, dtype=int)

    # Combine the data from both classes
    X = np.vstack((X_class0, X_class1))
    target = np.hstack((y_class0, y_class1))

    # Add some random noise to the data set
    X += 1.5 * np.random.randn(X.shape[0], X.shape[1])

    r1 = create_redutant_varaible(X[:, 0], X[:, 1], 0.45, -0.5, 1)
    r2 = create_redutant_varaible(X[:, 0], X[:, 1], 0.85, -0.95, 1)
    r3 = create_redutant_varaible(X[:, 0], X[:, 1], -0.12, 0.75, 1)
    r4 = create_redutant_varaible(X[:, 0], X[:, 1], -0.0, -0.3, 1)
    repeated = X[:, 0].reshape(-1, 1)
    unusefull = np.random.random_sample((2 * num_samples, 1))
    X = np.concatenate([X, r1, r2, r3, r4, repeated, unusefull], axis=1)

    labels = [f"informative_{n}" for n in range(n_informative)]
    labels.extend([f"redundant_{n}" for n in range(n_redundant)])
    labels.extend([f"repeated_{n}" for n in range(n_repeated)])
    labels.extend([f"unusefull_{n}" for n in range(n_unusefull)])

    k = 2
    verbose = 1
    max_time_in_seconds = 10 * 60
    processes = 6

    # MAXIMIZE
    m = shannon.process_call_corrcoef
    costs = shannon.calculate_information_matrix(matrix=X,
                                                 target=target,
                                                 func=m,
                                                 is_symmetric=True,
                                                 discretize=False,
                                                 verbose=verbose,
                                                 processes=6)
    pandas_costs = pd.DataFrame(data=costs.copy(), index=labels,
                                columns=labels)

    arpa = ARPA(k=k,
               costs=pandas_costs.values,
               verbose=verbose,
               processes=processes,
               )
    arpa.fit(X, target)
    print(
        f"Total number of variables: {np.logical_not(np.isnan(costs)).sum()}")
    print(
        f"{arpa.cols} for metric: {m.__name__} and {arpa.elapsed_time} seconds")
    model = arpa.model
    node_iteration = nx.get_node_attributes(model['components'], 'iteration')

    components = model["iterations"]
    T = model["tree"]

    mapping = {v: labels[v] for v in range(len(pandas_costs))}

    print("Draw graph for each iteration")
    for c in components:
        draw_components(c["G"].to_undirected(), round=c["round"], mapping=mapping)

    print("Draw final tree")
    root = mapping[model["root"]]
    node_list = [mapping[node] for node in np.sort(model["sorted_nodes"])]
    draw_tree(T.to_undirected(), root=root, node_list=node_list, mapping=mapping)

