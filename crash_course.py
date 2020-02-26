

def plot_decision_regions(clf, Xs, augmentor = lambda x: x):
    def graph_grid(xbounds, ybounds, step_size = 0.01):
        xs = np.arange(xbounds[0], xbounds[1], step_size)
        ys = np.arange(ybounds[0], ybounds[1], step_size)
        return np.meshgrid(xs, ys)

    xs = Xs[:, 0]
    ys = Xs[:, 1]
    xx, yy = graph_grid(graph_bounds(xs), graph_bounds(ys))
    preds = clf.predict(
        np.apply_along_axis(augmentor, 1, np.c_[xx.ravel(), yy.ravel()])
    )
    preds = preds.reshape(xx.shape)*2 - 1
    plt.contourf(xx, yy, preds, alpha=0.3)

