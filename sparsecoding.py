__author__ = 'scip'

import os

import numpy as np

lam = .4


# Block Coordinate Decent
# https://www.youtube.com/watch?v=UMdNfhgPKTc
def update_dictionary(data, representation, dictionary):
    length_of_data = data.shape[0]
    number_of_samples = data.shape[1]
    number_of_atoms = dictionary.shape[1]

    A = np.zeros([number_of_atoms, number_of_atoms])
    B = np.zeros([length_of_data, number_of_atoms])
    for t in range(number_of_samples):
        A += np.outer(representation[:, t], representation[:, t].T)
        B += np.outer(data[:, t], representation[:, t].T)

    zero_diag = A.diagonal() == 0
    if zero_diag.any():
        print zero_diag.all()
        mean_diag = A.diagonal().mean()
        print mean_diag
        for i, b in enumerate(zero_diag):
            if b:
                A[i, i] = mean_diag
                print i,

    for j in range(number_of_atoms):
        new_atom = B[:, j] - np.dot(dictionary, A[:, j]) + dictionary[:, j] * A[j, j]
        if A[j, j]:
            new_atom /= A[j, j]
        else:
            print 'oopsie:', j
        if np.isnan(new_atom).any():
            print 'nan atom error!'
        if np.isinf(new_atom).any():
            print 'inf atom error!'
        dictionary[:, j] = new_atom

    norm = np.linalg.norm(dictionary)
    norm_dict = dictionary / norm
    if np.isnan(norm_dict).any():
        print 'nan norm error!'
    if np.isinf(norm_dict).any():
        print 'inf norm error!'

    dictionary /= norm


def shrink(h, b):
    pass

def update_representation(data, representation, dictionary):
    number_of_atoms = dictionary.shape[1]

    reconstruction = np.dot(dictionary, representation)
    reconstruction_error = (reconstruction - data)

    s_data = data.shape
    s_repr = representation.shape
    s_dict = dictionary.shape


    dictionary_eigen_values = np.linalg.eigvals(np.dot(dictionary.T, dictionary))
    alp = 0.1 / abs(dictionary_eigen_values).max()

    sum_of_changes = np.inf
    run_count = 0

    # while not(sum_of_changes < .1 or run_count > 100):
    for _ in range(1):
        sum_of_changes = 0
        run_count += 1

        for k in range(number_of_atoms):
            reconstruction_update = alp * np.dot(dictionary[:, k].T, reconstruction_error)

            # for xxx in range(representation.shape[1]):
            #     if representation[k, xxx] == 0:
            #         if reconstruction_update[xxx]:
            #             print 'ru:', reconstruction_update[xxx]

            # print representation[k, :] / reconstruction_update
            # print reconstruction_update.mean()
            representation[k, :] -= reconstruction_update
        # less_than_alpha = representation < alp * lam
        # representation[less_than_alpha] = 0
            representation_sign = np.sign(representation[k, :])
            representation_update = representation_sign != np.sign(representation[k, :] - alp*lam*representation_sign)
            # print representation_update
            for t, b in enumerate(representation_update):
                if b:
                    # print 'gtz:', abs(representation[k, t])
                    sum_of_changes += abs(representation[k, t])
                    representation[k, t] = 0
                else:
                    sum_of_changes += abs(alp*lam*representation_sign[t])
                    representation[k, t] -= alp*lam*representation_sign[t]
    # print sum_of_changes,
    # print run_count,


if __name__ == '__main__':
    import makeupdata
    import matplotlib.pyplot as plt

    # new_dict = makeupdata.dictionary + np.random.rand(*makeupdata.dictionary.shape) * 10
    # new_repr = makeupdata.representation + np.random.rand(*makeupdata.representation.shape) * 10
    new_dict = np.ones(makeupdata.dictionary.shape)
    new_repr = np.ones(makeupdata.representation.shape)
    count = '_'

    for file_name in os.listdir('plots'):
        os.remove('plots/{}'.format(file_name))

    recons_error_list = []

    def plot():
        plt.subplot(5, 2, 1)
        plt.title('New Dict')
        plt.plot(new_dict)
        plt.subplot(5, 2, 2)
        plt.title('Dict')
        plt.plot(makeupdata.dictionary)
        plt.subplot(5, 2, 3)
        plt.title('New Repr')
        plt.plot(new_repr)
        plt.subplot(5, 2, 4)
        plt.title('Representations')
        plt.plot(makeupdata.representation)
        plt.subplot(5, 2, 5)
        plt.title('Reconstructions')
        plt.plot(np.dot(new_dict, new_repr))
        plt.subplot(5, 2, 6)
        plt.title('Data')
        plt.plot(makeupdata.data)
        plt.subplot(5, 1, 4)
        plt.title('Differences')
        plt.plot(makeupdata.data-np.dot(new_dict, new_repr))
        plt.subplot(5, 1, 5)
        plt.title('Differences')
        plt.yscale('log')
        plt.plot(recons_error_list)
        plt.savefig('plots/{}.png'.format(count))
        plt.close()
        # plt.show()

    plot()
    # print new_repr
    for count in range(100):
        for _ in range(10):
            for _ in range(1):
                update_dictionary(makeupdata.data, new_repr, new_dict)
            for _ in range(1):
                update_representation(makeupdata.data, new_repr, new_dict)
        recons = np.dot(new_dict, new_repr)
        reconstruction_error = (recons - makeupdata.data)
        err = (reconstruction_error**2).mean().mean()
        recons_error_list.append(err)
        print 'reer:', err
        plot()
        # print new_repr
