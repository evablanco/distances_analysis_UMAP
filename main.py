import argparse
import exps_visualization_toy_data, exps_word_embeddings, exps_f_mnist_ovelap_ratio, exps_classification

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_type", type=int, default=0,
                        help="0: visualization toy data, 1: visualization word2vec, 2: visualization fmnist, 3: classification.")
    #  TO-DO: set distance measures
    args = parser.parse_args()
    print(args)

    exp_type = args.exp_type

    if exp_type == 1:
        exps_visualization_toy_data.start_visualization_toy_exps()
    elif exp_type == 2:
        exps_word_embeddings.start_exps_word_embeddings()
    elif exp_type == 3:
        exps_f_mnist_ovelap_ratio.start_exps_fmnist()
    elif exp_type == 4:
        exps_classification.start_exps_classification()
    else:
        print('Not implemented yet...')