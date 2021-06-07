import argparse

from trainer import model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        help="Batch size for training steps",
        type=int,
        default=32
    )
    parser.add_argument(
        "--eval_data_path",
        help="GCS location pattern of eval files",
        required=True
    )
    parser.add_argument(
        "--nnsize",
        help="Hidden layer sizes (provide space-separated sizes)",
        nargs="+",
        type=int,
        default=[32, 8]
    )
    
    # TODO: add nbuckets hyperparameter
    
    parser.add_argument(
        "--nbuckets",
        help="number to define bucketized columns",
        type=int,
        default=16
    )
        

    parser.add_argument(
        "--lr",
        help = "learning rate for optimizer",
        type = float,
        default = 0.001
    )
    parser.add_argument(
        "--num_evals",
        help="Number of times to evaluate model on eval data training.",
        type=int,
        default=5
    )
    
    # TODO: add num_examples_to_train_on hyperparameter  
    
    parser.add_argument(
        "--num_examples_to_train_on",
        help="Number of examples to train on hyperparams.",
        type=int,
        default=100
    )

    parser.add_argument(
        "--output_dir",
        help="GCS location to write checkpoints and export models",
        required=True
    )
    parser.add_argument(
        "--train_data_path",
        help="GCS location pattern of train files containing eval URLs",
        required=True
    )
    parser.add_argument(
        "--job-dir",
        help="this model ignores this field, but it is required by gcloud",
        default="junk"
    )
    args = parser.parse_args()
    hparams = args.__dict__
    hparams.pop("job-dir", None)

    model.train_and_evaluate(hparams)
