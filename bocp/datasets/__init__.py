# Import all dataset modules
import sys

# Other internal imports - from utils package
from utils.data import zip_expert_data, zip_model_data, select_models

# Synthetic
from .filepath import FilepathDataset
from .cifar_10h import Cifar10H
from .imagenet_16h import Imagenet16H
from .imagenet16h_distshift import Imagenet16hDistShift

def get_data(args):
    data = None
    if args.dataset:
        if args.dataset not in DATASET_ROSTER:
            args.data_path=args.dataset
            args.dataset_key = "filepath"
        else: args.dataset_key = args.dataset
        data_gen = DATASET_ROSTER[args.dataset_key](args)
        data_gen.get_data()
        data = data_gen.data

    if args.expert_data in DATASET_ROSTER:

        expert_data_gen = DATASET_ROSTER[args.expert_data](args)
        expert_data_gen.get_data()
        expert_data = expert_data_gen.data
        data = zip_expert_data(data, expert_data)
        true_targets = data['true_targets'] if data else None

    if args.expert_data == "synthetic":
        assert args.target_expert_perf is not None and (args.num_experts ==
                args.target_expert_perf.shape[0]),\
            (f"Number of experts is {args.num_experts} while queries is" +
             f" {args.num_experrt_queries}" +
             f" and params is {args.target_expert_perf.shape[0]}")
        expert_generator = SyntheticExpertGenerator(args)
        true_targets = data['true_targets'] if data else None
        expert_generator.get_data(true_targets)
        synth_expert_data = expert_generator.data
        data = zip_expert_data(data, synth_expert_data)
        # sys.exit(1)

    if args.model_data == "synthetic":
        assert args.target_model_perf is not None and (
            args.num_models == args.num_model_queries ==
            args.target_model_perf.shape[0]),\
            (f"Number of models is {args.num_models} while queries is {args.num_model_queries}" +
            f" and params is {args.target_model_perf}")
        model_generator = SyntheticModelGenerator(args)
        model_generator.get_data(data['targets'])
        synth_model_data = model_generator.data
        data = zip_model_data(data, synth_model_data)


    # If there are several models to choose from, select which to use
    # Either by random, performance, or direct ID
    select_models(args,data)

    return data

def noise_experts(args,data):
    noiser = ExpertNoiseWrapper(data, args)
    noiser.noise_experts()

def noise_models(args,data):
    noiser = ModelNoiseWrapper(data, args)
    noiser.noise_models()

#######################################################################
#
#######################################################################


DATASET_ROSTER = {
    "cifar10h": Cifar10H,
    "filepath": FilepathDataset,
    "imagenet16h": Imagenet16H,
    "imagenet16h_distshift":Imagenet16hDistShift,
}


