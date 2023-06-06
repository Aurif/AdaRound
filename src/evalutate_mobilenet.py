
########################################################################################
# from Examples.common.utils import accuracy
########################################################################################
# =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
#
# =============================================================================

"""
General utility functions for AIMET examples
"""


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
########################################################################################

import sys

import os
import torch

##############################################################################################################
# from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator
##############################################################################################################


# !/usr/bin/env python
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================


"""
Creates Evaluator for Image-Net dataset
"""
import logging

import progressbar
import torch
from torch import nn

logger = logging.getLogger('Eval')


class ImageNetEvaluator:
    """
    For validation of a trained model using the ImageNet dataset.
    """

    def __init__(self, images_dir: str, image_size: int, batch_size: int = 128,
                 num_workers: int = 32, num_val_samples_per_class: int = None):
        """
        :param images_dir: The path to the data directory
        :param image_size: The length of the image
        :param batch_size: The batch size to use for training and validation
        :param num_workers: Indiicates to the data loader how many sub-processes to use for data loading.
        :param num_train_samples_per_class: Number of samples to use per class.
        """
        self._val_data_loader = ImageNetDataLoader(images_dir,
                                                   image_size=image_size,
                                                   batch_size=batch_size,
                                                   is_training=False,
                                                   num_workers=num_workers,
                                                   num_samples_per_class=num_val_samples_per_class).data_loader

    def evaluate(self, model: nn.Module, iterations: int = None, use_cuda: bool = False) -> float:
        """
        Evaluate the specified model using the specified number of samples batches from the
        validation set.
        :param model: The model to be evaluated.
        :param iterations: The number of batches to use from the validation set.
        :param use_cuda: If True then use a GPU for inference.
        :return: The accuracy for the sample with the maximum accuracy.
        """

        device = torch.device('cpu')
        if use_cuda:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                logger.error('use_cuda is selected but no cuda device found.')
                raise RuntimeError("Found no CUDA Device while use_cuda is selected")

        if iterations is None:
            logger.info('No value of iteration is provided, running evaluation on complete dataset.')
            iterations = len(self._val_data_loader)
        if iterations <= 0:
            logger.error('Cannot evaluate on %d iterations', iterations)

        acc_top1 = 0
        acc_top5 = 0

        logger.info("Evaluating nn.Module for %d iterations with batch_size %d",
                    iterations, self._val_data_loader.batch_size)

        model = model.to(device)
        model = model.eval()

        batch_cntr = 1
        with progressbar.ProgressBar(max_value=iterations) as progress_bar:
            with torch.no_grad():
                for input_data, target_data in self._val_data_loader:

                    inputs_batch = input_data.to(device)
                    target_batch = target_data.to(device)

                    predicted_batch = model(inputs_batch)

                    batch_avg_top_1_5 = accuracy(output=predicted_batch, target=target_batch,
                                                 topk=(1, 5))

                    acc_top1 += batch_avg_top_1_5[0].item()
                    acc_top5 += batch_avg_top_1_5[1].item()

                    progress_bar.update(batch_cntr)

                    batch_cntr += 1
                    if batch_cntr > iterations:
                        break

        acc_top1 /= iterations
        acc_top5 /= iterations

        logger.info('Avg accuracy Top 1: %f Avg accuracy Top 5: %f on validation Dataset',
                    acc_top1, acc_top5)

        return acc_top1
###################################################################################################################


###################################################################################################################
#from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader 
###################################################################################################################
# !/usr/bin/env python
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

"""
Creates data-loader for Image-Net dataset
"""
import logging
import os

from torchvision import transforms
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
from torch.utils.data import Dataset
import torch.utils.data as torch_data


logger = logging.getLogger('Dataloader')

IMG_EXTENSIONS = '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif'


def make_dataset(directory: str, class_to_idx: dict, extensions: tuple, num_samples_per_class: int) -> list:
    """
    Creates a dataset of images with num_samples_per_class images in each class

    :param directory: The string path to the data directory.
    :param class_to_idx: A dictionary mapping the name of the class to the index (label)
    :param extensions: list of valid extensions to load data
    :param num_samples_per_class: Number of samples to use per class.

    :return: list of images containing the entire dataset.
    """
    images = []
    num_classes = 0
    directory = os.path.expanduser(directory)
    for class_name in sorted(class_to_idx.keys()):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            class_idx = class_to_idx[class_name]
            class_images = add_images_for_class(class_path, extensions, num_samples_per_class, class_idx)
            images.extend(class_images)
            num_classes += 1
        if num_samples_per_class and num_classes >= num_samples_per_class:
            break

    logger.info("Dataset consists of %d images in %d classes", len(images), num_classes)
    return images


def add_images_for_class(class_path: str, extensions: tuple, num_samples_per_class: int, class_idx: int) -> list:
    """
    For a given class, adds num_samples_per_class images to a list.

    :param class_path: The string path to the class directory.
    :param extensions: List of valid extensions to load data
    :param num_samples_per_class: Number of samples to use per class.
    :param class_idx: numerical index of class.

    :return: list of images for given class.
    """
    class_images = []
    count = 0
    for file_name in os.listdir(class_path):
        if num_samples_per_class and count >= num_samples_per_class:
            break
        if has_file_allowed_extension(file_name, extensions):
            image_path = os.path.join(class_path, file_name)
            item = (image_path, class_idx)
            class_images.append(item)
            count += 1

    return class_images


class ImageFolder(Dataset):
    """
    Dataset class inspired by torchvision.datasets.folder.DatasetFolder for images organized as
        individual files grouped by category.
    """

    def __init__(self, root: str, transform=None, target_transform=None,
                 num_samples_per_class: int = None):

        """
        :param root: The path to the data directory.
        :param transform: The required processing to be applied on the sample.
        :param target_transform:  The required processing to be applied on the target.
        :param num_samples_per_class: Number of samples to use per class.

        """
        Dataset.__init__(self)
        classes, class_to_idx = self._find_classes(root)
        self.samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS, num_samples_per_class)
        if not self.samples:
            raise (RuntimeError(
                "Found 0 files in sub folders of: {}\nSupported extensions are: {}".format(
                    root, ",".join(IMG_EXTENSIONS))))

        self.root = root
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in self.samples]

        self.transform = transform
        self.target_transform = target_transform

        self.imgs = self.samples

    @staticmethod
    def _find_classes(directory: str):
        classes = [d for d in os.listdir(directory) if
                   os.path.isdir(os.path.join(directory, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


class ImageNetDataLoader:
    """
    For loading Validation data from the ImageNet dataset.
    """

    def __init__(self, images_dir: str, image_size: int, batch_size: int = 128,
                 is_training: bool = False, num_workers: int = 8, num_samples_per_class: int = None):
        """
        :param images_dir: The path to the data directory
        :param image_size: The length of the image
        :param batch_size: The batch size to use for training and validation
        :param is_training: Indicates whether to load the training or validation data
        :param num_workers: Indiicates to the data loader how many sub-processes to use for data loading.
        :param num_samples_per_class: Number of samples to use per class.
        """

        # For normalization, mean and std dev values are calculated per channel
        # and can be found on the web.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        self.val_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize])

        if is_training:
            data_set = ImageFolder(
                root=os.path.join(images_dir, 'train'), transform=self.train_transforms,
                num_samples_per_class=num_samples_per_class)
        else:
            data_set = ImageFolder(
                root=os.path.join(images_dir, 'val'), transform=self.val_transforms,
                num_samples_per_class=num_samples_per_class)

        self._data_loader = torch_data.DataLoader(
            data_set, batch_size=batch_size, shuffle=is_training,
            num_workers=num_workers, pin_memory=True)

    @property
    def data_loader(self) -> torch_data.DataLoader:
        """
        Returns the data-loader
        """
        return self._data_loader
###################################################################################################################

class ImageNetDataPipeline:

    @staticmethod
    def get_val_dataloader(DATASET_DIR, N, batch_size, image_size) -> torch.utils.data.DataLoader:
        """
        Instantiates a validation dataloader for ImageNet dataset and returns it
        """
        data_loader = ImageNetDataLoader(DATASET_DIR,
                                         image_size=image_size,
                                         batch_size=batch_size,
                                         is_training=False,
                                         num_workers=1,
                                         num_samples_per_class=N).data_loader
        return data_loader

    @staticmethod
    def evaluate(model: torch.nn.Module, use_cuda: bool, DATASET_DIR, N, batch_size, image_size) -> float:
        """
        Given a torch model, evaluates its Top-1 accuracy on the dataset
        :param model: the model to evaluate
        :param use_cuda: whether or not the GPU should be used.
        """
        evaluator = ImageNetEvaluator(DATASET_DIR, image_size=image_size,
                                      batch_size=batch_size,
                                      num_workers=1,
                                      num_val_samples_per_class=N)

        return evaluator.evaluate(model, iterations=None, use_cuda=use_cuda)
    

def get_pass_calibration_data(SAMPLES_TO_COMPUTE_ENCODINGS, DATASET_DIR, N, batch_size, image_size):
    def pass_calibration_data(sim_model, use_cuda, DATASET_DIR, N, batch_size, image_size):
        data_loader = ImageNetDataPipeline.get_val_dataloader(DATASET_DIR, N, batch_size, image_size)
        batch_size = data_loader.batch_size

        if use_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        sim_model.eval()
        samples = SAMPLES_TO_COMPUTE_ENCODINGS

        batch_cntr = 0
        with torch.no_grad():
            for input_data, target_data in data_loader:

                inputs_batch = input_data.to(device)
                sim_model(inputs_batch)

                batch_cntr += 1
                if (batch_cntr * batch_size) > samples:
                    break
    return lambda sim_model, use_cuda: pass_calibration_data(sim_model, use_cuda, DATASET_DIR, N, batch_size, image_size)

###########################################################################################################

###########################################################################################################

###########################################################################################################

def run_experiment(model, ROOT_DATA_AND_OUTPUTS, DATASET_FOLDER_PATH, N, IMAGE_SIZE, BATCH_SIZE, BIWIDTH,
                   BIWIDTH_ACTIVATION, SAMPLES_TO_COMPUTE_ENCODINGS, ADAROUND_ITERATIONS, ADAROUND_NUM_BATCHES,
                   model_name, output_dir):


    import os
    DATASET_DIR = f'{ROOT_DATA_AND_OUTPUTS}{DATASET_FOLDER_PATH}'
    os.makedirs(output_dir, exist_ok=True)


    from aimet_torch.model_preparer import prepare_model
    model = prepare_model(model)
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        model.to(torch.device('cuda'))
    print('Using cuda: {}'.format(use_cuda))

    pass_calibration_data = get_pass_calibration_data(SAMPLES_TO_COMPUTE_ENCODINGS, DATASET_DIR, N, BATCH_SIZE, IMAGE_SIZE)

    ############################################################### original accuracy - start
    accuracy = ImageNetDataPipeline.evaluate(model, use_cuda, DATASET_DIR, N, BATCH_SIZE, IMAGE_SIZE)
    print(accuracy)

    original_accuracy = accuracy
    ############################################################### original accuracy - end
    
    from aimet_torch.batch_norm_fold import fold_all_batch_norms

    _ = fold_all_batch_norms(model, input_shapes=(1, 3, IMAGE_SIZE, IMAGE_SIZE))

    from aimet_common.defs import QuantScheme
    from aimet_torch.quantsim import QuantizationSimModel

    dummy_input = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)   
    if use_cuda:
        dummy_input = dummy_input.cuda()

    ############################################################### quantized accuracy - nearest - start
    sim = QuantizationSimModel(model=model,
                            quant_scheme=QuantScheme.post_training_tf_enhanced,
                            dummy_input=dummy_input,
                            rounding_mode = 'nearest',
                            default_output_bw=BIWIDTH_ACTIVATION,
                            default_param_bw=BIWIDTH)
    

    sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                        forward_pass_callback_args=use_cuda)

    accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda, DATASET_DIR, N, BATCH_SIZE, IMAGE_SIZE)
    print(accuracy)
    quantized_accuracy_nearest = accuracy
    ############################################################### quantized accuracy - nearest - end


    ############################################################### quantized accuracy - stochastic - start
    sim = QuantizationSimModel(model=model,
                            quant_scheme=QuantScheme.post_training_tf_enhanced,
                            dummy_input=dummy_input,
                            rounding_mode = 'stochastic',
                            default_output_bw=BIWIDTH_ACTIVATION,
                            default_param_bw=BIWIDTH)
    

    sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                        forward_pass_callback_args=use_cuda)

    accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda, DATASET_DIR, N, BATCH_SIZE, IMAGE_SIZE)
    print(accuracy)
    quantized_accuracy_stochastic = accuracy
    ############################################################### quantized accuracy - stochastic - end

    ############################################################### quantized accuracy - adaround - start

    from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters

    data_loader = ImageNetDataPipeline.get_val_dataloader(DATASET_DIR, N, BATCH_SIZE, IMAGE_SIZE)
    params = AdaroundParameters(data_loader=data_loader, num_batches=ADAROUND_NUM_BATCHES, default_num_iterations=ADAROUND_ITERATIONS)

    dummy_input = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    if use_cuda:
        dummy_input = dummy_input.cuda()

    ada_model = Adaround.apply_adaround(model, dummy_input, params,
                                        path=f'{output_dir}', 
                                        filename_prefix='adaround', 
                                        default_param_bw=BIWIDTH,
                                        default_quant_scheme=QuantScheme.post_training_tf_enhanced)
    
    sim = QuantizationSimModel(model=ada_model,
                            dummy_input=dummy_input,
                            quant_scheme=QuantScheme.post_training_tf_enhanced,
                            default_output_bw=BIWIDTH_ACTIVATION, 
                            default_param_bw=BIWIDTH)

    sim.set_and_freeze_param_encodings(encoding_path=f'{output_dir}adaround.encodings')

    sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                        forward_pass_callback_args=use_cuda)
    
    accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda, DATASET_DIR, N, BATCH_SIZE, IMAGE_SIZE)
    print(accuracy)
    quantized_accuracy_adaround = accuracy

    from aimet_torch import quantsim 
    quantsim.save_checkpoint(quant_sim_model=sim, file_path=f'{output_dir}/sim_after_adaround_checkpoint.pth')

    ############################################################### quantized accuracy - adaround - end
    
    results = {
        'original_accuracy': original_accuracy,
        'quantized_accuracy_nearest': quantized_accuracy_nearest,
        'quantized_accuracy_stochastic': quantized_accuracy_stochastic,
        'quantized_accuracy_adaround': quantized_accuracy_adaround,
        'BIWIDTH': BIWIDTH,
        'BIWIDTH_ACTIVATION': BIWIDTH_ACTIVATION,
        'SAMPLES_TO_COMPUTE_ENCODINGS': SAMPLES_TO_COMPUTE_ENCODINGS,
        'ADAROUND_NUM_BATCHES': ADAROUND_NUM_BATCHES,
        'ADAROUND_ITERATIONS': ADAROUND_ITERATIONS,
        'BATCH_SIZE': BATCH_SIZE,
        'IMAGE_SIZE': IMAGE_SIZE,
        'N': N,
        'DATASET_DIR': DATASET_DIR,
        'output_dir': output_dir,
        'MODEL_NAME': model_name
    }
    return results

################################################################################################ end of functions. usage below


# Set the parameters below

# place data and set the path below to folder containing train and val folders

# results csv is printed to stdout in the last 2 lines of the output, ready to be copied elsewhere
# also, results csv is saved in the output folder


def main(argv):


    print('Number of arguments:', len(argv), 'arguments.')
    print('Argument List:', str(argv))

    # available model names
    # ['cifar100_mobilenetv2_x0_5',
    #  'cifar100_mobilenetv2_x0_75',
    #  'cifar100_mobilenetv2_x1_0',
    #  'cifar100_mobilenetv2_x1_4',
    #  'cifar10_mobilenetv2_x0_5',
    #  'cifar10_mobilenetv2_x0_75',
    #  'cifar10_mobilenetv2_x1_0',
    #  'cifar10_mobilenetv2_x1_4']
    model_name = 'cifar10_mobilenetv2_x0_5'

    import torch
    m = torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True)
    if model_name not in m:
        print(f"model {model_name} not found in {m}")
        exit(1)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True)


    ROOT_DATA_AND_OUTPUTS = './'
    DATASET_FOLDER_PATH = 'cifar10-data/data/'
    N = 16 # number of classes and samples per class
    IMAGE_SIZE = 32
    BATCH_SIZE = 32

    BIWIDTH = 4
    BIWIDTH_ACTIVATION = 8 #quantization on the input and output of the layer
    
    SAMPLES_TO_COMPUTE_ENCODINGS = 1000 # default 1000

    # batch size of evaluation is 32
    # in the paper they used 2048 images -> 64 batches !!!!!!!!!
    ADAROUND_ITERATIONS = 2
    ADAROUND_NUM_BATCHES = 2

    ############################33 main computations

    import datetime
    output_dir = f'{ROOT_DATA_AND_OUTPUTS}output_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}/'

    results = run_experiment(model, ROOT_DATA_AND_OUTPUTS, DATASET_FOLDER_PATH, N, IMAGE_SIZE, BATCH_SIZE, BIWIDTH,
                     BIWIDTH_ACTIVATION, SAMPLES_TO_COMPUTE_ENCODINGS, ADAROUND_ITERATIONS, ADAROUND_NUM_BATCHES,
                     model_name, output_dir)

    ################################## printing results to file as csv

    output_path = f'{output_dir}/results.csv'
    resuts = ['','']
    for key, value in results.items():
        resuts[0] += f'{key};'
        resuts[1] += f'{value};'
    resuts[0] += '\n'
    with open(output_path, 'w') as f:
        f.writelines(resuts)
    for line in resuts:
        print(line)
        



if __name__ == '__main__':
    main(sys.argv[1:])
    