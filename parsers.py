#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import ast
import inspect
import json
import os
import re
from pathlib import Path
from queue import LifoQueue
from typing import Any, Dict, Generator, List, Tuple

# Libs
import torch as th

# Custom


##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

###############################
# Utility Class - ModelParser #
###############################

class ModelParser:
    """
    Given a model, parses model architecture to extract out model parameters 
    into a JSON compatible dictionary.

    Example:

    model_params = {
        'num_channels': 1,
        'num_filters': 64,
        'kernel_h': 7,
        'kernel_w': 3,
        'kernel_c': 1,
        'stride_conv': 1,
        'pool': 2,
        'stride_pool': 2,
        'num_class': 3,
        'epochs': 6
    }

    model = ReLayNet(model_params)
    parser = ModelParser(model)
    architecture = parser.parse(verbose=False)
    
    out_path = "/path/to/my/destination/architecture.json"
    parser.export(out_path)

    Attributes:
        model (th.nn.Module): Model to be parsed
        children (list): Cached list of children layers obtained by flattening
            the possibly nested architecture of specified model (Default: None)
        architecture (List[Dict[str, Any]]): Cached architecture obtained by
            parsing the specified model (Default: None) 
    """
    def __init__(self, model: th.nn.Module):
        self.model = model
        self.children = None
        self.architecture = None

    ###########
    # Helpers #
    ###########

    def load_children(self) -> Generator[th.nn.Module, None, None]:
        """ Iteratively looks through nested model layers to obtain list 
            of flattened child layers.

            Note: Process is idempotent.

        Args:
            model (th.nn.Module): model architecture

        Returns:
            List[th.nn.Module]: List of model children layers
        """
        if not self.children:

            _exhausted  = object()

            stack = LifoQueue()
            stack.put(self.model)
            
            layers_reversed = []
            while not stack.empty():

                curr_module = stack.get()

                if next(curr_module.children(), _exhausted) is _exhausted:
                    layers_reversed.append(curr_module)

                else:
                    for child_module in curr_module.children():
                        stack.put(child_module)
                        
            self.children = reversed(layers_reversed)

        return self.children


    def detect_layer_name(self, layer: th.nn.Module)-> str:
        """ Function to iterate through string of model children layers to 
            obtain functional signature of model children layers

        Args:
            lines (str): Model children layers

        Returns:
            dict_keys (list): functional signature of model children layers
        """
        return layer.__class__.__name__


    def detect_layer_signature(
        self, 
        layer: th.nn.Module
    )-> Tuple[List[str], Dict[str, Any]]:
        """ Function to iterate through string of model children layers to 
            obtain functional signature of model children layers

        Args:
            lines (str): Model children layers

        Returns:
            dict_keys (list): functional signature of model children layers
        """
        layer_class = layer.__class__
        layer_signature = inspect.signature(layer_class).parameters
        layer_signature_keys = [*layer_signature.keys()]

        layer_signature_defaults = {
            k: v.default
            for k, v in layer_signature.items()
            if v.default is not inspect.Parameter.empty
        }

        return layer_signature_keys, layer_signature_defaults


    def detect_layer_arguments(
        self, 
        layer: th.nn.Module
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """ Perform regex splitting to remove th.nn functions ie. Conv2d, 
            BatchNorm from model layers

        Args:
            header (str)): Model layers eg. Conv2d['kernel_size=(2, 2)', 
                'stride=(2, 2)', 'padding=(0, 0)']

        Returns:
            list : eg. ['kernel_size=(2, 2)', 'stride=(2, 2)', 'padding=(0, 0)']
        """
        header = str(layer)
        # print(header)

        # (?<=\(): positive lookbehind to match the open bracket `(`
        # (?=\)) : positive lookahead to match the close bracket `)`
        signature_pattern = '(?<=\().+(?=\))'
        signature = re.search(signature_pattern, header)

        detected_args = []
        detected_kwargs = {}

        if signature:
            detect_value = "[A-Za-z0-9_@./#&+-]+"
            detect_kwarg = "=?"
            detect_tuple = "\(.*?,.*?\)"
            params_pattern = f"({detect_value})(?:{detect_kwarg})({detect_tuple}|{detect_value})?"
            parameters = re.findall(params_pattern, signature.group())

            for val1, val2 in parameters:
                
                # Case 1: If val2 is "", then val1 is a positional argument
                if not val2:
                    detected_args.append(ast.literal_eval(val1))

                # Case 2: Otherwise val1 is the keyword, and val2 is the value
                else:
                    detected_kwargs[val1] = ast.literal_eval(val2)

        return detected_args, detected_kwargs


    ##################
    # Core Functions #
    ##################

    def parse(self, verbose: bool = False)-> Dict[str, Any]:
        """ Main function to parse model.

            Note: Process is idempotent.

        Returns:
            Dict[str, Any]: Dictionary representing parsed model architecture
        """
        self.load_children()

        if not self.architecture:

            parsed_layers = []
            for layer_idx, layer in enumerate(self.children):

                module = {}

                # 1. Flag if current module is the input layer
                module["is_input"] = (layer_idx == 0)

                layer_name = self.detect_layer_name(layer) 
                layer_signature, defaults = self.detect_layer_signature(layer)

                # print(f"layer_name: {layer_name}")
                # print(f"layer_signature: {layer_signature}")

                layer_args, layer_kwargs = self.detect_layer_arguments(layer)
                
                # 2. Populate the layer name
                module["l_type"] = layer_name
                structure = defaults if verbose else {}

                # 3a. Align the positional arguments
                for pos_param, pos_arg in zip(layer_signature, layer_args):
                    structure[pos_param] = pos_arg
                # 3b. Populate the remaining keyword arguments
                structure.update(layer_kwargs)

                ##################################################################
                # Implementation Footnote - Deactivate activation-layer coupling #
                ##################################################################

                # [Cause]
                # Auto-detection casts all activation layers as modules 
                # (i.e. th.nn.Module) instead of functions (i.e. th.nn.functional)

                # [Problems]
                # More work needs to be put in to cast detected modules into their
                # relevant functional counterparts, with little added benefit, even
                # though this syntax is applicable for manually defined architectures

                # [Solution]
                # Leave activation key as None. Let Synergos' parser read all
                # layers of a model that has been auto-parsed as Modules 

                # 4. Define the activation function (if any)
                module["activation"] = None
                
                module["structure"] = structure
                parsed_layers.append(module)

            self.architecture = parsed_layers

        return self.architecture
      

    def export(self, dst_dir: str, dst_filename: str = "architecture") -> str:
        """ Exports the cached architecture as a JSON file. Raises a
            RuntimeError if architecture has not been parsed.

        Args:
            dst_dir (str): Destination directory to export architecture to
            dst_filename (str): Filename of exported architecture
        
        Returns:
            (str): Path to generated architecture
        """
        if not self.architecture:
            raise RuntimeError("Model architecture has not been parsed! Please parse and try again!")

        out_dir = Path(dst_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, f"{dst_filename}.json")

        with open(out_path, 'w') as fp:
            json.dump(self.architecture, fp, sort_keys=True, indent=4)

        return out_path