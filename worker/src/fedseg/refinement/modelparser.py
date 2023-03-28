# from models.relaynet.relay_net import ReLayNet
from relay_net import ReLayNet
# from model4c import HRNetRefine
# from channelconfig import get_cfg_defaults
import json
import re
import torch
import inspect

"""
This is how you run the class

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
layers_json = parser.parse_model()


"""

class ModelParser():
   """Parses model architecture to extract out model params in json dictionary
      Args:
         model(torch.nn): model architecture
         activations_list (list): list of possible activation functions
         children (list): List of children layers obtained by recursive iteration of model layers
   """
   def __init__(self,model: torch.nn) -> None:
      self.model = model
      self.activations_list = ["relu","sigmoid"]
      self.children = None

   @staticmethod
   def get_children(model: torch.nn.Module)-> list:
      """Function to recursively iterate through model layers to obtain list of flattened children 

      Args:
         model (torch.nn.Module): model architecture

      Returns:
         flatt_children: List of model children layers
      """
    # get children form model!
      children = list(model.children())
      flatt_children = []
      if children == []:
         # if model has no children; model is last child! :O
         return model
      else:
         # look for children from children... to the last child!
         for child in children:
               try:
                  flatt_children.extend(ModelParser.get_children(child))
               except TypeError:
                  flatt_children.append(ModelParser.get_children(child))
      return flatt_children

   def _get_layers(self)-> dict:
      """Append flattened model children layers into a dictionary
      Args:
         self.children (list): List of model children layers
      Returns:
         architecture_dict (dict): Dictionary of model children layers
      """     
      architecture_dict = {}
      layer = 0
      for idx,module in enumerate(self.children):     
         architecture_dict[layer] =  str(module)
         layer += 1

      return architecture_dict

   def get_layer_signature(self,lines:str)-> list:
      """Function to iterate through string of model children layers to obtain functional signature of model children layers

      Args:
         lines (str): Model children layers

      Returns:
         dict_keys (list): functional signature of model children layers
      """
      # extracted = re.search("(\w+\(.+\))",lines)
      # lines = "torch.nn." + extracted.group()
      lines = "torch.nn." + lines
      print(lines)
      layer = eval(lines)
      layer_name=layer.__class__
      layer_signature=inspect.signature(layer_name).parameters
      layer_signature_keys = layer_signature.keys()
      dict_keys = []
      for i in layer_signature_keys:
         dict_keys.append(i)
      return dict_keys

   def _signature(self,header:str)->list:
      """Perform regex splitting to remove torch.nn functions ie. Conv2d, BatchNorm from model layers

      Args:
         header (str)): Model layers eg. Conv2d['kernel_size=(2, 2)', 'stride=(2, 2)', 'padding=(0, 0)']

      Returns:
         list : eg. ['kernel_size=(2, 2)', 'stride=(2, 2)', 'padding=(0, 0)']
      """
      # (?<=\() : positive lookbehind to match the open bracket (
      # [\w\W] : any of \w(word,digit,whitespace) or \W(NOT word,digit,whitespace)
      #   s = re.findall('(?<=\()[\w\W]+(?=\)$)', header)
      s = re.findall('(?<=\().+(?=\))', header)
      # print(f"SIGNATURE:{s}")
      # print(s) = ['3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)']
      # ,\s(?=\w\w) : match , followed by \s(whitespace) and then (?=\w\W) [any of \w(word,digit,whitespace) \W(NOT word,digit,whitespace)]

      return re.split(',\s(?=\w\w)', s[0]) if s else ''

   def _extract_params(self,signature:list):
      """Extraction params from model layers

      Args:
         signature (list): eg. ['kernel_size=(2, 2)', 'stride=(2, 2)', 'padding=(0, 0)']

      Returns:
         params (list): model params (integer, tuple, float) 
                        eg. ['64', '(1, 1)', '(1, 1)']
      """
      params = []
      for each in signature:
         # Note this MUST be .match. It will not work if it is .search
         integers = re.match("\d+",each)
         if integers:
            params.append(integers.group())
            # print(f"Digit:{integers}")
         else:
            regexed = re.search("(\d+\.\d+)|(\d+[e])(\+|\-)\d+|(?<=\=).+|(?<=\=)\(.+\)", each)
            params.append(regexed.group())
   
      return params
      
   # def _extract_params_backup(self,signature):
   #    params = []
   #    for each in signature:
   #       # Note this MUST be .match. It will not work if it is .search
   #       integers = re.match("\d+",each)
   #       if integers:
   #          params.append(integers.group())
   #          # print(f"Digit:{integers}")
   #       else:
   #          flt = re.search("(\d+\.\d+)",each)
   #          if flt:
   #             params.append(flt.group())
   #             # print(f"Float:{flt.group()}")
   #          else:
   #             scientific = re.search("(\d+[e])(\+|\-)\d+",each)
   #             if scientific:
   #                params.append(scientific.group())
   #                # print(f"Scientific:{scientific.group()}")
   #             else:
   #                # string includes booleans
   #                string = re.search("(?<=\=).+",each)
   #                if string:
   #                   params.append(string.group())
   #                   # print(f"string(incl bool): {string.group()}")
   #                else:
   #                   # get tuples
   #                   # Look for the = sign
   #                   # If the = sign exists, check for tuple
   #                   tuple = re.search("(?<=\=)\(.+\)",each)
   #                   if tuple:
   #                      params.append(tuple.group())
   #                      # print(tuple.group())
   #                      # print(f"Tuple: {tuple}")
   #    return params
   
   def parse_model(self)-> json:
      """Main function to parse model

      Returns:
         json.dumps(parsed_layers) (json): JSON dictionary of model layers
      """
      parsed_layers = []
      self.children = ModelParser.get_children(self.model)
      architecture = self._get_layers()
      for idx in architecture:
         param = {}
         module = {}
         layer_sig = self._signature(architecture[idx])
         param_value = self._extract_params(signature=layer_sig)
         param_name = self.get_layer_signature(architecture[idx])
         module_name = re.match("^\w+",architecture[idx]).group()
         params = zip(param_name,param_value)

         for name,value in params:
            param[name] = eval(value)

         for each in self.activations_list:
            if each in module_name.lower():
               module["activation"] = each
            else:
               module["activation"] = None
         if idx == 0:
            module["is_input"] = True
         
         module["l_type"] = module_name.lower()
         module["structure"] = param
         parsed_layers.append(module)
      
      return json.dumps(parsed_layers)
      
if __name__ == "__main__":
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
   # SIMULATE FOR RELAYNET
   model = ReLayNet(model_params)

   # SIMULATE FOR HRNET
   # configs = get_cfg_defaults()
   # model = HRNetRefine(configs)

   parser = ModelParser(model)
   layers_json = parser.parse_model()
   print(layers_json)
