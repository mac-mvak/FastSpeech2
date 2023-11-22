from hw_sg.utils.parse_config import ConfigParser
from hw_sg.utils.object_loading import get_dataloaders
import hw_sg.model as module_arch

import json


with open('/home/mac-mvak/code_disk/FastSpeech2/hw_sg/configs/config_fastspeech.json') as f:
    cfg = json.load(f)


cfg = ConfigParser(cfg)

dataloaders = get_dataloaders(cfg)

tr_d = dataloaders['train']

for batch in tr_d:
    print(batch[0].keys())
    break

model = cfg.init_obj(cfg["arch"], module_arch)

for u in model.parameters():
    print('aaa')
    break