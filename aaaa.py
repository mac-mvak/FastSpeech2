from hw_sg.utils.parse_config import ConfigParser
from hw_sg.utils.object_loading import get_dataloaders
import json


with open('/home/mac-mvak/code_disk/FastSpeech2/hw_sg/configs/config_fastspeech.json') as f:
    cfg = json.load(f)


cfg = ConfigParser(cfg)

dataloaders = get_dataloaders(cfg)

tr_d = dataloaders['train']

for batch in tr_d:
    print(batch[0].keys())
    break



