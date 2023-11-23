# FastSpeech project barebones

## Installation guide

Copy this repo.

```shell
pip install -r ./requirements.txt
gdown https://drive.google.com/uc?id=1SzRWUMCG6LaiyM9VYtmBe4TW5itennWP -O final_data/model.pth
bash gen_scripts/downloader
python3 gen_scripts/energy_generation.py
python3 gen_scripts/pitch_generation.py
```

Model is located in directory `hw_sg/model/gen_model` and consists out of three blocks.
`transformer.py` for transformer blocks, `predictors.py` for variance adapter and
`FastSpeech.py` for final model.

We also modified `trainer.py`, `ljspeech_dataset.py` and `collate.py` in order to make our model work.

Also we have some generation scripts in directory `gen_scripts` and we shall run them on startup.

`final_data/config.json` is a trainer and tester config. 

In order to test our model you should call

```shell
python3 test.py -c final_data/config.json -r final_data/model.pth
```

Needed texts and energies can be changed manually in `test.py`. Results will be saved to results folder.

Training is standard with `train.py`.

## Wandb Report and Samples

[Link to audio Samples](https://drive.google.com/drive/folders/15gv8mJdrGPHJLd2sm51cDmKfbYxjKdsz?usp=sharing)

[Link to report](https://api.wandb.ai/links/svak/yr86fuax)

