# Cross-Architecture Knowledge Distillation

This capstone research project studies how a Vision Transformer teacher can transfer features and predictions to a smaller MobileNetV2 student for plant leaf disease classification. It uses PyTorch, torchvision, and timm.

## Explore the project

- [`main.py`](main.py) combines the teacher, student, feature projectors, and loss functions into a training pipeline.
- [`dataset.py`](dataset.py) loads images organized by class and builds train, validation, and test splits.
- [`test.py`](test.py) evaluates a saved student checkpoint.
- The [Burmese grape experiments](https://github.com/zoe1410/capstone-knowledge-distillation/tree/Burmese_v4%28Strategy4%29) and [tomato leaf baselines](https://github.com/zoe1410/capstone-knowledge-distillation/tree/TomatoLeaf_VaibhavSolapure) retain their own evaluation code and setup notes.

## Run an experiment

Create a Python environment with a PyTorch build that suits your hardware, then install the project dependencies.

```sh
python -m pip install -r requirements.txt
```

Provide an image dataset with one subdirectory per class and a compatible five-class teacher checkpoint. Set the dataset directory in the configuration at the end of `main.py`, and set `checkpoint_path` where `TeacherExtractor` is initialized. Run `python main.py` from the repository root. To evaluate a trained student, match the dataset and class count used in training, set the student checkpoint path in `test.py`, and run `python test.py`.

The training inputs and model checkpoints are external artifacts. This portfolio edition preserves the experiment branches and excludes credentials, Python bytecode caches, and bundled copies of the reference paper.

## Reference

The feature projection approach is based on the paper by Liu et al. [Cross-Architecture Knowledge Distillation, ACCV 2022](https://arxiv.org/abs/2207.05273).
