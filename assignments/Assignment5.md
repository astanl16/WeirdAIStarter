# Teaching Weird AI to Write Lyrics

## Overview
In the previous lessons, you built the major components of the Weird AI language model. In this lesson, you will train those components together as a complete language model.

In this assignment, you will pretrain Weird AI from randomly initialized weights. You are not loading weights from an existing pretrained language model. The goal is to experience the complete pretraining process and produce Weird AI's own foundation model. Later in the course, you will work with pretrained language models and compare that workflow with training Weird AI from scratch.

This is an important transition in the project. Until now, the model architecture and its components existed, but the model's weights had not learned the patterns in the lyric dataset. During pretraining, Weird AI will learn by repeatedly predicting the next token and adjusting its weights based on the errors it makes.  The pretrained model you create in this lesson will become the foundation model used by later assignments.

By the end of this assignment you should be able to:
1. Generate text using Weird AI
2. Calculate cross-entropy loss
3. Measure training and validation performance
4. Implement a training loop
5. Observe loss reduction during training
6. Save model checkpoints

## Part 1: Application implementation

Read the application code and comments in the following files:
- ```losses.py```
- ```generation.py```
- ```trainer.py```

Complete the TODO tasks in each. Your implementation should support:
- Calculating training loss
- Calculating validation loss
- Training the model using backpropagation
- Tracking training progress
- Generating sample text during training
- Saving a training checkpoint
- Loading a previously saved checkpoint

Make sure all of the files pass the tests in ```test_losses.py``` and ```test_trainer.py```.  There are tests for each of the files you will work on in this lesson.

You can run the tests with the following command from the VS Code terminal:
```python -m pytest tests/<testfile>.py```

When all tests are passing be sure to check your code into your GitHub repository.

## Part 2: Create Your Pretrained Weird AI Checkpoint

The model you train in this lesson will be used as a starting point in later Weird AI assignments.

After completing your training implementation, train the Weird AI model and save the resulting model using the save_checkpoint() function you implemented in trainer.py.

Create the following directory if it does not already exist:

models/
  lesson-05-pretrained/

Save your checkpoint as: ```models/lesson-05-pretrained/checkpoint.pt```

Your checkpoint should contain:
- The model's learned weights
- The optimizer state
- The completed training epoch
- Training loss history
- Validation loss history
- The number of tokens seen during training

Verify Your Checkpoint

>Saving a file is not enough to demonstrate that the model can actually be restored.  Create a new Weird AI model and optimizer using the same configuration used for training.

Use your ```load_checkpoint()``` function to restore the saved checkpoint.

Then:
- Display the restored checkpoint metadata.
- Generate text from the restored model.
- Verify that the restored model produces reasonable output comparable to the model immediately after training.

> Note: You do not need to retrain the model after loading it.

**Important**: Do not overwrite this checkpoint in later assignments. This is your pretrained foundation model. Later assignments may load this checkpoint and create new models from it.

## Model State for This Lesson

- Starting model: Randomly initialized Weird AI model
- Does this lesson update model weights: Yes
- Model produced: Pretrained Weird AI foundation model
- Required checkpoint: models/lesson-05-pretrained/checkpoint.pt

> Later assignments will load this pretrained model rather than beginning again with randomly initialized weights.

## Part 3: Reflection Questions
1. Why does an untrained model generate nonsense text?
2. What is the difference between logits and probabilities?
3. Why is cross-entropy loss useful?
4. What does perplexity measure?
5. Why do we need both training and validation datasets?
6. What role does backpropagation play in learning?
7. Why should model checkpoints be saved during training?
